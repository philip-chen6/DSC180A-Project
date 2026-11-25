# Owner: TBD (self-speculative draft study)
"""Quantized self-speculation: a frozen low-bit copy of the base model plus the
policy's live LoRA adapter as the draft model.

During LoRA RL training the policy is ``W + BA`` (bf16 base ``W``, adapter
``BA``). The proposed draft is ``Q(W) + BA``: the same base, quantized once and
kept frozen, sharing whatever adapter the policy currently has. The draft then
tracks the policy by construction, so the remaining draft/target mismatch is
quantization error rather than training staleness.

This module measures that design without depending on speculative-decoding
kernels:

* ``rtn_quantize_model_`` simulates an N-bit round-to-nearest (RTN) frozen
  base. Weights are quantized then stored dequantized, so accuracy matches a
  real RTN kernel while memory/speed do not (speed is measured separately in
  vLLM by ``run.py draftcost``).
* ``measure_acceptance`` teacher-forces policy rollouts through the target and
  a draft and reports expected per-token acceptance. Acceptance is a property
  of the two distributions, so it does not need a fast engine.
* ``expected_speedup`` combines acceptance with the measured draft cost ratio.
* ``run_acceptance_study`` sweeps saved GRPO adapter checkpoints and draft
  variants (live adapter, no adapter, periodically refreshed adapter, other
  bit-widths, a separate small model) and writes JSON plus a plot.

Torch, transformers, and peft are imported lazily so importing this module on
a CPU-only machine without them is safe.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from src.data import format_prompt, load_gsm8k
from src.reward import reward


# Base-weight treatments for a draft. "rtnN" simulates N-bit round-to-nearest;
# "nf4" uses bitsandbytes NF4 (QLoRA-style); "none" keeps bf16 weights.
QUANT_PATTERN = re.compile(r"^(none|nf4|rtn([2-8]))$")

# Linear layers left in bf16 by default: the (often tied) output head is
# usually unquantized in 4-bit checkpoints, and quantizing it distorts logits.
DEFAULT_SKIP_MODULES = ("lm_head",)

# How the draft acceptance is computed under sampling.
#   probabilistic: draft samples from q; acceptance = sum_x min(p(x), q(x)).
#   greedy: draft proposes argmax q; acceptance = p(argmax q).
DRAFT_SAMPLING = ("probabilistic", "greedy")

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------

def parse_quant(quant: str) -> tuple[str, int | None]:
    """Return ("none"|"nf4"|"rtn", bits) for a quant string such as "rtn4"."""
    match = QUANT_PATTERN.match(quant) if isinstance(quant, str) else None
    if match is None:
        raise ValueError(f"quant must be 'none', 'nf4', or 'rtn2'..'rtn8', got {quant!r}.")
    if match.group(2):
        return "rtn", int(match.group(2))
    return match.group(1), None


def rtn_quantize_weight(weight: Any, bits: int, group_size: int | None = 128) -> Any:
    """Return an asymmetric round-to-nearest quantize-dequantize copy of ``weight``.

    ``weight`` is a 2-D [out_features, in_features] tensor. Each row is split
    into groups of ``group_size`` input columns (one group per row when
    ``group_size`` is None or does not divide ``in_features``). Each group gets
    its own scale and integer zero point over ``2**bits`` levels, matching the
    usual RTN/GPTQ-without-calibration scheme. The result has the input's dtype
    and shape; every group's values lie on its own ``2**bits``-level grid.
    """
    import torch

    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}.")
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError(f"bits must be an int in [2, 8], got {bits!r}.")
    out_features, in_features = weight.shape
    group = group_size if group_size and in_features % group_size == 0 else in_features

    w = weight.detach().to(torch.float32).reshape(out_features, in_features // group, group)
    w_min = w.amin(dim=-1, keepdim=True)
    w_max = w.amax(dim=-1, keepdim=True)
    q_max = 2 ** bits - 1
    scale = ((w_max - w_min) / q_max).clamp(min=1e-10)
    zero = torch.round(-w_min / scale)
    q = torch.clamp(torch.round(w / scale) + zero, 0, q_max)
    return ((q - zero) * scale).reshape(out_features, in_features).to(weight.dtype)


def rtn_quantize_model_(model: Any, bits: int, group_size: int | None = 128,
                        skip: Iterable[str] = DEFAULT_SKIP_MODULES) -> int:
    """Quantize every ``nn.Linear`` weight of ``model`` in place; return the count.

    Call this on the *base* model before attaching a LoRA adapter, so the
    adapter's A/B matrices stay in full precision (the proposed design).
    Modules whose qualified name contains any string in ``skip`` are left alone.
    Embeddings are not ``nn.Linear`` and are never touched.
    """
    import torch

    skip = tuple(skip)
    count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not any(s in name for s in skip):
                module.weight.copy_(rtn_quantize_weight(module.weight, bits, group_size))
                count += 1
    if count == 0:
        raise RuntimeError("rtn_quantize_model_ found no Linear layers to quantize.")
    return count


def load_model(model_name: str, quant: str = "none", adapter: str | None = None,
               device: str = "cuda", group_size: int | None = 128) -> Any:
    """Load a causal LM with an optional quantized base and optional LoRA adapter.

    The adapter is attached *after* quantizing the base and is never merged,
    so the draft is exactly Q(W) + BA with BA in full precision.
    """
    import torch
    from transformers import AutoModelForCausalLM

    kind, bits = parse_quant(quant)
    kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if kind == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,
        )
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if kind != "nf4":
        model = model.to(device)
    if kind == "rtn":
        count = rtn_quantize_model_(model, bits, group_size)
        print(f"RTN-{bits} quantized {count} Linear layers of {model_name} "
              f"(group size {group_size}).", flush=True)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Acceptance and speedup
# --------------------------------------------------------------------------

def token_acceptance(target_logits: Any, draft_logits: Any, temperature: float,
                     draft_sampling: str = "probabilistic") -> Any:
    """Return expected acceptance probability per position, shape [..., ].

    With ``temperature == 0`` both models decode greedily and a draft token is
    accepted exactly when the two argmaxes agree. With sampling, target
    distribution p and draft distribution q are softmax(logits / T):

    * probabilistic proposals (draft samples x ~ q, standard rejection
      sampling): P(accept) = sum_x min(p(x), q(x)) = 1 - TV(p, q).
    * greedy proposals (draft proposes argmax q, as vLLM does by default):
      P(accept) = p(argmax q).

    Logits must share their last (vocabulary) dimension.
    """
    import torch

    if target_logits.shape != draft_logits.shape:
        raise ValueError(f"logit shapes differ: {tuple(target_logits.shape)} vs "
                         f"{tuple(draft_logits.shape)}.")
    if draft_sampling not in DRAFT_SAMPLING:
        raise ValueError(f"draft_sampling must be one of {DRAFT_SAMPLING}.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if temperature == 0:
        return (target_logits.argmax(-1) == draft_logits.argmax(-1)).to(torch.float32)
    p = torch.softmax(target_logits.to(torch.float32) / temperature, dim=-1)
    if draft_sampling == "greedy":
        return p.gather(-1, draft_logits.argmax(-1, keepdim=True)).squeeze(-1)
    q = torch.softmax(draft_logits.to(torch.float32) / temperature, dim=-1)
    return torch.minimum(p, q).sum(-1)


def expected_tokens_per_step(alpha: float, k: int) -> float:
    """Expected tokens emitted per verification step with ``k`` draft tokens.

    Uses the i.i.d. acceptance model of Leviathan et al. (2023):
    (1 - alpha**(k+1)) / (1 - alpha), which equals k + 1 when alpha = 1.
    Includes the bonus/correction token, like vLLM's mean acceptance length.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}.")
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """Predicted speedup over plain decoding.

    ``cost_ratio`` (c) is the time of one draft forward step divided by one
    target step. One speculative step costs about k*c + 1 target steps and
    yields ``expected_tokens_per_step`` tokens. Values below 1 are slowdowns.
    Ignores verification overhead beyond one target step, so treat it as an
    upper-level estimate to compare drafts, not a replacement for timing.
    """
    if cost_ratio < 0:
        raise ValueError(f"cost_ratio must be non-negative, got {cost_ratio}.")
    return expected_tokens_per_step(alpha, k) / (k * cost_ratio + 1.0)


def _bootstrap_ci(accepted: list[float], counts: list[int], seed: int,
                  n_boot: int = 1000) -> tuple[float, float] | None:
    """95% percentile bootstrap CI of token-weighted acceptance over sequences."""
    if len(counts) < 2:
        return None
    rng = random.Random(seed)
    n = len(counts)
    estimates = []
    for _ in range(n_boot):
        picks = [rng.randrange(n) for _ in range(n)]
        total = sum(counts[i] for i in picks)
        if total:
            estimates.append(sum(accepted[i] for i in picks) / total)
    estimates.sort()
    return estimates[int(0.025 * len(estimates))], estimates[int(0.975 * len(estimates)) - 1]


def measure_acceptance(target: Any, draft: Any, sequences: list[tuple[list[int], list[int]]],
                       temperature: float, draft_sampling: str = "probabilistic",
                       device: str = "cuda", seed: int = 0) -> dict:
    """Teacher-force (prompt_ids, completion_ids) pairs through both models.

    For every completion token position, compare the target's and draft's
    next-token distributions given the same policy-generated prefix (this is
    the context in which a real draft would be verified). Return:

        {"alpha": token-weighted mean acceptance,
         "alpha_ci95": [lo, hi] | None (bootstrap over sequences),
         "greedy_agreement": fraction of positions where argmaxes agree,
         "n_tokens": int, "n_sequences": int}

    Sequences are processed one at a time with ``logits_to_keep`` so only the
    completion positions' full-vocabulary logits are materialized.
    """
    import torch

    accepted, agreed, counts = [], [], []
    with torch.no_grad():
        for prompt_ids, completion_ids in sequences:
            if not prompt_ids or not completion_ids:
                continue
            ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
            keep = len(completion_ids) + 1
            # Positions len(prompt)-1 .. len-2 predict the completion tokens.
            target_logits = target(input_ids=ids, logits_to_keep=keep).logits[0, :-1]
            draft_logits = draft(input_ids=ids, logits_to_keep=keep).logits[0, :-1]
            if target_logits.shape[-1] != draft_logits.shape[-1]:
                vocab = min(target_logits.shape[-1], draft_logits.shape[-1])
                target_logits, draft_logits = target_logits[..., :vocab], draft_logits[..., :vocab]
            per_token = token_acceptance(target_logits, draft_logits, temperature, draft_sampling)
            accepted.append(float(per_token.sum()))
            agreed.append(float((target_logits.argmax(-1) == draft_logits.argmax(-1)).sum()))
            counts.append(int(per_token.numel()))
    total = sum(counts)
    if total == 0:
        raise ValueError("measure_acceptance received no non-empty sequences.")
    ci = _bootstrap_ci(accepted, counts, seed)
    return {
        "alpha": sum(accepted) / total,
        "alpha_ci95": list(ci) if ci else None,
        "greedy_agreement": sum(agreed) / total,
        "n_tokens": total,
        "n_sequences": len(counts),
    }


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------

def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    return set(eos if isinstance(eos, (list, tuple)) else [eos])


def sample_rollouts(model: Any, tokenizer: Any, prompts: list[str], temperature: float,
                    max_tokens: int, seed: int, batch_size: int = 8,
                    device: str = "cuda") -> list[tuple[list[int], list[int]]]:
    """Sample one completion per prompt from ``model`` (the current policy).

    Prompts are already chat-formatted, so no special tokens are added. Each
    completion is cut after its first EOS token (kept, since a real draft also
    has to predict it) and padding is dropped.
    """
    import torch

    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = _eos_ids(model, tokenizer)
    results = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
        generate_args = {"max_new_tokens": max_tokens, "pad_token_id": tokenizer.pad_token_id}
        if temperature > 0:
            generate_args.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            generate_args.update(do_sample=False)
        with torch.no_grad():
            output = model.generate(**encoded, **generate_args)
        width = encoded["input_ids"].shape[1]
        for row, mask in enumerate(encoded["attention_mask"]):
            prompt_ids = encoded["input_ids"][row][mask.bool()].tolist()
            completion = []
            for token in output[row, width:].tolist():
                completion.append(token)
                if token in eos_ids:
                    break
            results.append((prompt_ids, completion))
        print(f"Rollouts: {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return results


# --------------------------------------------------------------------------
# Checkpoints and draft variants
# --------------------------------------------------------------------------

def discover_checkpoints(adapter_dir: str | None, steps: list[int] | None = None
                         ) -> list[tuple[int, str | None]]:
    """Return [(step, adapter_path)] sorted by step, starting with (0, None).

    Step 0 is the untrained policy (no adapter). ``adapter_dir`` holds TRL
    ``checkpoint-<step>`` folders; a missing directory yields only step 0.
    ``steps`` optionally restricts the result to those steps.
    """
    found = [(0, None)]
    if adapter_dir is not None and Path(adapter_dir).is_dir():
        for child in Path(adapter_dir).iterdir():
            match = CHECKPOINT_PATTERN.match(child.name)
            if match and child.is_dir() and (child / "adapter_config.json").exists():
                found.append((int(match.group(1)), str(child)))
    found.sort(key=lambda item: item[0])
    if steps is not None:
        wanted = set(steps)
        missing = wanted - {step for step, _ in found}
        if missing:
            raise ValueError(f"requested checkpoint steps not found: {sorted(missing)}")
        found = [item for item in found if item[0] in wanted]
    return found


def validate_variant(variant: dict, target_model: str) -> None:
    """Reject draft variants that cannot share the policy adapter."""
    if not isinstance(variant.get("name"), str) or not variant["name"]:
        raise ValueError(f"every variant needs a non-empty name: {variant!r}")
    parse_quant(variant.get("quant", "none"))
    adapter = variant.get("adapter", "live")
    if isinstance(adapter, dict):
        every = adapter.get("refresh_every")
        if not isinstance(every, int) or every < 1:
            raise ValueError(f"{variant['name']}: refresh_every must be a positive int.")
    elif adapter not in ("live", "none"):
        raise ValueError(f"{variant['name']}: adapter must be 'live', 'none', "
                         "or {'refresh_every': N}.")
    if variant.get("model", target_model) != target_model and adapter != "none":
        raise ValueError(f"{variant['name']}: a different draft architecture cannot "
                         "load the policy LoRA adapter; use adapter 'none'.")


def draft_adapter(variant: dict, step: int, checkpoints: list[tuple[int, str | None]]
                  ) -> str | None:
    """Return the adapter path the draft uses when the policy is at ``step``.

    * "live": the policy's current adapter (the proposed design).
    * "none": no adapter, i.e. the frozen step-0 policy (fully stale).
    * {"refresh_every": N}: the newest checkpoint at or before the last refresh
      boundary floor(step / N) * N, emulating a draft that syncs every N steps.
    """
    adapter = variant.get("adapter", "live")
    if adapter == "none":
        return None
    boundary = step if adapter == "live" else (step // adapter["refresh_every"]) * adapter["refresh_every"]
    eligible = [path for ckpt_step, path in checkpoints if ckpt_step <= boundary]
    return eligible[-1] if eligible else None


# --------------------------------------------------------------------------
# Study driver and plot
# --------------------------------------------------------------------------

def _cost_ratios(config: dict) -> dict[str, float]:
    """Read measured cost ratios from the draftcost output, if configured."""
    speedup = config.get("speedup") or {}
    ratios = dict(speedup.get("cost_ratio", {}))
    path = speedup.get("cost_ratio_path")
    if path and Path(path).exists():
        measured = json.loads(Path(path).read_text(encoding="utf-8")).get("cost_ratio", {})
        for variant in config["variants"]:
            key = variant.get("cost_key")
            if key in measured and variant["name"] not in ratios:
                ratios[variant["name"]] = measured[key]
    return ratios


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_acceptance_study(config: dict, root: Path = Path(".")) -> dict:
    """Measure draft acceptance for every (checkpoint, variant) pair.

    For each saved policy checkpoint: load the policy, sample rollouts on held-out
    GSM8K test questions, score their accuracy, then load each draft variant and
    teacher-force the rollouts through both. Results are rewritten to
    ``results_path`` after every record so an interrupted run keeps its data.
    """
    import torch
    from transformers import AutoTokenizer

    target_model = config["target_model"]
    for variant in config["variants"]:
        validate_variant(variant, target_model)
    names = [variant["name"] for variant in config["variants"]]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique.")
    temperature = config["temperature"]
    draft_sampling = config.get("draft_sampling", "probabilistic")
    device = config.get("device", "cuda")
    group_size = config.get("group_size", 128)
    adapter_dir = config.get("adapter_dir")
    adapter_dir = str(root / adapter_dir) if adapter_dir else None
    checkpoints = discover_checkpoints(adapter_dir, config.get("steps"))
    print(f"Policy checkpoints: {[step for step, _ in checkpoints]}", flush=True)
    ratios = _cost_ratios(config)
    ks = (config.get("speedup") or {}).get("ks", [1, 2, 3, 5])

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    examples = load_gsm8k("test", config["n_prompts"], config["seed"])
    prompts = [format_prompt(example["question"], tokenizer) for example in examples]
    results_path = root / config["results_path"]
    payload = {"config": config, "records": []}

    for step, policy_adapter in checkpoints:
        print(f"== Policy step {step} (adapter: {policy_adapter}) ==", flush=True)
        target = load_model(target_model, "none", policy_adapter, device)
        rollouts = sample_rollouts(target, tokenizer, prompts, temperature,
                                   config["max_tokens"], config["seed"] + step,
                                   config.get("batch_size", 8), device)
        texts = [tokenizer.decode(completion, skip_special_tokens=True)
                 for _, completion in rollouts]
        accuracy = sum(reward(text, example["answer"])
                       for text, example in zip(texts, examples, strict=True)) / len(examples)
        print(f"Policy accuracy at step {step}: {accuracy:.3f}", flush=True)

        for variant in config["variants"]:
            adapter = draft_adapter(variant, step, checkpoints)
            draft_model = variant.get("model", target_model)
            draft = load_model(draft_model, variant.get("quant", "none"), adapter,
                               device, group_size)
            measured = measure_acceptance(target, draft, rollouts, temperature,
                                          draft_sampling, device, config["seed"])
            record = {"step": step, "variant": variant["name"], "draft_model": draft_model,
                      "quant": variant.get("quant", "none"), "draft_adapter": adapter,
                      "policy_adapter": policy_adapter, "policy_accuracy": accuracy,
                      "temperature": temperature, "draft_sampling": draft_sampling,
                      **measured}
            if variant["name"] in ratios:
                c = ratios[variant["name"]]
                alpha = min(max(measured["alpha"], 0.0), 1.0)
                record["cost_ratio"] = c
                record["predicted_speedup"] = {str(k): expected_speedup(alpha, k, c) for k in ks}
            payload["records"].append(record)
            _write(results_path, payload)
            print(f"  {variant['name']}: alpha={measured['alpha']:.4f} "
                  f"greedy_agree={measured['greedy_agreement']:.4f} "
                  f"(draft adapter: {adapter})", flush=True)
            del draft
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Acceptance study saved to {results_path}")
    if config.get("plot_path"):
        plot_acceptance(str(results_path), str(root / config["plot_path"]))
    return payload


def plot_acceptance(results_path: str, out_path: str) -> None:
    """Plot acceptance vs. policy step per variant, with policy accuracy below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = json.loads(Path(results_path).read_text(encoding="utf-8"))["records"]
    if not records:
        raise ValueError(f"{results_path} has no records to plot.")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    for name in dict.fromkeys(record["variant"] for record in records):
        rows = sorted((r for r in records if r["variant"] == name), key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        top.plot(steps, [r["alpha"] for r in rows], marker="o", label=name)
        lows = [r["alpha_ci95"][0] if r["alpha_ci95"] else r["alpha"] for r in rows]
        highs = [r["alpha_ci95"][1] if r["alpha_ci95"] else r["alpha"] for r in rows]
        top.fill_between(steps, lows, highs, alpha=0.15)
    top.set_ylabel("Expected per-token acceptance")
    top.set_title("Draft acceptance as the LoRA policy trains")
    top.grid(alpha=0.3)
    top.legend(fontsize=8)
    policy = {r["step"]: r["policy_accuracy"] for r in records}
    bottom.plot(sorted(policy), [policy[s] for s in sorted(policy)], marker="o", color="black")
    bottom.set_ylabel("Policy accuracy")
    bottom.set_xlabel("GRPO step")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Acceptance plot saved to {out_path}")
