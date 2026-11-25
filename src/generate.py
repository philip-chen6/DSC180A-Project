# Owner: Philip
"""Baseline and speculative generation through vLLM."""

from time import perf_counter
from typing import Any
import os


DRAFT_CONFIG_KEYS = ("draft_model", "num_speculative_tokens", "draft_sample_method",
                     "draft_quantization")


# Names returned by vLLM's V1 metrics reader (without Prometheus's _total suffix).
SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


# Read the counters exposed by the runtime rather than inventing missing values.
def _spec_counters(llm: Any) -> dict[str, float]:
    """Read V1 Counter.name/value objects, summing across engine labels.

    API source: vllm/v1/metrics/reader.py and the official offline speculative
    decoding example. Unsupported APIs yield no counters; other errors surface.
    """
    get_metrics = getattr(llm, "get_metrics", None)
    if get_metrics is None:
        print("LLM.get_metrics() unavailable; acceptance metrics will be null.")
        return {}
    try:
        metrics = get_metrics()
    except NotImplementedError:
        print("LLM.get_metrics() unsupported; acceptance metrics will be null.")
        return {}

    counters = {}
    names = sorted({metric.name for metric in metrics if "spec_decode" in metric.name})
    print(f"LLM.get_metrics() speculative names: {', '.join(names) or '(none)'}")
    for metric in metrics:
        if metric.name in SPEC_COUNTERS:
            counters[metric.name] = counters.get(metric.name, 0.0) + metric.value
    return counters


# Generate completions and collect run statistics.
def run_generation(prompts: list[str], config: dict) -> tuple[list[str], dict]:
    """Generate one completion per prompt using the supplied JSON config.

    Use target_model, temperature, top_p (default 1.0), max_tokens, and seed. If draft_model
    is present, enable speculative decoding with num_speculative_tokens and
    optionally draft_sample_method (vLLM 0.32.0 defaults to greedy proposals)
    and draft_quantization (passed to vLLM's speculative_config.quantization;
    omit it to let vLLM detect AWQ/GPTQ checkpoints); otherwise run the baseline.
    Optional quantization applies to the target. Optional lora_adapter (a
    local PEFT adapter folder, e.g. a GRPO checkpoint) serves the target as the
    LoRA policy, with max_lora_rank (default 16). Return completions in input order and a
    statistics record matching this schema:

        {
            "config": "baseline" | "specdec" | "baseline_sample" | "specdec_sample",
            "n_prompts": int,
            "tokens_per_sec": float,
            "mean_latency_s": float,
            "mean_output_len": float,
            "mean_acceptance_length": float | null,
            "acceptance_rate": float | null,
            "accuracy": float | null
        }

    Set config from the presence of draft_model, appending _sample when
    temperature is positive, and set n_prompts from the
    actual prompt count. Run one prompt per blocking LLM.generate call.
    Each latency measures wall time around that call, including tokenization,
    prefill, decoding, and output processing, but excluding model initialization
    and metric reads. Throughput is output token count divided by the sum of
    those latencies: concurrency one, not batched rollout throughput. There is
    no separate warmup; any first-call overhead is included in both modes.

    Print the installed vLLM version and the actual get_metrics() counter
    names. Use before/after counter deltas for this run. Mean acceptance length
    is 1 + accepted_tokens / num_drafts (includes the bonus/correction token);
    acceptance_rate is accepted_tokens / draft_tokens, a fraction in [0, 1].
    Use None (JSON null) for baseline acceptance metrics and for metrics
    unavailable from vLLM or with zero denominators. Leave accuracy as None.

    Reject an empty prompt list. Import vllm lazily; importing this module
    does not download weights or initialize a GPU. The V1 metrics API was
    validated with vLLM 0.32.0 on the team's RTX 5070 GPU environment.
    """
    if not prompts:
        raise ValueError("run_generation requires at least one prompt.")

    import vllm
    from vllm import LLM, SamplingParams

    print(f"vLLM version: {vllm.__version__}", flush=True)
    print(f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT', '0')}", flush=True)
    speculative = "draft_model" in config
    llm_args = {
        "model": config["target_model"],
        "seed": config["seed"],
        "tensor_parallel_size": 1,
        "disable_log_stats": False,
    }
    for option in ("gpu_memory_utilization", "max_model_len", "max_num_seqs", "enforce_eager",
                   "quantization"):
        if option in config:
            llm_args[option] = config[option]
    if speculative:
        llm_args["speculative_config"] = {
            "model": config["draft_model"],
            "num_speculative_tokens": config["num_speculative_tokens"],
        }
        if "draft_sample_method" in config:
            llm_args["speculative_config"]["draft_sample_method"] = config["draft_sample_method"]
        if "draft_quantization" in config:
            llm_args["speculative_config"]["quantization"] = config["draft_quantization"]
        print(f"Draft sampling method: {config.get('draft_sample_method', 'greedy')}", flush=True)
    generate_kwargs = {"use_tqdm": False}
    if config.get("lora_adapter"):
        from vllm.lora.request import LoRARequest

        adapter = os.path.abspath(config["lora_adapter"])
        if not os.path.isdir(adapter):
            raise RuntimeError(f"lora_adapter folder not found: {adapter}")
        llm_args["enable_lora"] = True
        llm_args["max_lora_rank"] = config.get("max_lora_rank", 16)
        generate_kwargs["lora_request"] = LoRARequest("policy", 1, adapter)
        print(f"LoRA policy adapter: {adapter}", flush=True)
        if speculative:
            # vLLM attaches the LoRARequest to the target; whether the draft
            # also runs it is version dependent. Self-speculation needs the
            # draft to share the adapter for high acceptance, and correctness
            # needs the target to apply it: validate with `run.py check`.
            print("Note: verify that the draft shares the LoRA adapter (compare this "
                  "acceptance_rate with `run.py acceptance`) and that the target applies "
                  "it (`run.py check` must report 0 differences).", flush=True)
    llm = LLM(**llm_args)
    sampling = SamplingParams(
        temperature=config["temperature"],
        top_p=config.get("top_p", 1.0),
        max_tokens=config["max_tokens"],
        seed=config["seed"],
        n=1,
    )
    before = _spec_counters(llm) if speculative else {}
    completions = []
    latencies = []
    output_lengths = []
    for index, prompt in enumerate(prompts, start=1):
        start = perf_counter()
        outputs = llm.generate([prompt], sampling, **generate_kwargs)
        latencies.append(perf_counter() - start)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("Expected exactly one completion per prompt.")
        completion = outputs[0].outputs[0]
        completions.append(completion.text)
        output_lengths.append(len(completion.token_ids))
        if index % 10 == 0 or index == len(prompts):
            print(f"Generation progress: {index}/{len(prompts)} prompts", flush=True)

    mean_acceptance_length = None
    acceptance_rate = None
    if speculative:
        after = _spec_counters(llm)
        if all(name in after for name in SPEC_COUNTERS):
            print(f"Acceptance counter names used: {', '.join(SPEC_COUNTERS)}")
            drafts, proposed, accepted = (
                after[name] - before.get(name, 0.0) for name in SPEC_COUNTERS
            )
            if min(drafts, proposed, accepted) < 0 or accepted > proposed:
                raise RuntimeError("Speculative counters reset or are inconsistent.")
            if drafts > 0:
                mean_acceptance_length = 1.0 + accepted / drafts
            if proposed > 0:
                acceptance_rate = accepted / proposed
        else:
            print("Acceptance counter names used: none (required counters unavailable).")
        print(f"Acceptance metrics: mean_acceptance_length={mean_acceptance_length}, "
              f"acceptance_rate={acceptance_rate}", flush=True)
    else:
        print("Acceptance counter names used: none (baseline).")

    elapsed = sum(latencies)
    label = "specdec" if speculative else "baseline"
    if config["temperature"] > 0:
        label += "_sample"
    stats = {
        "config": label,
        "n_prompts": len(prompts),
        "tokens_per_sec": sum(output_lengths) / elapsed,
        "mean_latency_s": elapsed / len(prompts),
        "mean_output_len": sum(output_lengths) / len(prompts),
        "mean_acceptance_length": mean_acceptance_length,
        "acceptance_rate": acceptance_rate,
        "accuracy": None,
    }
    return completions, stats
