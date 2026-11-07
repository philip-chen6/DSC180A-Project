# Owner: Philip
"""Baseline and speculative generation through vLLM."""

from time import perf_counter
from typing import Any


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

    Use target_model, temperature, max_tokens, and seed. If draft_model
    is present, enable speculative decoding with num_speculative_tokens;
    otherwise run the baseline. Return completions in input order and a
    statistics record matching this schema:

        {
            "config": "baseline" | "specdec",
            "n_prompts": int,
            "tokens_per_sec": float,
            "mean_latency_s": float,
            "mean_output_len": float,
            "mean_acceptance_length": float | null,
            "acceptance_rate": float | null,
            "accuracy": float | null
        }

    Set config from the presence of draft_model and n_prompts from the
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
    checked against upstream source; it still requires validation against
    the team's installed GPU environment.
    """
    if not prompts:
        raise ValueError("run_generation requires at least one prompt.")

    import vllm
    from vllm import LLM, SamplingParams

    print(f"vLLM version: {vllm.__version__}", flush=True)
    speculative = "draft_model" in config
    llm_args = {
        "model": config["target_model"],
        "seed": config["seed"],
        "tensor_parallel_size": 1,
        "disable_log_stats": False,
    }
    if speculative:
        llm_args["speculative_config"] = {
            "model": config["draft_model"],
            "num_speculative_tokens": config["num_speculative_tokens"],
        }
    llm = LLM(**llm_args)
    sampling = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        seed=config["seed"],
        n=1,
    )
    before = _spec_counters(llm) if speculative else {}
    completions = []
    latencies = []
    output_lengths = []
    for prompt in prompts:
        start = perf_counter()
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        latencies.append(perf_counter() - start)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("Expected exactly one completion per prompt.")
        completion = outputs[0].outputs[0]
        completions.append(completion.text)
        output_lengths.append(len(completion.token_ids))

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
    else:
        print("Acceptance counter names used: none (baseline).")

    elapsed = sum(latencies)
    stats = {
        "config": "specdec" if speculative else "baseline",
        "n_prompts": len(prompts),
        "tokens_per_sec": sum(output_lengths) / elapsed,
        "mean_latency_s": elapsed / len(prompts),
        "mean_output_len": sum(output_lengths) / len(prompts),
        "mean_acceptance_length": mean_acceptance_length,
        "acceptance_rate": acceptance_rate,
        "accuracy": None,
    }
    return completions, stats
