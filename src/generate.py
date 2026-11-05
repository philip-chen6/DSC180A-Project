# Owner: Philip
"""Baseline and speculative generation through vLLM."""


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
    actual prompt count. Measure generated output tokens per second,
    latency in seconds, and output length in tokens. Document timing
    boundaries when implemented so both modes use the same measurement.
    Use None (JSON null) for baseline acceptance metrics and for metrics
    unavailable from vLLM; never substitute a fabricated zero. Leave
    accuracy as None until evaluation fills it in.

    Import vllm inside this function when implemented. Module imports
    must not load weights, download models, or initialize a GPU.
    """
    raise NotImplementedError("generate.run_generation is not implemented yet (Philip).")
