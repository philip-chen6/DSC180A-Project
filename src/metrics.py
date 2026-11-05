# Owner: Person 1
"""Result persistence and baseline/speculative comparison plots."""


# Append one run record to the results file.
def save_results(stats: dict, path: str) -> None:
    """Append stats to a JSON list at path, preserving existing records.

    Create the parent directory and file if missing. Each record must
    follow this schema, with Python None serialized as JSON null:

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

    Baseline acceptance metrics are null; accuracy is null unless the
    run was evaluated. Report malformed existing JSON instead of silently
    replacing it. The caller provides the full run record.
    """
    raise NotImplementedError("metrics.save_results is not implemented yet (Person 1).")


# Plot throughput and acceptance length for both generation modes.
def plot_comparison(results_path: str, out_path: str) -> None:
    """Read saved records and write a baseline/specdec comparison image.

    Plot bar charts for tokens_per_sec and mean_acceptance_length with
    labeled units. Represent null acceptance values as unavailable, not
    zero. Document how repeated runs are selected or aggregated when
    implemented. Create the output parent directory if needed and save
    the figure to out_path. Import matplotlib inside this function so
    importing the module does not require plotting dependencies.
    """
    raise NotImplementedError("metrics.plot_comparison is not implemented yet (Person 1).")
