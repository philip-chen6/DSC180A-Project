# Owner: Maggie
"""Result persistence and baseline/speculative comparison plots."""

import json
import os
from typing import Dict, Any


def save_results(stats: Dict[str, Any], path: str) -> None:
    """Append stats to a JSON list at path, preserving existing records."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Existing JSON at {path} is not a list.")
            except json.JSONDecodeError:
                raise ValueError(f"Malformed JSON in {path}.")
    else:
        data = []

    # Enforce strict schema from the requirements
    required_keys = {
        "config": str,
        "n_prompts": int,
        "tokens_per_sec": (float, int),
        "mean_latency_s": (float, int),
        "mean_output_len": (float, int),
        "mean_acceptance_length": (float, int, type(None)),
        "acceptance_rate": (float, int, type(None)),
        "accuracy": (float, int, type(None))
    }

    for key, expected_type in required_keys.items():
        if key not in stats:
            raise KeyError(f"Missing required key '{key}' in stats.")
        if stats[key] is not None and not isinstance(stats[key], expected_type):
            raise TypeError(
                f"Key '{key}' should be {expected_type}, got {type(stats[key])}")

    data.append(stats)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def plot_comparison(results_path: str, out_path: str) -> None:
    """Read saved records and write a baseline/specdec comparison image."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.exists(results_path):
        print(f"Error: Results file {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("No data to plot.")
        return

    # Grab the latest run for each config
    latest_runs = {run["config"]: run for run in data}
    configs = list(latest_runs.keys())

    tokens_per_sec = [latest_runs[c].get("tokens_per_sec", 0) for c in configs]

    # Handle null acceptance lengths safely (treat as 0 for plotting the baseline)
    acc_lengths = []
    for c in configs:
        val = latest_runs[c].get("mean_acceptance_length")
        acc_lengths.append(val if val is not None else 0.0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.35

    # Plot tokens/sec on primary y-axis
    color_tps = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Tokens / Sec', color=color_tps)
    bars1 = ax1.bar(x - width/2, tokens_per_sec, width,
                    label='Tokens / Sec', color=color_tps)
    ax1.tick_params(axis='y', labelcolor=color_tps)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)

    # Plot mean acceptance length on secondary y-axis
    ax2 = ax1.twinx()
    color_acc = 'tab:orange'
    ax2.set_ylabel('Mean Acceptance Length', color=color_acc)
    bars2 = ax2.bar(x + width/2, acc_lengths, width,
                    label='Mean Accept Length', color=color_acc, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title('Speculative Decoding: Throughput and Acceptance')
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
