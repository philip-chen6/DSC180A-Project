"""Dispatch capstone targets using JSON configs and src functions."""

import json
from pathlib import Path
import subprocess
import sys

from src.check import check_correctness
from src.data import format_prompt, load_gsm8k
from src.generate import run_generation
from src.metrics import plot_comparison, save_results
from src.reward import reward


ROOT = Path(__file__).resolve().parent
TARGETS = ("baseline", "specdec", "eval", "plot", "test", "check")


# Load the selected config and call the corresponding library functions.
def main() -> int:
    """Run one target; report unfinished stubs with a nonzero exit code."""
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        print("Usage: python run.py <baseline|specdec|eval|plot|test|check>")
        return 0 if len(sys.argv) == 1 else 2

    target = sys.argv[1]
    if target == "test":
        return subprocess.run(
            [sys.executable, "-m", "pytest"], cwd=ROOT
        ).returncode

    try:
        if target == "plot":
            results_path = ROOT / "results/results.json"
            if not results_path.exists():
                results_path = ROOT / "results/fake_data.json"
                print(
                    "No results/results.json found; plotting results/fake_data.json."
                )
            plot_comparison(
                str(results_path),
                str(ROOT / "results/comparison.png"),
            )
            return 0

        config_name = "specdec" if target in ("specdec", "check") else "baseline"
        with (ROOT / "config" / f"{config_name}.json").open() as file:
            config = json.load(file)

        if target == "check":
            with (ROOT / "config/check.json").open() as file:
                prompts = json.load(file)["prompts"]
            return int(check_correctness(prompts, config) > 0)

        examples = load_gsm8k("test", config["n_prompts"], config["seed"])

        # Load the tokenizer only after data loading is implemented.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        prompts = [format_prompt(example["question"], tokenizer) for example in examples]
        completions, stats = run_generation(prompts, config)

        if target == "eval":
            stats["accuracy"] = sum(
                reward(completion, example["answer"])
                for completion, example in zip(completions, examples, strict=True)
            ) / len(examples)

        save_results(stats, str(ROOT / config["results_path"]))
        return 0
    except (NotImplementedError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
