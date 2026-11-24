"""Dispatch capstone targets using JSON configs and src functions."""

import argparse
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
def main(argv: list[str] | None = None) -> int:
    """Run one target; report unfinished stubs with a nonzero exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=TARGETS)
    parser.add_argument("--config", type=Path, help="JSON config, relative to the repo root")
    parser.add_argument("--outputs", type=Path, help="Save questions, gold answers, and completions as JSON")
    args = parser.parse_args(argv)
    if args.target is None:
        parser.print_help()
        return 0
    target = args.target
    if args.config and target in ("plot", "test"):
        parser.error("--config applies to generation, eval, and check")
    if args.outputs and target in ("plot", "test"):
        parser.error("--outputs applies to generation, eval, and check")
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
        config_path = ROOT / args.config if args.config else ROOT / "config" / f"{config_name}.json"
        with config_path.open() as file:
            config = json.load(file)
        if target == "baseline" and "draft_model" in config:
            raise ValueError("baseline requires a config without draft_model")
        if target == "specdec" and "draft_model" not in config:
            raise ValueError("specdec requires a config with draft_model")

        if target == "check":
            with (ROOT / "config/check.json").open() as file:
                prompts = json.load(file)["prompts"]
            report_path = str(ROOT / args.outputs) if args.outputs else None
            return int(check_correctness(prompts, config, report_path=report_path) > 0)

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

        if args.outputs:
            output_path = ROOT / args.outputs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({
                "config": config, "stats": stats,
                "examples": [dict(example, completion=completion)
                             for example, completion in zip(examples, completions, strict=True)],
            }, indent=2), encoding="utf-8")
        save_results(stats, str(ROOT / config["results_path"]))
        print(json.dumps(stats, indent=2))
        return 0
    except (NotImplementedError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
