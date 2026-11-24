"""Dispatch capstone targets using JSON configs and src functions."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from src.check import check_correctness
from src.data import format_prompt, load_gsm8k
from src.generate import DRAFT_CONFIG_KEYS, run_generation
from src.metrics import plot_comparison, save_results
from src.reward import reward


ROOT = Path(__file__).resolve().parent
TARGETS = ("baseline", "specdec", "baseline_sample", "specdec_sample",
           "eval", "eval_sample", "plot", "test", "check")


def _evaluate_sampled_pair(baseline_path: Path, specdec_path: Path,
                           outputs_path: Path | None) -> int:
    """Score matched sampled configs in separate, sequential GPU processes."""
    configs = {}
    for mode, path in (("baseline_sample", baseline_path), ("specdec_sample", specdec_path)):
        configs[mode] = json.loads(path.read_text(encoding="utf-8"))
    baseline = configs["baseline_sample"]
    speculative = configs["specdec_sample"]
    if "draft_model" in baseline or "draft_model" not in speculative:
        raise ValueError("eval_sample requires a baseline config and a speculative config")
    if baseline["temperature"] <= 0 or speculative["temperature"] <= 0:
        raise ValueError("eval_sample requires positive sampling temperatures")
    expected_baseline = {key: value for key, value in speculative.items()
                         if key not in DRAFT_CONFIG_KEYS}
    expected_baseline.setdefault("top_p", 1.0)
    if dict(baseline, top_p=baseline.get("top_p", 1.0)) != expected_baseline:
        raise ValueError("Sampled evaluation configs must match except for draft settings")

    records = {}
    with TemporaryDirectory(prefix="specdec-sample-eval-") as directory:
        for mode, path in (("baseline_sample", baseline_path), ("specdec_sample", specdec_path)):
            archive = Path(directory) / f"{mode}.json"
            print(f"Sampled evaluation: running {mode}", flush=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "run.py"), "eval", "--config", str(path),
                 "--outputs", str(archive)], cwd=ROOT,
            )
            if result.returncode:
                print(f"{mode} evaluation failed; sampled evaluation aborted.", file=sys.stderr)
                return result.returncode
            record = json.loads(archive.read_text(encoding="utf-8"))
            if record["stats"]["config"] != mode or record["stats"]["accuracy"] is None:
                raise RuntimeError(f"{mode} did not return a labeled accuracy record")
            records[mode] = record
    if outputs_path is not None:
        outputs_path.parent.mkdir(parents=True, exist_ok=True)
        outputs_path.write_text(json.dumps({"runs": records}, indent=2), encoding="utf-8")
    print("Sampled accuracy comparison (finite samples may differ):")
    print(json.dumps({mode: record["stats"]["accuracy"] for mode, record in records.items()}, indent=2))
    return 0


# Load the selected config and call the corresponding library functions.
def main(argv: list[str] | None = None) -> int:
    """Run one benchmark, evaluation, plotting, or validation target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=TARGETS)
    parser.add_argument("--config", type=Path, help="JSON config, relative to the repo root")
    parser.add_argument("--outputs", type=Path, help="Save exact completions and run settings as JSON")
    parser.add_argument("--baseline-config", type=Path, help="Baseline config for eval_sample")
    parser.add_argument("--specdec-config", type=Path, help="Speculative config for eval_sample")
    args = parser.parse_args(argv)
    if args.target is None:
        parser.print_help()
        return 0
    target = args.target
    if target == "eval_sample" and args.config:
        parser.error("eval_sample uses --baseline-config and --specdec-config")
    if target != "eval_sample" and (args.baseline_config or args.specdec_config):
        parser.error("--baseline-config and --specdec-config apply only to eval_sample")
    if args.config and target in ("plot", "test"):
        parser.error("--config applies to generation, eval, and check")
    if args.outputs and target in ("plot", "test"):
        parser.error("--outputs applies to generation, eval, and check")
    if target == "test":
        return subprocess.run(
            [sys.executable, "-m", "pytest"], cwd=ROOT
        ).returncode

    try:
        if target == "eval_sample":
            return _evaluate_sampled_pair(
                ROOT / (args.baseline_config or Path("config/baseline_sample.json")),
                ROOT / (args.specdec_config or Path("config/specdec_sample.json")),
                ROOT / args.outputs if args.outputs else None,
            )
        if target == "plot":
            results_path = ROOT / "results/results.json"
            if not results_path.exists():
                results_path = ROOT / "results/fake_data.json"
                print(
                    "No results/results.json found; plotting results/fake_data.json."
                )
            print(f"Plot input: {results_path}")
            plot_comparison(
                str(results_path),
                str(ROOT / "results/comparison.png"),
            )
            return 0

        config_name = {"check": "specdec", "eval": "baseline"}.get(target, target)
        config_path = ROOT / args.config if args.config else ROOT / "config" / f"{config_name}.json"
        with config_path.open() as file:
            config = json.load(file)
        if target in ("baseline", "baseline_sample") and "draft_model" in config:
            raise ValueError("baseline requires a config without draft_model")
        if target in ("specdec", "specdec_sample") and "draft_model" not in config:
            raise ValueError("specdec requires a config with draft_model")
        if target in ("baseline_sample", "specdec_sample") and config["temperature"] <= 0:
            raise ValueError(f"{target} requires a positive sampling temperature")

        if target == "check":
            with (ROOT / "config/check.json").open() as file:
                prompts = json.load(file)["prompts"]
            report_path = str(ROOT / args.outputs) if args.outputs else None
            # Exact greedy checks need consistent arithmetic across decode shapes.
            previous = os.environ.get("VLLM_BATCH_INVARIANT")
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            print("Greedy correctness check: VLLM_BATCH_INVARIANT=1", flush=True)
            try:
                return int(check_correctness(prompts, config, report_path=report_path) > 0)
            finally:
                if previous is None:
                    os.environ.pop("VLLM_BATCH_INVARIANT", None)
                else:
                    os.environ["VLLM_BATCH_INVARIANT"] = previous

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
                "runtime": {"VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT", "0")},
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
