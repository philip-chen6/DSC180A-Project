# Owner: Philip
"""Temperature-zero comparison, with one GPU engine process at a time."""

from importlib.util import find_spec
import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory

from src.generate import run_generation


# Save outputs in a child process so exiting releases the entire GPU engine.
def _generate_to_file(prompts: list[str], config: dict, path: str) -> None:
    """Run one mode and write exact completion strings for the parent to read."""
    completions, _ = run_generation(prompts, config)
    Path(path).write_text(json.dumps(completions), encoding="utf-8")


# Compare greedy outputs without holding both model engines in GPU memory.
def check_correctness(prompts: list[str], config: dict) -> int:
    """Report and return the number of exact baseline/specdec text mismatches.

    Require a speculative config and at least one already-formatted prompt.
    Copy the config, force temperature zero, and remove only draft settings
    for baseline. Both runs receive identical prompts, seed, and token limit.
    Spawn and join each child sequentially to release GPU memory before the
    next mode starts. Call from a guarded script entry point (as in run.py).

    This is a finite greedy smoke check, not a GSM8K accuracy evaluation or
    proof of distributional equivalence. It performs actual model generation;
    a failed child aborts the check instead of reporting a mismatch count.
    """
    if not prompts:
        raise ValueError("check_correctness requires at least one prompt.")
    if "draft_model" not in config:
        raise ValueError("check_correctness requires a draft_model.")
    if find_spec("vllm") is None:
        raise RuntimeError("vLLM is not installed; run this check in the NVIDIA environment.")

    context = multiprocessing.get_context("spawn")
    outputs = {}
    with TemporaryDirectory(prefix="specdec-check-") as directory:
        for mode in ("baseline", "specdec"):
            run_config = dict(config, temperature=0.0, n_prompts=len(prompts))
            if mode == "baseline":
                run_config.pop("draft_model")
                run_config.pop("num_speculative_tokens", None)
            path = Path(directory) / f"{mode}.json"
            print(f"Greedy check: running {mode} on {len(prompts)} prompts", flush=True)
            process = context.Process(
                target=_generate_to_file, args=(prompts, run_config, str(path))
            )
            process.start()
            try:
                process.join()
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"{mode} generation failed (exit {process.exitcode}); check aborted.")
            outputs[mode] = json.loads(path.read_text(encoding="utf-8"))
            if len(outputs[mode]) != len(prompts):
                raise RuntimeError(f"{mode} returned the wrong number of completions.")

    different = [
        index for index, (baseline, specdec) in enumerate(
            zip(outputs["baseline"], outputs["specdec"], strict=True), start=1
        ) if baseline != specdec
    ]
    print(f"Temperature-0 correctness: {len(different)}/{len(prompts)} outputs differ.")
    if different:
        print(f"Differing prompt numbers (1-based): {different}")
    return len(different)
