# DSC180A-Project

## Overview

UCSD data science capstone studying speculative decoding for RL rollouts.
As an RL policy changes during training, a fixed draft model may become stale
and its acceptance rate may fall. The first milestone is to run speculative
decoding with vLLM on GSM8K and compare speed, acceptance, and accuracy with a
baseline. RL training comes later.

- Target: `Qwen/Qwen2.5-1.5B-Instruct`.
- Draft: `Qwen/Qwen2.5-0.5B-Instruct` (shared tokenizer).
- Planned official hardware: one RTX 5070, 12 GB VRAM, Blackwell SM120.
  Teammates develop on their own NVIDIA GPUs.

**Current status: skeleton only.** Data loading, prompt formatting, generation,
rewards, result saving, and plotting are intentionally unimplemented.

| Owner | Files |
| --- | --- |
| Philip | `src/generate.py` |
| Person 1 | `src/metrics.py` |
| Person 2 | `src/data.py`, `src/reward.py`, `tests/test_reward.py` |

## Setup

Use Python 3.10 or newer for this skeleton. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest
```

This is enough to check imports, CLI wiring, and the skipped placeholder test.
No model downloads or GPU are needed for these checks.

For future GPU runs, install the full dependencies:

```bash
python -m pip install -r requirements.txt
```

TODO: validate the Python, driver, CUDA, PyTorch, and vLLM combination on the
official GPU, then pin exact package versions in `requirements.txt`. Hardware
compatibility and model memory usage have not been validated by this skeleton.

## Data

Use the GSM8K test split for the initial benchmark. `load_gsm8k` will select a
reproducible sample and return `{"question": str, "answer": str}` records, with
the final numeric answer extracted from the dataset's `#### <number>` line.
`format_prompt` will apply the target tokenizer's chat template and instruct
the model to end with a numeric answer.

Both configs default to 200 prompts, seed 0, temperature 0.0, and 512 maximum
output tokens. Speculative decoding defaults to 5 draft tokens. Change run
parameters in `config/baseline.json` or `config/specdec.json`.

## Running each target

```bash
python run.py             # print usage
python run.py baseline    # baseline config: load, format, generate, save
python run.py specdec     # speculative config: load, format, generate, save
python run.py eval        # baseline config: generate, score rewards, save accuracy
python run.py plot        # results/results.json -> results/comparison.png
python run.py test        # run pytest with the current Python interpreter
```

For now, `baseline`, `specdec`, and `eval` exit with code 1 and a
`NotImplementedError` naming `data.load_gsm8k`. `plot` does the same for
`metrics.plot_comparison`. `test` succeeds with one skipped placeholder; it
does not yet validate reward behavior. Unknown targets print usage and exit
with code 2. Paths are resolved relative to the repository, so the script also
works when called from another directory.

## Results

No benchmark results yet. Future runs append records to the JSON list at
`results/results.json`, as configured by `results_path`. The full record schema
is documented in `src/generate.py` and `src/metrics.py`.

Records contain the config label, prompt count, output tokens per second,
mean latency in seconds, mean output length, acceptance metrics, and accuracy.
Baseline acceptance metrics and unevaluated accuracy are `null`. The `eval`
target saves a baseline record with accuracy filled in. Evaluating speculative
accuracy remains future work beyond these requested targets.

The planned plot compares throughput and mean acceptance length. Generated
results, plots, model weights, and caches are ignored by Git; `results/.gitkeep`
keeps the output directory in the repo.
