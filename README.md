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

**Current status:** generation, timing, acceptance metric extraction, and a
greedy comparison command are implemented. Data loading, prompt formatting,
rewards, result saving, and plotting remain stubs. Real GPU validation is pending.

| Owner | Files |
| --- | --- |
| Philip | `src/generate.py`, `src/check.py`, and their tests |
| Person 1 | `src/metrics.py` |
| Person 2 | `src/data.py`, `src/reward.py`, `tests/test_reward.py` |

## Setup

Use Python 3.10 or newer for this skeleton. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest
```

This is enough to check imports, CLI wiring, and the CPU tests.
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
python run.py check       # compare greedy baseline/specdec outputs on a GPU
```

For now, `baseline`, `specdec`, and `eval` exit with code 1 and a
`NotImplementedError` naming `data.load_gsm8k`. `plot` does the same for
`metrics.plot_comparison`. `test` runs mocked generation/comparison tests plus
one skipped reward placeholder; it does not validate real model behavior.
Unknown targets print usage and exit
with code 2. Paths are resolved relative to the repository, so the script also
works when called from another directory.

`check` reads the raw math prompts in `config/check.json` and model settings
from `config/specdec.json`, forces temperature 0, and compares exact completion
strings. It bypasses the unfinished GSM8K loader and uses one child process at
a time so baseline GPU memory is released before speculative generation starts.
It prints `Temperature-0 correctness: X/N outputs differ.` and exits nonzero
on a mismatch or failed run. These few prompts are a smoke check, not a GSM8K
accuracy evaluation. This command requires a working vLLM GPU installation.

Generation currently sends one prompt at a time. Each request's wall time
includes tokenization, prefill, decoding, and output handling. Model startup and
metric reads are excluded; there is no extra warmup, so first-call overhead is
included. `tokens_per_sec` is total generated tokens / summed request time and
`mean_latency_s` is the mean of those request times. This measures concurrency
one, not batched RL rollout throughput.

Acceptance uses before/after deltas from `LLM.get_metrics()` with stats enabled.
The runtime prints its vLLM version and observed speculative metric names.
The [official reader](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/reader.py)
returns counters without the Prometheus `_total` suffix:

- `vllm:spec_decode_num_drafts`
- `vllm:spec_decode_num_draft_tokens`
- `vllm:spec_decode_num_accepted_tokens`

Following [vLLM's formulas](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py),
mean acceptance length is `1 + accepted / drafts` (including the bonus token),
and acceptance rate is `accepted / draft_tokens` as a fraction. Missing metrics
or zero denominators produce `null`. The API was checked against upstream
source; no local vLLM installation was available to validate the installed API
or measure an actual mismatch count. Validate and pin the team's GPU version.

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
