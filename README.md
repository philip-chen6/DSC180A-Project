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

**Current status:** data loading, prompt formatting, numeric rewards, baseline
and speculative generation, timing, acceptance metric extraction, result
saving, plotting, and a greedy output comparison command are implemented.
Accuracy evaluation supports both baseline and speculative configs.
CPU tests pass; validation of the complete pipeline on real GPU
hardware is still pending.

| Owner | Files |
| --- | --- |
| Philip | `src/generate.py`, `src/check.py`, and their tests |
| Maggie | `src/metrics.py` |
| Chentao | `src/data.py`, `src/reward.py`, `tests/test_reward.py` |

## Setup

Use Python 3.12 on the RTX 5070 machine. A virtual environment is recommended
so project packages do not affect other Python projects. Run these commands
from the repository root in a Linux or WSL2 terminal:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For WSL2, install Ubuntu and verify GPU access from the Ubuntu terminal:

```bash
nvidia-smi
```

WSL2 uses the Windows NVIDIA driver; do not install a separate Linux NVIDIA
driver inside the WSL2 distribution. Native Linux systems should use their
normal NVIDIA driver and CUDA setup.

The CPU-only setup is enough to run the tests and fake-data plot. Install the
test and plotting dependencies with:

```bash
python -m pip install pytest matplotlib
```

For GPU runs on native Linux or WSL2, install the pinned stack:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The pinned GPU stack is intended for Linux and WSL2. The `baseline`,
`specdec`, `eval`, and `check` targets require a compatible NVIDIA GPU with
sufficient VRAM, a working CUDA runtime, model downloads, and the pinned vLLM
stack. Official benchmark results will be collected on the RTX 5070.

## Tested GPU setup

The GPU smoke test passed on November 23, 2026 with the following setup.
The original package pins installed successfully; no GPU driver or system CUDA
installation was changed. Full benchmark validation is in progress on this branch.

| Component | Tested value |
| --- | --- |
| Host | Windows 25H2, build 26200.9974 |
| WSL package / Linux kernel | 3.0.5 / 6.18.58.1-1 |
| Distribution | Ubuntu 24.04.5 LTS, WSL2 |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB, compute capability 12.0 |
| Windows NVIDIA driver | 621.01; driver-reported CUDA support 13.5 |
| Python | 3.12.3 |
| vLLM | 0.32.0 |
| PyTorch | 2.13.0 (`torch.__version__`: `2.13.0+cu130`), CUDA runtime 13.0 |
| Transformers / Datasets | 5.19.0 / 4.7.0 |

PyTorch reports `sm_120` among its compiled architectures. A CUDA matrix multiply
passed, and vLLM generated `Hello! How can I assist you today?` with
`Qwen/Qwen2.5-0.5B-Instruct`. All 62 CPU tests passed in this Linux environment.
`requirements-gpu-lock.txt` records the exact installed packages and is loaded
as constraints by `requirements.txt`.

From administrator PowerShell, install WSL and Ubuntu. If Windows requests a
reboot, restart and rerun the install command to finish adding the distribution:

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
```

Install the Linux prerequisites from PowerShell (these commands use Ubuntu's
root account and do not require a Linux password):

```powershell
wsl -d Ubuntu-24.04 -u root -- apt-get update
wsl -d Ubuntu-24.04 -u root -- apt-get install -y python3.12-venv python3.12-dev build-essential libnuma1
wsl -d Ubuntu-24.04
```

In Ubuntu, use the repository root and a venv on the Linux filesystem. On the
validation machine the repository is `/mnt/c/Dev/Projects/DSC180A-Project` and
the venv is `/home/philip/venvs/dsc180a-gpu`:

```bash
cd /mnt/c/Dev/Projects/DSC180A-Project  # adjust for your checkout
python3.12 -m venv ~/venvs/dsc180a-gpu
source ~/venvs/dsc180a-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

# Use the CUDA compiler delivered inside this venv, without system CUDA changes.
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
python run.py test
python run.py check
```

The `python3.12-dev` package is required: without its `Python.h`, Triton's
runtime C extension fails to compile during engine startup. Activate the venv
and export the compiler paths in every new shell. Initial startup downloads
model weights and compiles/autotunes kernels; later starts reuse local caches.

Both benchmark configs retain 200 prompts and use
`gpu_memory_utilization=0.7`, `max_model_len=2048`, and `max_num_seqs=1`.
The shared limits leave VRAM for the Windows desktop and use the same context
and request concurrency in both modes. The standalone draft smoke test used
`gpu_memory_utilization=0.65` and eager execution; benchmark configs keep vLLM's
normal compilation and CUDA graph settings.

## Data

Use the GSM8K test split for the initial benchmark. `load_gsm8k` selects a
reproducible sample and returns `{"question": str, "answer": str}` records, with
the final numeric answer extracted from the dataset's `#### <number>` line.
`format_prompt` applies the target tokenizer's chat template and instructs the
model to end with a numeric answer. The dataset is downloaded lazily when a
generation or evaluation target runs.

Both configs default to 200 prompts, seed 0, temperature 0.0, and 512 maximum
output tokens. Speculative decoding defaults to 5 draft tokens. Change run
parameters in `config/baseline.json` or `config/specdec.json`.

## Running each target

```bash
python run.py             # print usage
python run.py baseline    # baseline config: load, format, generate, save
python run.py specdec     # speculative config: load, format, generate, save
python run.py eval        # baseline config: generate, score rewards, save accuracy
python run.py eval --config config/specdec.json  # speculative accuracy
python run.py plot        # results/results.json or fake_data.json -> comparison.png
python run.py test        # run pytest with the current Python interpreter
python run.py check       # compare greedy baseline/specdec outputs on a GPU
```

Use `--config <path>` with `baseline`, `specdec`, `eval`, or `check` to select
a config copy without changing the originals. Use `--outputs <path>` with
generation or evaluation to save questions, gold answers, exact completions,
the config, and run statistics for inspection.

`baseline`, `specdec`, and `eval` require the full dependencies, a model
download, and a working GPU runtime. They load GSM8K, format prompts, run
generation, and append metrics to `results/results.json`. `eval` also scores
the completions with the numeric reward function. `test` runs the CPU test
suite, including mocked generation/check tests and reward/data tests; it does
not validate real model behavior.

`check` reads the math prompts in `config/check.json` directly, independently
of the GSM8K loading and prompt-formatting pipeline. It reads model settings
from `config/specdec.json`, forces temperature 0, and compares exact
completion strings on a small prompt set. It uses one child process at a time
so baseline GPU memory is released before speculative generation starts. It
prints `Greedy output comparison: X/N completions differ.` and exits nonzero
on a mismatch or failed run. This checks agreement between the two decoding
configurations; mathematical correctness is evaluated through the answer
scoring pipeline. This command requires a working vLLM GPU installation.

Generation currently sends one prompt at a time. Each request's wall time
includes tokenization, prefill, decoding, and output handling. Model startup and
metric reads are excluded; there is no extra warmup, so first-call overhead is
included. `tokens_per_sec` is total generated tokens / summed request time and
`mean_latency_s` is the mean of those request times. This measures concurrency
one, not batched RL rollout throughput.

Acceptance uses before/after deltas from `LLM.get_metrics()` with stats enabled.
The runtime prints its vLLM version and observed speculative metric names.
The [official reader](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/metrics/reader.py)
returns counters without the Prometheus `_total` suffix:

- `vllm:spec_decode_num_drafts`
- `vllm:spec_decode_num_draft_tokens`
- `vllm:spec_decode_num_accepted_tokens`

Following [vLLM's formulas](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/spec_decode/metrics.py),
mean acceptance length is `1 + accepted / drafts` (including the bonus token),
and acceptance rate is `accepted / draft_tokens` as a fraction. Missing metrics
or zero denominators produce `null`. The API was checked against upstream
source; no local vLLM installation was available to validate the installed API
or measure an actual mismatch count. Validate and pin the team's GPU version.

## Results

Benchmark records are appended to the JSON list at
`results/results.json`, as configured by `results_path`. The full record schema
is documented in `src/generate.py` and `src/metrics.py`.

Records contain the config label, prompt count, output tokens per second,
mean latency in seconds, mean output length, acceptance metrics, and accuracy.
Baseline acceptance metrics are `null`. The `baseline` and `specdec` targets
leave accuracy as `null`; the `eval` target runs the baseline configuration and
fills in its accuracy using the reward function. Select `config/specdec.json`
with `eval --config` to evaluate speculative accuracy.

The `plot` target reads `results/results.json` when benchmark results exist.
If that file is absent, it explicitly falls back to the committed synthetic
fixture `results/fake_data.json` and prints a message identifying that input.
It then generates a dual-axis bar chart comparing tokens per second and mean
acceptance length across configurations. Generated results, plots, model
weights, and caches are ignored by Git; `results/.gitkeep` keeps the output
directory in the repo.

## Known Bugs
In `README.md` and `requirements.txt`, we specify a plan for running on a specific environment, which is integral to the main topic (limited compute for RL speculative decoding). It is possible that the given instructions do not run on all GPU setups and OS because of the specific devices the authors are attempting to run on. Broader coverage will possibly be added in future iterations if they do not obscure the focus of the project scope.

There is a reward extraction bug giving credit for incorrect answers, and the metrics pipeline doesn't yet enforce the strict scheme we plan to finalize with. The current saved results do not form a complete benchmark but are demonstrative of currently working features for the checkpoint. Our realistic goal for the quarter project is to create a working MVP to build off of for next quarter. 

## Human and AI Contributions

Philip, Maggie, and Chentao (the authors) developed the project's research objectives with guidance from Zaiyang Zhang (domain PhD student mentor) and Amirhossein
Panahi (course TA). The authors developed the experimental plan and wrote the
initial abstract and introduction.

ChatGPT and Claude generated substantial portions of the implementation from
requirements and prompts supplied by the authors. The authors directed
revisions, integrated the generated code, and performed testing and review.
AI tools also assisted with proofreading, checking technical claims, and
LaTeX formatting.

The authors take responsibility for the submitted work. `AI_USE.md` documents prompt use, affected components, verification procedures, and remaining
validation limitations.
