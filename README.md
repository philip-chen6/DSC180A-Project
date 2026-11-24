# DSC180A-Project

## Overview

UCSD data science capstone studying speculative decoding for RL rollouts.
As an RL policy changes during training, a fixed draft model may become stale
and its acceptance rate may fall. The first milestone is to run speculative
decoding with vLLM on GSM8K and compare speed, acceptance, and accuracy with a
baseline. RL training comes later.

- Initial benchmark target: `Qwen/Qwen2.5-1.5B-Instruct`.
- Validated acceleration target: `Qwen/Qwen2.5-3B-Instruct`.
- Draft: `Qwen/Qwen2.5-0.5B-Instruct` (shared tokenizer).
- Validated hardware: one RTX 5070, 12 GB VRAM, Blackwell SM120.
  Teammates develop on their own NVIDIA GPUs.

**Current status:** data loading, prompt formatting, numeric rewards, baseline
and speculative generation, timing, acceptance metric extraction, result
saving, plotting, and a greedy output comparison command are implemented.
Accuracy evaluation supports both baseline and speculative configs.
All 75 CPU tests and a real RTX 5070 vLLM smoke test pass. The GPU greedy
comparison passes with 0/3 differences in two consecutive batch-invariant runs.
The matched 20-prompt greedy and sampled GPU preflights passed. Clean full
200-prompt greedy generation and evaluation runs completed with 0/200
exact-string differences and 71.5% accuracy in both modes. Plots were generated
from real results. All four modes now have matched 200-prompt generation and
evaluation results, including sampling at temperature 1.0 and top_p 1.0.
A larger 3B target achieved 1.30x sampled throughput in a 20-question preflight
and 1.24x in the completed 200-question confirmation. All experiments so far
use fixed model weights; no RL training has run.
The matched 3B greedy comparison also completed: 1.59x throughput, 0/200
differing strings, and 83.5% accuracy in both modes.
The main sampled profile now uses optimized execution and probabilistic draft
proposals. Its 20-question preflight passed at 1.29x throughput and its completed
200-question comparison measured 81.60 vs 101.34 tokens/sec (1.24x). All 83 CPU
tests pass. The earlier full results used batch invariance and greedy proposals.

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
stack. The measured RTX 5070 results are recorded below.

## Tested GPU setup

The GPU smoke test passed on November 23, 2026 with the following setup.
The original PyTorch/vLLM pins installed successfully. Compiler dependencies
were aligned with the CUDA 13.0 runtime; no GPU driver or system CUDA
installation was changed. The greedy check now passes with batch-invariant
execution enabled in both modes.

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
| Venv CUDA compiler | NVCC / CRT / NVVM 13.0.88; CCCL 13.0.85 |
| Transformers / Datasets | 5.19.0 / 4.7.0 |

PyTorch reports `sm_120` among its compiled architectures. A CUDA matrix multiply
passed, and vLLM generated `Hello! How can I assist you today?` with
`Qwen/Qwen2.5-0.5B-Instruct`. All 75 CPU tests passed in this Linux environment.
`requirements-gpu-lock.txt` records the exact installed packages and is loaded
as constraints by `requirements.txt`. A final `pip check` passed, and all 218
installed package versions exactly matched that lock.

The validated Hugging Face model snapshots were
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` for the target and
`7ae557604adf67be50417f59c2c2f167def9a775` for the draft. The generated
`results/gpu-environment.json` also records these revisions, package versions,
the CUDA architecture list, driver output, and hashes of the selected questions.

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

# Set venv compiler paths and WSL driver linking, without system CUDA changes.
source scripts/activate-gpu.sh

nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
python run.py test
python run.py check
```

The `python3.12-dev` package is required: without its `Python.h`, Triton's
runtime C extension fails to compile during engine startup. Activate the venv
and source `scripts/activate-gpu.sh` in every new shell. Initial startup downloads
model weights and compiles/autotunes kernels; later starts reuse local caches.

An unconstrained install selected NVCC/CRT/NVVM 13.5.37 and CCCL 13.3.5.0.6
alongside PyTorch's CUDA 13.0 runtime headers. FlashInfer sampling JIT then
failed with `CUDA compiler and CUDA toolkit headers are incompatible` during
the greedy check's warmup. The explicit `cuda-toolkit[nvcc,cccl]==13.0.3.0`
requirement and matching compiler pins in the lock fix that mismatch entirely
inside the venv, following [NVIDIA's component versions](https://pypi.org/project/cuda-toolkit/13.0.3.0/).
The activation helper also supplies venv-local `lib64` and `libcudart.so`
symlinks expected by FlashInfer, and adds `/usr/lib/wsl/lib` to the compiler's
library search path. Without these paths its JIT linker reports missing
`-lcudart` and `-lcuda`. The FlashInfer sampling kernel was compiled and executed
successfully after these fixes.

The four original 1.5B benchmark configs retain 200 prompts and use
`gpu_memory_utilization=0.55`, `max_model_len=2048`, and `max_num_seqs=1`.
The shared limits leave VRAM for the Windows desktop and use the same context
and request concurrency in both modes. The standalone draft smoke test used
`gpu_memory_utilization=0.65` and eager execution. The activation helper now
defaults `VLLM_BATCH_INVARIANT=0` for vLLM's normal optimized benchmark execution.
The `check` target separately enables batch invariance for its strict greedy
comparison and restores the caller's environment afterward. In vLLM 0.32.0 on
this GPU, batch invariance selects consistent kernels, disables torch.compile,
and enables breakable CUDA graphs. Earlier measured tables explicitly used
`VLLM_BATCH_INVARIANT=1`; export that value before running their commands to
reproduce that execution profile. Exact greedy agreement is not guaranteed
with default kernels. Both arms of a performance comparison must use the same
execution profile; enabling default optimizations does not guarantee a larger
speculative speedup.
The earlier greedy preflight used a 0.7 memory budget. Fresh idle-GPU runs use
0.55 in every mode because the Windows desktop was consuming about 3.5 GB of
VRAM before model startup; the lower budget leaves more room for those allocations.

### GPU validation outcome

An initial default-kernel `python run.py check` reported 1/3 differing strings.
The original failed run did not preserve its completion strings. The diagnostic
capture added afterward reproduced 0/3 differences with unchanged settings;
both captured responses to `17 + 25` ended with 42. A token-probability trace
showed different target probabilities across the baseline and speculative
execution paths, including an exact top-choice tie at generated token index 24
between `:\n\n` and ` together` after the words `the two numbers`.

This is consistent with numerical sensitivity around a tied greedy choice; the
exact divergence in the original unretained output cannot be established.
[vLLM documents that default execution is not reproducible](https://docs.vllm.ai/en/v0.32.0/usage/reproducibility/).
The supported [batch-invariance setting](https://docs.vllm.ai/en/v0.32.0/features/batch_invariance/)
is enabled for both modes rather than weakening the exact-string comparison.
Two consecutive independent checks then reported:

```text
LLM.get_metrics() speculative names: vllm:spec_decode_num_accepted_tokens, vllm:spec_decode_num_accepted_tokens_per_pos, vllm:spec_decode_num_draft_tokens, vllm:spec_decode_num_drafts
Acceptance metrics: mean_acceptance_length=4.1, acceptance_rate=0.62
Greedy output comparison: 0/3 completions differ.
```

The three required counter names match `SPEC_COUNTERS`; no rename was needed.
These are genuine acceptance measurements from the three raw math prompts in
`config/check.json`, not GSM8K benchmark results. `check --outputs <path>` now
retains prompts and both exact completion strings, including failures. The
reports and benchmark output archives record `VLLM_BATCH_INVARIANT` explicitly.

## Data

Use the GSM8K test split for the initial benchmark. `load_gsm8k` selects a
reproducible sample and returns `{"question": str, "answer": str}` records, with
the final numeric answer extracted from the dataset's `#### <number>` line.
`format_prompt` applies the target tokenizer's chat template and instructs the
model to end with a numeric answer. The dataset is downloaded lazily when a
generation or evaluation target runs.

The greedy configs (`config/baseline.json`, `config/specdec.json`) default to
200 prompts, seed 0, temperature 0.0, and 512 maximum output tokens. The sampled
configs (`config/baseline_sample.json`, `config/specdec_sample.json`) retain the
same settings except for temperature 1.0 and top_p 1.0. Speculative decoding
defaults to 5 draft tokens. `top_p` defaults to 1.0 when absent from a config.
`draft_sample_method` is forwarded to vLLM when supplied. In the pinned vLLM
0.32.0, omitting it selects greedy draft proposals, even with a sampled target.
Set it to `"probabilistic"` to sample draft proposals and retain their full
probabilities for the standard acceptance-ratio and residual-correction rule.
Both proposal methods preserve the target sampling distribution in the
algorithm; they can differ in acceptance, memory use, and throughput. The
earlier sampled tables used greedy draft proposals, not probabilistic ones.

## Recommended sampled benchmark

The main sampling comparison uses the 3B target and 0.5B draft in BF16 with
vLLM's default optimized execution. Both arms sample at temperature 1.0 and
top_p 1.0; the speculative arm also samples its draft proposals using
`draft_sample_method="probabilistic"`. This follows the full-probability
proposal and correction algorithm, rather than vLLM's greedy-proposal default.
See [vLLM's proposal options](https://docs.vllm.ai/en/v0.32.0/api/vllm/config/speculative/#vllm.config.speculative.SpeculativeConfig.draft_sample_method).

The matched configs are `config/experiments/baseline_sample_3b_optimized.json`
and `config/experiments/specdec_sample_3b_optimized.json`: 200 questions, seed 0,
512 output tokens, 2,048 context tokens, concurrency one, a 0.7 memory budget,
and five draft tokens for speculation. `enforce_eager=false` leaves compilation
and CUDA graph optimizations available. Existing configs and historical results
are retained; this profile writes to `results/target3b_sample_optimized/`.

From the Ubuntu repo root with the GPU venv activated:

```bash
export VLLM_BATCH_INVARIANT=0
source scripts/activate-gpu.sh
python run.py test
# check temporarily uses batch-invariant, greedy execution, separate from timings.
python run.py check --config config/experiments/specdec_sample_3b_optimized.json --outputs results/target3b_sample_optimized/greedy-check.json
# Each eval measures generation and scores that same set of completions once.
python run.py eval --config config/experiments/baseline_sample_3b_optimized.json --outputs results/target3b_sample_optimized/baseline-200.json
python run.py eval --config config/experiments/specdec_sample_3b_optimized.json --outputs results/target3b_sample_optimized/specdec-200.json
MPLBACKEND=Agg python -c 'from src.metrics import plot_comparison; plot_comparison("results/target3b_sample_optimized/results.json", "results/target3b_sample_optimized/comparison-200.png")'
```

Use copies with `n_prompts=20` for a preflight. The saved output archives record
the config, runtime batch-invariance flag, completions, accuracy, and timings.
The strict greedy check is a separate diagnostic; sampled strings can differ
between modes, and finite accuracy differences do not prove a distributional
change. This profile changes both proposal strategy and execution settings, so
comparison against the earlier tables does not isolate either change's effect.
The completed measurements for this profile are recorded below.

### Optimized sampled preflight (20 questions)

The recommended profile passed its strict greedy GPU check with 0/3 differing
strings, followed by matched sampled evaluations on 20 questions:

| Mode | Tokens/sec | Mean latency (s) | Mean acceptance length | Acceptance rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optimized baseline | 79.00 | 2.355 | n/a | n/a | 75% (15/20) |
| Optimized specdec, sampled draft | 102.19 | 1.702 | 4.186 | 63.72% | 85% (17/20) |

The measured throughput ratio was 1.294x. Both timing runs recorded
`VLLM_BATCH_INVARIANT=0`, BF16 weights, and active torch.compile and CUDA graphs;
the correctness check separately recorded batch invariance enabled. Source and
config hashes, exact completions, rescored accuracy, prompt alignment, and
acceptance ranges were audited. The real-data plot was visually inspected.
`results/target3b_sample_optimized/summary-20.json` records the measurements;
the full confirmation follows. This small single-seed accuracy difference is
not evidence that speculation improves answer quality.

### Optimized sampled comparison (200 questions)

Both evaluations completed with the same 200 questions and seed, optimized
execution, BF16 models, and the limits listed above:

| Mode | Tokens/sec | Mean latency (s) | Mean output tokens | Mean acceptance length | Acceptance rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Optimized baseline | 81.60 | 2.572 | 209.855 | n/a | n/a | 78.5% (157/200) |
| Optimized specdec, sampled draft | 101.34 | 2.051 | 207.845 | 4.242 | 64.85% | 77.0% (154/200) |

This is 1.242x output-token throughput (24.2% higher) and 20.2% lower mean
latency. Output lengths differ under sampling, so latency also reflects
different generated work. Relative throughput gain is similar to the earlier
batch-invariant, greedy-proposal 3B comparison (1.243x), while both arms now
deliver about twice that profile's absolute throughput. Execution and proposal
settings changed together; these results do not isolate the effect of either
change. The three-answer accuracy gap in one sample does not establish a
change in the target distribution or model quality. All 200 sampled strings
differed across modes, which is expected to be allowed under sampling.

Saved configs, source hashes, model revisions, matching questions and gold
answers, independently recomputed rewards, metric ranges, and the actual
runtime logs were checked. Both timing arms recorded `VLLM_BATCH_INVARIANT=0`,
`CompilationMode.VLLM_COMPILE`, CUDA graphs, BF16, and no quantization. One
answer-parser audit flag was inspected: question 151 correctly ended with
`$36 / 18 hours = #### 2`; the existing scorer credited the correct final 2.
No data, reward, or metrics code changes were needed.

Artifacts are in `results/target3b_sample_optimized/`: `baseline-200.json`,
`specdec-200.json`, `summary-200.json`, `environment.json`,
`source-manifest.json`, `answer-audit-200.json`, and the visually inspected
`comparison-200.png`. The plot uses a frozen input derived from the real
results history. Package pins and model revisions are unchanged from the
tested 3B setup. These remain fixed-policy inference measurements at
concurrency one; RL training and batched rollouts remain future work.

### Sampled draft-length sweep

The next controlled experiment varies only `num_speculative_tokens` over
1, 2, 3, 5, and 8 using the optimized BF16 3B/0.5B sampled profile. The existing
five-token profile is the reference; the added configs are
`config/experiments/specdec_sample_3b_optimized_k{1,2,3,8}.json`.
All retain 200 questions and the same seed, target/draft sampling, context,
output limit, concurrency, and memory budget. Each length writes to its own
results history under `results/target3b_sample_draft_length/k<length>/` so
the shared `specdec_sample` label cannot collapse different lengths in a plot.

First evaluate copies with `n_prompts=20` on the same questions as the saved
optimized baseline and length-5 preflight. Reuse those exact archived references
after checking config/source compatibility. Select by output-token throughput,
then confirm a clear improvement on 200 questions; do not choose a draft length
by its sampled accuracy on a small preflight. Acceptance rate alone is also
insufficient: a longer draft can accept more tokens per verification while
costing more draft time. Completed measurements will be recorded here.

Example full evaluation from the GPU environment with batch invariance off:

```bash
python run.py eval --config config/experiments/specdec_sample_3b_optimized_k3.json --outputs results/target3b_sample_draft_length/k3/eval-200.json
```

## Running each target

```bash
python run.py             # print usage
python run.py baseline    # baseline config: load, format, generate, save
python run.py specdec     # speculative config: load, format, generate, save
python run.py baseline_sample  # sampled baseline: temperature 1, top_p 1
python run.py specdec_sample   # sampled speculative: temperature 1, top_p 1
python run.py eval        # baseline config: generate, score rewards, save accuracy
python run.py eval --config config/specdec.json  # speculative accuracy
python run.py eval_sample # generate and score both sampled modes sequentially
python run.py plot        # results/results.json or fake_data.json -> comparison.png
python run.py test        # run pytest with the current Python interpreter
python run.py check       # compare greedy baseline/specdec outputs on a GPU
```

Use `--config <path>` with a generation target, `eval`, or `check` to select
a config copy without changing the originals. Use `--outputs <path>` with
generation or evaluation to save questions, gold answers, exact completions,
the config, and run statistics for inspection.
With `check`, `--outputs` instead saves the exact baseline/speculative comparison.
For `eval_sample`, use `--baseline-config` and `--specdec-config` to select a
matched pair of config copies. Its `--outputs` file contains both complete run
archives under `runs.baseline_sample` and `runs.specdec_sample`.

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
`check` stays greedy-only because sampled runs need not produce identical strings, even with the same seed.

### Sampled benchmarks

This suite uses temperature 1.0 and top_p 1.0 as an explicit sampled workload;
those settings are experiment choices rather than universal RL requirements.
vLLM supports stochastic draft-model speculative decoding (see its
[draft-model example](https://docs.vllm.ai/en/v0.32.0/features/speculative_decoding/draft_model/)).
The effect of sampling on speedup is measured rather than assumed.

`baseline_sample` and `specdec_sample` generate and save timing/acceptance
records. `eval_sample` regenerates and scores both modes with `reward()`, using
one child process at a time so each GPU engine releases its memory before the
next starts. Its configs must match except for draft-model settings; it uses
the same questions, seed, sampling settings, and limits for both modes.

Saved labels distinguish the workloads:

| Temperature | No draft model | With draft model |
| --- | --- | --- |
| 0 | `baseline` | `specdec` |
| Positive | `baseline_sample` | `specdec_sample` |

The existing plot groups by these labels and therefore keeps all four modes
separate. Sampled target overrides require a positive temperature. Baseline
acceptance metrics stay `null`, and generation-only targets leave accuracy
`null` until evaluation. `eval_sample` appends two accuracy records to the
configured results file and does not require the measured accuracies to match:
finite samples can differ by chance. A single 20-question run cannot establish
distributional equivalence or an accuracy difference. This remains one sampled
completion per question at concurrency one, without an RL training loop.

For a 20-prompt sampled preflight, create config copies and run the following
only when the GPU is idle:

```bash
mkdir -p .cache/gpu-validation
python - <<'PY'
import json
from pathlib import Path
for mode in ('baseline_sample', 'specdec_sample'):
    config = json.loads(Path(f'config/{mode}.json').read_text())
    config['n_prompts'] = 20
    Path(f'.cache/gpu-validation/{mode}-20.json').write_text(json.dumps(config, indent=2))
PY
python run.py baseline_sample --config .cache/gpu-validation/baseline_sample-20.json --outputs results/small-baseline-sample.json
python run.py specdec_sample --config .cache/gpu-validation/specdec_sample-20.json --outputs results/small-specdec-sample.json
python run.py eval_sample --baseline-config .cache/gpu-validation/baseline_sample-20.json --specdec-config .cache/gpu-validation/specdec_sample-20.json --outputs results/small-sampled-eval.json
```

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
or zero denominators produce `null`. The installed vLLM 0.32.0 API was validated
on the RTX 5070: counters were available and acceptance values were numeric.
The observed mismatch count and exact environment are recorded in the tested
GPU setup above.

## Results

### Idle-GPU full greedy and sampled comparison (200 prompts)

All four original 200-prompt configs completed on the same questions with the
shared 0.55 memory budget and `VLLM_BATCH_INVARIANT=1`, with Valorant closed.
The final evaluation records in `results/results.json` were:

| Mode | Temperature | Tokens/sec | Mean latency (s) | Mean output tokens | Mean acceptance length | Acceptance rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0 | 65.80 | 3.001 | 197.470 | n/a | n/a | 71.5% (143/200) |
| `specdec` | 0 | 73.45 | 2.689 | 197.470 | 4.701 | 74.02% | 71.5% (143/200) |
| `baseline_sample` | 1 | 69.43 | 2.550 | 177.035 | n/a | n/a | 57% (114/200) |
| `specdec_sample` | 1 | 56.51 | 3.036 | 171.580 | 3.610 | 52.20% | 52.5% (105/200) |

All 200 greedy completion strings matched exactly between baseline and
speculative modes. Every mode's evaluation rerun reproduced its generation-only
outputs exactly. Greedy speculation achieved 1.12x throughput, with 10.4% lower
mean latency. Sampled speculation achieved 0.81x throughput (18.6% lower), with
19.1% higher mean latency. Sampled output lengths differed, so latency also
reflects different amounts of generated work. These are single-seed,
concurrency-one measurements using the reproducible kernel setting above.

The separate greedy generation-only runs measured 65.36 and 73.76 tokens/sec
(1.13x), respectively. Sampled generation-only runs measured 70.31 and 56.85
tokens/sec (0.81x), with the same accuracy scores as the evaluation reruns.

Accuracy was recomputed from every archived completion with `reward()`.
Prompt/gold alignment, runtime settings, acceptance ranges, and correspondence
with the latest 200-prompt result records passed the final four-mode audit,
saved as `results/idle-full-all-modes-summary.json`. The earlier greedy-only
audit remains in `results/idle-full-summary.json`. The greedy archives are
`results/idle-baseline-200.json`, `results/idle-specdec-200.json`,
`results/idle-eval-baseline-200.json`, and `results/idle-eval-specdec-200.json`.
No teammate-owned data, reward, or metrics logic was changed.

Reproduce the full greedy suite from the activated Ubuntu GPU environment:

```bash
python run.py baseline --outputs results/idle-baseline-200.json
python run.py specdec --outputs results/idle-specdec-200.json
python run.py eval --outputs results/idle-eval-baseline-200.json
python run.py eval --config config/specdec.json --outputs results/idle-eval-specdec-200.json
MPLBACKEND=Agg python run.py plot
```

### Full sampled GPU suite (200 prompts)

Both sampled generation runs completed on the same 200 questions as the greedy
runs. They retain temperature 1.0, top_p 1.0, seed 0, the 512-token output limit,
2,048-token context, 0.55 memory budget, and batch-invariant execution. Both
accuracy evaluation reruns also completed; their final measurements appear
in the four-mode table above. Run the suite from the activated Ubuntu GPU
environment:

```bash
python run.py baseline_sample --outputs results/idle-baseline-sample-200.json
python run.py specdec_sample --outputs results/idle-specdec-sample-200.json
python run.py eval_sample --outputs results/idle-eval-sample-200.json
```

The evaluation runs both sampled modes sequentially and records accuracy in
`results/results.json`; its output archive retains both complete sets of
responses. All 200 sampled strings differed between baseline and speculative
modes, which is allowed under sampling. Within each mode, generation and
evaluation strings matched exactly. There were 80 questions both modes got
right, 34 only the baseline got right, 25 only speculation got right, and 61
both got wrong. The observed nine-answer accuracy gap comes from one seed and
does not establish a distributional change.

Numeric sanity checks flagged two credited baseline responses; inspection
confirmed both were valid (a spaced decimal `88. 00` and a boxed `200` followed
by a reference to five days). No credited speculative response was flagged.
The full audit confirmed prompt/gold alignment and identical shared settings
across all four modes. Package versions, model revisions, GPU capability, and
dataset hashes matched the earlier tested environment; its snapshot is retained
as `results/gpu-environment-before-sampled-200.json`.

### Validated 3B greedy and sampled acceleration profiles

To test whether a larger target makes the draft cost easier to amortize,
`config/experiments/baseline_sample_3b.json` and
`config/experiments/specdec_sample_3b.json` use
`Qwen/Qwen2.5-3B-Instruct` with the existing 0.5B draft. The initial preflight
used 20 questions, temperature 1.0, top_p 1.0, seed 0, five draft tokens,
512 maximum output tokens, and the same 2,048-token context and concurrency one.
Both modes use a 0.70 GPU memory budget and `VLLM_BATCH_INVARIANT=1`.

The new pair loaded successfully on the RTX 5070: vLLM reported 6.73 GiB of
model memory and 1.31 GiB available for KV cache. Its greedy correctness gate
passed with 0/3 differing strings. The full 200-question confirmation completed:

| Mode | Tokens/sec | Mean latency (s) | Mean output tokens | Mean acceptance length | Acceptance rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3B greedy baseline | 39.71 | 5.415 | 215.045 | n/a | n/a | 83.5% (167/200) |
| 3B greedy specdec | 63.21 | 3.402 | 215.045 | 4.613 | 72.26% | 83.5% (167/200) |
| 3B sampled baseline | 39.97 | 5.224 | 208.810 | n/a | n/a | 80% (160/200) |
| 3B sampled specdec | 49.69 | 4.332 | 215.290 | 3.659 | 53.19% | 83% (166/200) |

Greedy speculation achieved 1.59x throughput (59.2% higher) and 37.2% lower
mean latency. All 200 greedy completion strings matched exactly, with identical
output lengths and accuracy. Sampled speculation achieved 1.24x throughput
(24.3% higher) and 17.1% lower mean latency. Its mean output was longer,
so the latency comparison also includes different amounts of generated work.
The six-answer accuracy gap is a single-seed observation, not evidence of a
quality improvement from speculation.

For comparison, the 1.5B sampled pair achieved 0.81x throughput with 52.20%
acceptance. The 3B pair achieved a speedup with a similar 53.19% acceptance
rate. This is consistent with the draft becoming less expensive relative to
the target, although no profiler breakdown has isolated that cause. Absolute
speed and output quality also change when switching target models. The two
model pairs use their own matched baselines and different memory budgets:
0.55 for the 1.5B target and 0.70 for the 3B target.

Both targets' completed 200-question comparisons were:

| Target / draft | Greedy throughput ratio | Sampled throughput ratio |
| --- | ---: | ---: |
| 1.5B / 0.5B | 1.12x | 0.81x |
| 3B / 0.5B | 1.59x | 1.24x |

Every ratio compares speculation with the baseline for the same target and
temperature. These are fixed-weight, single-seed, concurrency-one measurements
with batch-invariant execution, rather than end-to-end RL training timings.

The earlier 20-question preflight measured:

| Mode | Tokens/sec | Mean latency (s) | Mean output tokens | Mean acceptance length | Acceptance rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3B sampled baseline | 39.78 | 4.962 | 197.4 | n/a | n/a | 65% (13/20) |
| 3B sampled specdec | 51.58 | 3.358 | 173.2 | 3.587 | 51.74% | 80% (16/20) |

The preflight throughput ratio was 1.30x. Its small sampled accuracy difference
does not establish a quality improvement. The committed experiment profiles
use 200 questions for the full comparison.
Both sizes write separate result histories: the 3B pair uses
`results/target3b_sample/results.json`. Reproduce the 20-question preflight from
the activated GPU shell using copies:

```bash
mkdir -p .cache/gpu-validation
python - <<'PY'
import json
from pathlib import Path
for mode in ('baseline', 'specdec'):
    config = json.loads(Path(f'config/experiments/{mode}_sample_3b.json').read_text())
    config['n_prompts'] = 20
    Path(f'.cache/gpu-validation/{mode}_sample_3b-20.json').write_text(json.dumps(config, indent=2))
PY
python run.py check --config config/experiments/specdec_sample_3b.json --outputs results/target3b_sample/greedy-check.json
python run.py eval --config .cache/gpu-validation/baseline_sample_3b-20.json --outputs results/target3b_sample/baseline-20.json
python run.py eval --config .cache/gpu-validation/specdec_sample_3b-20.json --outputs results/target3b_sample/specdec-20.json
```

The full confirmation uses the committed 200-question profiles. Run the two
modes separately to retain each completion archive even if a later run fails:

```bash
python run.py eval --config config/experiments/baseline_sample_3b.json --outputs results/target3b_sample/baseline-200.json
python run.py eval --config config/experiments/specdec_sample_3b.json --outputs results/target3b_sample/specdec-200.json
```

The greedy counterparts, `config/experiments/baseline_3b.json` and
`config/experiments/specdec_3b.json`, use the same 200 questions, model pair,
seed, context, output limit, memory budget, and batch-invariant setting, with
temperature 0. Both full greedy runs completed, and their measurements appear
in the four-mode table above. These modes write to a separate history at
`results/target3b_greedy/results.json`:

```bash
python run.py eval --config config/experiments/baseline_3b.json --outputs results/target3b_greedy/baseline-200.json
python run.py eval --config config/experiments/specdec_3b.json --outputs results/target3b_greedy/specdec-200.json
```

The final audit recomputed accuracy from all saved answers, verified alignment
with the original 200 GSM8K questions, checked shared configs and metric ranges,
and matched the latest completed history rows to their output archives. The
two numeric sanity-check flags were valid marked final answers on inspection;
the greedy numeric sanity check flagged no credited answers. No teammate-owned
reward logic was changed. All 200 sampled strings differed
between modes, which is allowed. A historical interrupted baseline entry remains
in the append-only history; the table and plots use the latest completed pair.

`results/target3b_sample/summary-200.json` stores the audited measurements and
source hashes. `baseline-200.json` and `specdec-200.json` retain each mode's full
answers; `eval-200.json` combines those archives. Source/config checksums for
checkpoint `c9c3cac` are in `source-manifest.json`. The environment snapshot
records the 3B model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` and the
unchanged package versions. Results are saved in the main project folder.

`results/target3b/summary-200.json` combines the four audited 3B records, their
throughput ratios, the 0/200 greedy mismatch count, and hashes of the two
histories and output archives. The greedy source/config snapshot is retained
at `results/target3b_greedy/source-manifest.json` for checkpoint `bfbcff4`.
Package versions and both model revisions matched the sampled run's environment.

The visually checked `results/target3b_sample/comparison-200.png` shows the 3B
pair, and `results/target3b_sample/speedup-by-target.png` compares the measured
sampling speedup for each target against its own baseline. Both use real
benchmark records. To plot the 3B profile's own result history:

```bash
MPLBACKEND=Agg python -c "from src.metrics import plot_comparison; plot_comparison('results/target3b_sample/results.json', 'results/target3b_sample/comparison-200.png')"
```

The complete four-mode figure is `results/target3b/comparison-200.png`.
`results/target3b/speedup-by-target-and-mode.png` compares greedy and sampled
speedups for both target sizes. These figures were visually inspected. To
recreate the four-mode input and plot from the real 3B histories:

```bash
MPLBACKEND=Agg python - <<'PY'
import json
from pathlib import Path
from src.metrics import plot_comparison
latest = {}
for folder in ('target3b_greedy', 'target3b_sample'):
    for record in json.loads(Path(f'results/{folder}/results.json').read_text()):
        if record['n_prompts'] == 200:
            latest[record['config']] = record
labels = ('baseline', 'specdec', 'baseline_sample', 'specdec_sample')
output = Path('results/target3b')
output.mkdir(exist_ok=True)
plot_input = output / 'plot-input-200.json'
plot_input.write_text(json.dumps([latest[label] for label in labels], indent=2))
plot_comparison(str(plot_input), str(output / 'comparison-200.png'))
PY
```

This remains fixed-weight inference and answer scoring. RL would additionally
use rewards to update the target policy's weights and repeat rollout generation;
that training loop has not been run.

### Idle-GPU greedy and sampled comparison (20 prompts)

Fresh runs after Valorant closed used the same 20 questions, seed 0, 512-token
output limit, 2,048-token context, and 0.55 GPU memory budget. Both decoding
regimes used `VLLM_BATCH_INVARIANT=1`. The final evaluation records were:

| Mode | Temperature | Tokens/sec | Mean latency (s) | Mean output tokens | Mean acceptance length | Acceptance rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0 | 57.59 | 2.830 | 163.0 | n/a | n/a | 65% (13/20) |
| `specdec` | 0 | 65.65 | 2.483 | 163.0 | 4.653 | 73.06% | 65% (13/20) |
| `baseline_sample` | 1 | 59.92 | 2.924 | 175.2 | n/a | n/a | 70% (14/20) |
| `specdec_sample` | 1 | 50.71 | 3.348 | 169.8 | 3.597 | 51.95% | 60% (12/20) |

Observed throughput ratios were 1.14x for greedy speculation and 0.85x for
sampled speculation. Sampling therefore reduced acceptance and produced a
slowdown for this model pair in this preflight. These are single-seed,
20-question measurements, not a statistical claim about accuracy or a maximum
hardware throughput claim. Sampled output lengths also differed, so latency
includes that difference in generated work.

All 20 greedy completion strings matched. All 20 sampled completion strings
differed across baseline/speculative modes, which is allowed under sampling.
Within each sampled mode, the generation-only and evaluation reruns reproduced
the exact same strings. Accuracy was independently recomputed from each saved
completion, and metric ranges, prompt/gold alignment, and runtime settings were
checked. `results/idle-preflight-summary.json` records the audited table; full
output archives use the `results/idle-*` prefix. All raw records remain appended
in `results/results.json`.

### Earlier greedy GPU preflight (20 prompts)

Both modes and both `eval` reruns produced identical completion strings for all
20 questions. The final evaluation records in `results/results.json` are:

| Mode | Tokens/sec | Mean latency (s) | Mean acceptance length | Acceptance rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 66.24 | 2.461 | n/a | n/a | 65% (13/20) |
| Speculative | 70.18 | 2.323 | 4.653 | 73.06% | 65% (13/20) |

These are real RTX 5070 measurements with `VLLM_BATCH_INVARIANT=1`. The
generation-only preflight measured 66.03 and 69.90 tokens/sec respectively.
The accuracy scores were checked against the saved completions, including
manual inspection of responses that did not follow the requested `####` format.
No teammate-owned data, reward, or metrics logic was changed.

The earlier preflight used a 0.7 GPU memory budget and is superseded by the
matched idle-GPU table above. A later 200-prompt greedy baseline completed under
concurrent gaming load and is excluded from clean timing comparisons. Its
exploratory record is preserved in the append-only results history.

Create the preflight configs without changing the originals, from the Ubuntu
repo root after activating the GPU environment:

```bash
mkdir -p .cache/gpu-validation
python - <<'PY'
import json
from pathlib import Path
for mode in ('baseline', 'specdec'):
    config = json.loads(Path(f'config/{mode}.json').read_text())
    config['n_prompts'] = 20
    Path(f'.cache/gpu-validation/{mode}-20.json').write_text(json.dumps(config, indent=2))
PY
python run.py baseline --config .cache/gpu-validation/baseline-20.json --outputs results/small-baseline.json
python run.py specdec --config .cache/gpu-validation/specdec-20.json --outputs results/small-specdec.json
python run.py eval --config .cache/gpu-validation/baseline-20.json --outputs results/small-eval-baseline.json
python run.py eval --config .cache/gpu-validation/specdec-20.json --outputs results/small-eval-specdec.json
```

Benchmark records are appended to the JSON list at
`results/results.json`, as configured by `results_path`. The full record schema
is documented in `src/generate.py` and `src/metrics.py`.

Records contain the config label, prompt count, output tokens per second,
mean latency in seconds, mean output length, acceptance metrics, and accuracy.
Baseline acceptance metrics are `null`. All four generation targets
leave accuracy as `null`; the `eval` target runs the baseline configuration and
fills in its accuracy using the reward function. Select `config/specdec.json`
with `eval --config` to evaluate speculative accuracy. `eval_sample` scores both
sampled modes and saves their accuracy under the two sampled labels.

The `plot` target reads `results/results.json` when benchmark results exist.
If that file is absent, it explicitly falls back to the committed synthetic
fixture `results/fake_data.json` and prints a message identifying that input.
It then generates a dual-axis bar chart comparing tokens per second and mean
acceptance length across configurations. Generated results, plots, model
weights, and caches are ignored by Git; `results/.gitkeep` keeps the output
directory in the repo.

The completed GPU suite's `plot` log confirmed
`Plot input: /mnt/c/Dev/Projects/DSC180A-Project/results/results.json`.
The standard plot selects the latest record per label; `comparison.png` now
contains 200-prompt results for all four modes. The additional
`results/comparison-20.png` retains the four-mode preflight comparison, and
`results/comparison-200.png` shows all four full runs with the prompt count
in its title.
Their inputs, `results/plot-input-20.json` and `results/plot-input-200.json`,
were filtered from the real results history and checked against the audited
summary files. `results/plot-provenance.json` records the source file's SHA256.
Both matched plots were visually inspected; no synthetic data was used.

## Known Bugs
In `README.md` and `requirements.txt`, we specify a plan for running on a specific environment, which is integral to the main topic (limited compute for RL speculative decoding). It is possible that the given instructions do not run on all GPU setups and OS because of the specific devices the authors are attempting to run on. Broader coverage will possibly be added in future iterations if they do not obscure the focus of the project scope.

An earlier checkpoint reported incorrect credits from reward extraction. The
current scorer remains a decimal-only heuristic and does not interpret
fractions. Saved completions and numeric sanity checks support the reported
scores, but they do not prove the parser handles every possible response.
The GPU tables above are measured inference benchmarks; RL training and policy
drift experiments remain future work.

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
