# AI Use Disclosure

## Author Contributions

### Report Writing
The authors established the initial research question, compared hardware settings, and planned experimental contributions. We supplied the writing of the introduction, relevant sources for bibliography entries, and all figures and diagrams used in the report.

### Code Drafting and Verification
The authors provided co-drafted prompts and strict requirements for code generation to code agents (ChatGPT, Claude). Specific prompts expand over the following categories:

- *Data Contracts*: Enforcing strict Python typing and JSON schema validation for the metrics.py pipeline.

- *Visualization*: Specifying dual-axis matplotlib configurations to accurately plot tokens-per-second against draft acceptance rates.

- *Environment Constraints*: Providing explicit hardware bounds (12GB VRAM, RTX 5070, Linux) to generate conflict-free, pinned requirements.txt files for PyTorch and vLLM.

They also connected the generated functions to the repository’s actual entry points, checking argument names, configuration, tokenization, model loading, output formats, and checkpoint paths. Then, they manually ran the pipeline from the README documented commands. Lastly, the authors closely reviewed each other's implementation and figured out how each part works together for a bigger picture view of the repository.

## AI Contributions

### Technical Review & Writing
ChatGPT and Claude reviewed supplied introduction, fact-checking and suggesting rephrasings of some of the concepts. The specific prompt(s) asked ChatGPT to read over the current report sections according to pasted course guidelines and deliver constructive but strict feedback. 

ChatGPT helped with concepts including: qualifying when autoregressive decoding is limited by memory bandwidth, explaining that exact speculative sampling requires an acceptance and correction procedure, and presenting reduced draft acceptance during RL as a possible outcome to measure. These outputs were used as drafting assistance when writing the next iterations of the report drafts.

### Citation Assistance
ChatGPT compared the existing relevant literature with compiled bibliography against claims in the text. We prompted it to check for any possible hallucinations or misattributions on our part.

### LaTeX Editing
ChatGPT helped with LaTeX formatting to insert figures and math.

### Code
AI tools generated substantial portions of the implementation based on the aforementioned prompts and requirements provided by the authors. Subsequent iterations, if any problems were found, were also performed by code agents, held to standards set by authors.

### GPU Validation and Sampled Benchmarks

At Philip's direction, Codex configured Ubuntu under WSL2, installed and pinned
the venv GPU stack, diagnosed compiler/linker failures, and ran the pipeline on
the RTX 5070. It added exact-output archives, GPU memory controls, a shared
batch-invariant execution setting, and sampled benchmark/evaluation targets.
No NVIDIA driver or system CUDA changes were made. Teammate-owned data, reward,
and metrics logic was retained.

Verification includes 75 passing CPU tests, a real vLLM draft-model smoke test,
exact greedy output comparisons, and matched 20-question greedy and sampled
GSM8K runs. Full 200-question greedy and sampled generation and evaluation runs
also completed for the initial 1.5B target. All greedy outputs matched between
modes; each sampled mode's
evaluation reproduced its own generation outputs exactly. Sampled outputs
were allowed to differ across modes. The saved
completions were rescored with the existing reward function, and prompt
alignment and acceptance metric ranges were checked. Plots were generated
from the real results history and visually inspected.

Codex then tested a 3B target with the same 0.5B draft and five-token draft
length. A greedy correctness gate passed, and sampled evaluations on 20 and
200 questions measured a speedup against the 3B baseline. A full 200-question
greedy comparison also completed, with 0/200 string differences and 83.5%
accuracy in both modes. The full runs retained
per-mode completion archives, source/config checksums, audited scores, and an
environment snapshot. Its artifacts and plots were saved in the main project
folder; no source data, reward, or metrics logic changes were needed.
The combined figures compare all four 3B modes and both target sizes' measured
greedy and sampled throughput ratios.

The README records the final results and reproducible setup. These experiments
use one completion per question at concurrency one, without an RL training
loop. Small sampled accuracy differences do not establish a distributional
change, and the heuristic answer parser remains a validation limitation.

At Philip's direction, Codex subsequently separated performance execution
from the greedy correctness profile: the activation helper defaults to normal
vLLM execution, while `check` enables batch invariance for the exact comparison.
It added forwarding for probabilistic draft proposals without changing target
sampling, and updated paired evaluation and correctness config filtering.
All 83 CPU tests passed, including proposal forwarding, sampled paired
evaluation, and restoration of the benchmark environment after `check`.
Earlier GPU results retain their batch-invariant, greedy-proposal provenance.
The new 3B/0.5B BF16 profile passed its GPU greedy gate (0/3 differences) and
optimized 20-question sampled evaluations. Throughput was 79.00 tokens/sec
for baseline and 102.19 for speculative decoding, a 1.294x ratio. Runtime logs
confirmed torch.compile and CUDA graphs, with batch invariance off for both
timing arms. Saved completion scores, input alignment, config/source hashes,
and acceptance ranges were audited, and the real-data plot was inspected.
The matched 200-question optimized comparison also completed: 81.60 tokens/sec
baseline versus 101.34 with sampled draft proposals (1.242x). Accuracy was
157/200 and 154/200 respectively. Saved completions were rescored, question
alignment and source/config hashes checked, and engine logs verified BF16,
active compilation and CUDA graphs, with batch invariance off in both timing
arms. The one heuristic answer-audit flag was a valid final answer and needed
no scorer change. Exact output archives, the environment snapshot, an audited
summary, and a real-data plot were retained. This comparison changes both
execution profile and proposal strategy; it does not isolate their individual
effects or establish an accuracy change from the single sample.

