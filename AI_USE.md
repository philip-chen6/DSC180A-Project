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
200 questions measured a speedup against the 3B baseline. The full run retained
per-mode completion archives, source/config checksums, audited scores, and an
environment snapshot. Its artifacts and plots were saved in the main project
folder; no source data, reward, or metrics logic changes were needed.

The README records the final results and reproducible setup. These experiments
use one completion per question at concurrency one, without an RL training
loop. Small sampled accuracy differences do not establish a distributional
change, and the heuristic answer parser remains a validation limitation.

