# Owner: Person 2
"""GSM8K loading and chat prompt formatting."""

import random
from typing import Any


# Shared instruction so baseline, speculative, and eval runs see identical prompts.
INSTRUCTION = (
    "Solve the problem step by step. "
    "End your response with the final numeric answer on its own line in the "
    "form '#### <number>', using digits only: no currency symbols, units, "
    "thousands separators, or trailing explanation."
)

# Splits available in the canonical GSM8K config; anything else is a mistake.
SPLITS = ("train", "test")

# Marker that separates GSM8K worked solutions from their final answer.
ANSWER_MARKER = "####"


# Read the final answer field from one raw GSM8K record.
def _final_answer(answer: str, index: int) -> str:
    """Return the text after the last ``####`` marker, or fail loudly.

    A record without the marker or without text after it is malformed data,
    not a case to paper over with a default, so it raises ``ValueError``.
    """
    if ANSWER_MARKER not in answer:
        raise ValueError(f"GSM8K example {index} has no '{ANSWER_MARKER}' marker.")
    final = answer.rsplit(ANSWER_MARKER, 1)[1].strip()
    if not final:
        raise ValueError(f"GSM8K example {index} has an empty final answer.")
    return final


# Load a reproducible sample of GSM8K questions and final answers.
def load_gsm8k(split: str, n: int, seed: int) -> list[dict]:
    """Load ``n`` examples from the requested GSM8K split using ``seed``.

    Return records shaped as {"question": str, "answer": str}, preserving
    the question and extracting only the final number after the ``####``
    marker in the dataset answer. Use deterministic sampling so baseline
    and speculative runs with the same arguments see the same examples
    in the same order. Require a positive ``n`` within the split size.

    Import datasets inside this function when implemented. Importing this
    module must not download data or load a model.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}.")

    # Lazy import: importing src.data must not download the dataset. Convert a
    # missing install into an actionable RuntimeError, as src.check does for vLLM.
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "could not import load_dataset from datasets; run "
            f"'python -m pip install -r requirements.txt' ({error})"
        ) from error

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if n > len(dataset):
        raise ValueError(
            f"n={n} exceeds the {split} split size of {len(dataset)} examples."
        )

    # Sample integer indices rather than relying on dataset shuffling, so the
    # selection depends only on (split, n, seed) and not on the datasets version.
    # Sorting keeps the sampled subset in a stable, readable order.
    indices = random.Random(seed).sample(range(len(dataset)), n)
    indices.sort()

    examples = []
    for index in indices:
        record = dataset[index]
        examples.append(
            {
                "question": record["question"].strip(),
                "answer": _final_answer(record["answer"], index),
            }
        )
    return examples


# Format one question using the model's chat template.
def format_prompt(question: str, tokenizer: Any) -> str:
    """Return a chat-formatted prompt for a GSM8K question.

    Use the supplied tokenizer's chat template without tokenizing the
    result, and include the assistant generation prompt. Instruct the
    model to end its response with the final numeric answer. Keep this
    instruction identical for baseline, speculative, and evaluation runs.
    The caller supplies the tokenizer; this function does not load one.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("format_prompt requires a non-empty question string.")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        raise TypeError("tokenizer must provide apply_chat_template().")

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": question.strip()},
    ]
    prompt = apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("apply_chat_template did not return a prompt string.")
    return prompt
