# Owner: Person 2
"""GSM8K loading and chat prompt formatting."""

from typing import Any


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
    raise NotImplementedError("data.load_gsm8k is not implemented yet (Person 2).")


# Format one question using the model's chat template.
def format_prompt(question: str, tokenizer: Any) -> str:
    """Return a chat-formatted prompt for a GSM8K question.

    Use the supplied tokenizer's chat template without tokenizing the
    result, and include the assistant generation prompt. Instruct the
    model to end its response with the final numeric answer. Keep this
    instruction identical for baseline, speculative, and evaluation runs.
    The caller supplies the tokenizer; this function does not load one.
    """
    raise NotImplementedError("data.format_prompt is not implemented yet (Person 2).")
