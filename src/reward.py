# Owner: Person 2
"""Numeric answer extraction and correctness rewards.

Only decimal notation is compared. Fractions such as "3/4" are not
interpreted, so write answers as "0.75" on both sides of a comparison.
"""

import re
from decimal import Decimal, InvalidOperation


# Currency markers that wrap the number but carry no numeric meaning here.
CURRENCY_SYMBOLS = ("$", "€", "£", "¥")

# Dash code points models emit instead of ASCII minus, plus the no-break space.
MINUS_VARIANTS = ("\u2212", "\u2013", "\u2014", "\u2010")
NO_BREAK_SPACE = "\u00a0"

# One number with an optional currency symbol on either side of an optional
# sign, thousands separators, a decimal part, and an exponent. Requiring
# three-digit groups after a comma is what keeps "12, 34" from becoming 1234.
NUMBER_PATTERN = re.compile(
    r"[$€£¥]?\s*[-+\u2212\u2013\u2014]?\s*[$€£¥]?\s*"
    r"(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)

# Explicit answer markers, highest confidence first. The prompt asks for
# "#### <number>", so that marker wins over generic prose such as "answer is".
MARKER_PATTERNS = (
    re.compile(r"####"),
    re.compile(r"\\boxed\s*\{"),
    re.compile(r"final\s+answer\s*(?:is|:|=)", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:|=)", re.IGNORECASE),
)

# Refuse absurd exponents so formatting a value cannot allocate huge strings.
MAX_MAGNITUDE = 30


# Convert one matched number token into a canonical comparison string.
def _normalize_number(text: str) -> str | None:
    """Return a canonical numeric string such as "1200" or "-3.5", else None.

    Strips currency symbols, thousands separators, and unicode minus
    variants, then canonicalizes through ``Decimal`` so that "42.0",
    "42", "+42", and "4.2e1" all collapse to the same string. Non-finite,
    empty, and out-of-range values return None instead of a guess.
    """
    candidate = text
    for symbol in CURRENCY_SYMBOLS:
        candidate = candidate.replace(symbol, "")
    for dash in MINUS_VARIANTS:
        candidate = candidate.replace(dash, "-")
    candidate = candidate.replace(",", "")
    candidate = candidate.replace(" ", "").replace(NO_BREAK_SPACE, "")
    if candidate.startswith("+"):
        candidate = candidate[1:]
    if not candidate:
        return None

    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value != 0 and abs(value.adjusted()) > MAX_MAGNITUDE:
        return None

    normalized = format(value.normalize(), "f")
    if normalized.startswith("-") and Decimal(normalized) == 0:
        normalized = normalized[1:]
    return normalized


# Read the first usable number in a span of text.
def _first_number(text: str) -> str | None:
    """Return the normalized first number in ``text``, or None."""
    for match in NUMBER_PATTERN.finditer(text):
        value = _normalize_number(match.group())
        if value is not None:
            return value
    return None


# Extract the final numeric answer from a completion.
def extract_answer(completion: str) -> str | None:
    """Return the final numeric answer, or None if no answer is found.

    Handle currency symbols, thousands separators, signs, and decimal
    notation: for example, "$1,200" represents 1200 and "42.0" represents
    42. Prefer an explicitly marked final answer; otherwise use the final
    numeric answer in the completion. Return a normalized numeric string
    suitable for comparison, and None for empty or nonnumeric output.
    """
    if not isinstance(completion, str):
        raise TypeError(f"completion must be a str, got {type(completion).__name__}.")
    if not completion.strip():
        return None

    text = completion.replace(NO_BREAK_SPACE, " ")

    # Use the last occurrence of each marker, then the first number after it:
    # repeated "####" lines should agree, and text after the number (such as
    # "(see step 3)") must not be mistaken for the answer.
    for marker in MARKER_PATTERNS:
        matches = list(marker.finditer(text))
        if not matches:
            continue
        value = _first_number(text[matches[-1].end():])
        if value is not None:
            return value

    # No explicit marker: fall back to the final number in the completion.
    numbers = [
        value
        for value in (
            _normalize_number(match.group()) for match in NUMBER_PATTERN.finditer(text)
        )
        if value is not None
    ]
    return numbers[-1] if numbers else None


# Score a completion against its gold answer.
def reward(completion: str, gold: str) -> float:
    """Return 1.0 for a numerically correct final answer, otherwise 0.0.

    Use extract_answer for the completion and normalize the gold numeric
    string consistently. Equivalent values such as "42" and "42.0" must
    match. Missing or invalid answers receive 0.0. Keep this deterministic
    and independent of generation so it can later serve as the RL reward.
    """
    if not isinstance(completion, str):
        raise TypeError(f"completion must be a str, got {type(completion).__name__}.")
    if not isinstance(gold, str):
        raise TypeError(f"gold must be a str, got {type(gold).__name__}.")

    # Gold may arrive as a bare number or still carry its "####" line.
    expected = _normalize_number(gold) or extract_answer(gold)
    predicted = extract_answer(completion)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0
