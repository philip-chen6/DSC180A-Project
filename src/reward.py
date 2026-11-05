# Owner: Person 2
"""Numeric answer extraction and correctness rewards."""


# Extract the final numeric answer from a completion.
def extract_answer(completion: str) -> str | None:
    """Return the final numeric answer, or None if no answer is found.

    Handle currency symbols, thousands separators, signs, and decimal
    notation: for example, "$1,200" represents 1200 and "42.0" represents
    42. Prefer an explicitly marked final answer; otherwise use the final
    numeric answer in the completion. Return a normalized numeric string
    suitable for comparison, and None for empty or nonnumeric output.
    """
    raise NotImplementedError("reward.extract_answer is not implemented yet (Person 2).")


# Score a completion against its gold answer.
def reward(completion: str, gold: str) -> float:
    """Return 1.0 for a numerically correct final answer, otherwise 0.0.

    Use extract_answer for the completion and normalize the gold numeric
    string consistently. Equivalent values such as "42" and "42.0" must
    match. Missing or invalid answers receive 0.0. Keep this deterministic
    and independent of generation so it can later serve as the RL reward.
    """
    raise NotImplementedError("reward.reward is not implemented yet (Person 2).")
