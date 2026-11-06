# Owner: Person 2
"""CPU tests for answer extraction, rewards, and GSM8K data handling."""

import sys
from types import SimpleNamespace

import pytest

from src import data
from src.reward import extract_answer, reward


# Verify a plain, already-normalized answer scores as correct.
def test_exact_integer() -> None:
    """An exact numeric match scores 1.0."""
    assert reward("42", "42") == 1.0


# Equivalent spellings of the same value must not be penalized.
@pytest.mark.parametrize(
    "completion,gold",
    [
        ("42.0", "42"),
        ("42", "42.00"),
        ("-3.00", "-3"),
        ("+7", "7"),
        (".5", "0.5"),
        ("0.50", "0.5"),
    ],
)
def test_equivalent_numbers(completion: str, gold: str) -> None:
    """Decimal, sign, and trailing-zero variants compare equal."""
    assert reward(completion, gold) == 1.0


# Cover the messy formatting that models actually emit.
@pytest.mark.parametrize(
    "completion,gold",
    [
        ("The total is $1,200.", "1200"),
        ("#### $1,200", "1200"),
        ("#### 1,200.50", "1200.5"),
        ("It costs 18", "18"),
    ],
)
def test_currency_and_thousands(completion: str, gold: str) -> None:
    """Currency symbols and thousands separators are normalized away."""
    assert reward(completion, gold) == 1.0


# Different values, including near misses, must not match.
@pytest.mark.parametrize(
    "completion,gold",
    [
        ("43", "42"),
        ("1200", "120"),
        ("-3", "3"),
        ("42.5", "42"),
    ],
)
def test_wrong_numbers(completion: str, gold: str) -> None:
    """Incorrect or partially matching numbers score 0.0."""
    assert reward(completion, gold) == 0.0


# An explicit marker beats numbers mentioned earlier in the reasoning.
def test_marker_is_preferred() -> None:
    """The value after '####' wins over earlier numbers in the text."""
    completion = "I started with 5 apples, then got 2 more. #### 7"
    assert extract_answer(completion) == "7"
    assert reward(completion, "7") == 1.0
    assert reward(completion, "5") == 0.0


# Other explicit markers are honored when the model skips '####'.
@pytest.mark.parametrize(
    "completion,expected",
    [
        ("\\boxed{7}", "7"),
        ("The final answer is 1,200.", "1200"),
        ("Answer: -4", "-4"),
    ],
)
def test_alternate_markers(completion: str, expected: str) -> None:
    """Boxed answers and prose answer markers are recognized."""
    assert extract_answer(completion) == expected


# A marker with no number after it falls back to the last number in the text.
def test_marker_without_number_falls_back() -> None:
    """An empty marker does not discard an otherwise numeric completion."""
    assert extract_answer("I counted 12 items. #### unclear") == "12"


# Empty, whitespace, and prose-only completions have no answer.
@pytest.mark.parametrize(
    "completion", ["", "   ", "I cannot solve this.", "no idea"]
)
def test_missing_answer(completion: str) -> None:
    """Nonnumeric output yields no extraction and zero reward."""
    assert extract_answer(completion) is None
    assert reward(completion, "42") == 0.0


# Normalization must be stable and canonical, not merely numeric.
@pytest.mark.parametrize(
    "completion,expected",
    [
        ("#### 1,200", "1200"),
        ("#### 42.0", "42"),
        ("#### -0.0", "0"),
        ("#### .5", "0.5"),
        ("#### 4.2e1", "42"),
        ("#### $1,200", "1200"),
    ],
)
def test_extraction_normalization(completion: str, expected: str) -> None:
    """Extracted answers use one canonical string per value."""
    assert extract_answer(completion) == expected


# Gold answers may still carry their raw '####' line.
def test_gold_with_marker() -> None:
    """A raw GSM8K gold string is normalized like a completion."""
    assert reward("#### 18", "#### 18") == 1.0
    assert reward("18.0", "#### 18") == 1.0


# Invalid gold or completion types are programming errors, not zero reward.
def test_type_errors() -> None:
    """Non-string inputs raise instead of silently scoring zero."""
    with pytest.raises(TypeError):
        reward(None, "42")
    with pytest.raises(TypeError):
        reward("42", None)
    with pytest.raises(TypeError):
        extract_answer(5)


# Rewards must be reproducible for the same inputs.
def test_reward_is_deterministic() -> None:
    """Repeated calls return the same score."""
    completion = "So the answer is $1,200. #### 1,200"
    assert reward(completion, "1200") == reward(completion, "1200") == 1.0


# Build a stand-in GSM8K split without touching the network.
def _fake_dataset(size: int = 10) -> list[dict]:
    return [
        {"question": f"question {index}?", "answer": f"some work {index}\n#### {index * 3}"}
        for index in range(size)
    ]


# Install a fake datasets module so load_gsm8k runs on CPU only.
@pytest.fixture
def fake_datasets(monkeypatch):
    calls = []

    def load_dataset(name, config=None, split=None):
        calls.append({"name": name, "config": config, "split": split})
        return _fake_dataset()

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    return calls


# Loading must return question/answer pairs with the '####' text removed.
def test_load_gsm8k_records(fake_datasets) -> None:
    """Records expose the question and only the final numeric answer."""
    examples = data.load_gsm8k("test", 3, 0)
    assert len(examples) == 3
    assert set(examples[0]) == {"question", "answer"}
    for example in examples:
        assert example["question"].startswith("question ")
        assert example["answer"].isdigit()
    assert fake_datasets == [
        {"name": "openai/gsm8k", "config": "main", "split": "test"}
    ]


# The same arguments must yield the same examples in the same order.
def test_load_gsm8k_is_reproducible(fake_datasets) -> None:
    """Seed and n fully determine the sample, and the full split stays sorted."""
    assert data.load_gsm8k("test", 4, 0) == data.load_gsm8k("test", 4, 0)
    full = data.load_gsm8k("test", 10, 0)
    assert [example["question"] for example in full] == [
        f"question {index}?" for index in range(10)
    ]
    assert [example["answer"] for example in full] == [
        str(index * 3) for index in range(10)
    ]


# Invalid sampling arguments must fail before any download or silent default.
@pytest.mark.parametrize(
    "split,n,match",
    [
        ("validation", 3, "split must be"),
        ("test", 0, "n must be positive"),
        ("test", -1, "n must be positive"),
        ("test", 11, "exceeds"),
    ],
)
def test_load_gsm8k_validation(fake_datasets, split, n, match) -> None:
    """Bad splits and out-of-range counts raise ValueError."""
    with pytest.raises(ValueError, match=match):
        data.load_gsm8k(split, n, 0)


# Malformed dataset rows must be reported instead of defaulting to zero.
def test_load_gsm8k_rejects_malformed_answer(monkeypatch) -> None:
    """A record without '####' raises ValueError naming the example."""
    broken = [{"question": "q?", "answer": "no marker here"}]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: broken),
    )
    with pytest.raises(ValueError, match="no '####' marker"):
        data.load_gsm8k("test", 1, 0)


# A missing datasets install must fail as a clean RuntimeError, not a traceback.
def test_load_gsm8k_reports_missing_datasets(monkeypatch) -> None:
    """An import failure tells the caller how to install the dependency."""
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace())
    with pytest.raises(RuntimeError, match="pip install -r requirements.txt"):
        data.load_gsm8k("test", 1, 0)


# Capture how format_prompt talks to the tokenizer.
class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "<formatted prompt>"


# Formatting must use the chat template untokenized, with a generation prompt.
def test_format_prompt_uses_chat_template() -> None:
    """The instruction and question are sent as system/user messages."""
    tokenizer = RecordingTokenizer()
    prompt = data.format_prompt("  What is 2+2?  ", tokenizer)
    assert prompt == "<formatted prompt>"
    call = tokenizer.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert [message["role"] for message in call["messages"]] == ["system", "user"]
    assert call["messages"][0]["content"] == data.INSTRUCTION
    assert "####" in data.INSTRUCTION
    assert call["messages"][1]["content"] == "What is 2+2?"


# Empty questions and tokenizers without a chat template are rejected.
def test_format_prompt_validation() -> None:
    """Missing questions or chat templates raise instead of producing junk."""
    tokenizer = RecordingTokenizer()
    with pytest.raises(ValueError, match="non-empty question"):
        data.format_prompt("   ", tokenizer)
    with pytest.raises(TypeError, match="apply_chat_template"):
        data.format_prompt("What is 2+2?", object())
