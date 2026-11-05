# Owner: Philip
"""CPU tests for generation wiring and measurement; these do not run vLLM."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import generate


# Provide deterministic outputs, timings, and the documented V1 metric shape.
@pytest.fixture
def fake_vllm(monkeypatch):
    config = {
        "target_model": "target",
        "temperature": 0.0,
        "max_tokens": 512,
        "n_prompts": 200,
        "seed": 0,
    }
    llm = Mock()
    llm.generate.side_effect = [
        [SimpleNamespace(outputs=[SimpleNamespace(text="42", token_ids=[1, 2, 3])])],
        [SimpleNamespace(outputs=[SimpleNamespace(text="7", token_ids=[4, 5])])],
    ]
    module = SimpleNamespace(__version__="test-version", LLM=Mock(return_value=llm),
                             SamplingParams=Mock())
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(generate, "perf_counter", Mock(side_effect=[10.0, 12.0, 20.0, 23.0]))
    return config, module, llm


# Construct independent snapshots with duplicate engine labels.
def counters(drafts, proposed, accepted):
    return [SimpleNamespace(name=name, value=value / 2)
            for name, value in zip(generate.SPEC_COUNTERS, (drafts, proposed, accepted))
            for _ in range(2)]


# Verify real output-token counting and per-call timing, not elapsed/n latency.
def test_baseline_measurements(fake_vllm, capsys):
    config, module, llm = fake_vllm
    completions, stats = generate.run_generation(["one", "two"], config)
    assert completions == ["42", "7"]
    assert stats == {
        "config": "baseline", "n_prompts": 2, "tokens_per_sec": 1.0,
        "mean_latency_s": 2.5, "mean_output_len": 2.5,
        "mean_acceptance_length": None, "acceptance_rate": None, "accuracy": None,
    }
    module.LLM.assert_called_once_with(model="target", seed=0, tensor_parallel_size=1,
                                       disable_log_stats=False)
    module.SamplingParams.assert_called_once_with(temperature=0.0, max_tokens=512, seed=0, n=1)
    assert [call.args[0] for call in llm.generate.call_args_list] == [["one"], ["two"]]
    assert all(call.kwargs == {"use_tqdm": False} for call in llm.generate.call_args_list)
    llm.get_metrics.assert_not_called()
    assert "vLLM version: test-version" in capsys.readouterr().out


# Use deltas, sum labels, and include one bonus token per verification step.
def test_speculative_metrics(fake_vllm, capsys):
    config, module, llm = fake_vllm
    config.update(draft_model="draft", num_speculative_tokens=5)
    llm.get_metrics.side_effect = [counters(100, 500, 300), counters(104, 520, 312)]
    _, stats = generate.run_generation(["one", "two"], config)
    assert module.LLM.call_args.kwargs["speculative_config"] == {
        "model": "draft", "num_speculative_tokens": 5,
    }
    assert stats["config"] == "specdec"
    assert stats["mean_acceptance_length"] == 4.0
    assert stats["acceptance_rate"] == pytest.approx(0.6)
    log = capsys.readouterr().out
    assert all(name in log for name in generate.SPEC_COUNTERS)


# Distinguish unavailable metrics and unused speculation from measured rejection.
@pytest.mark.parametrize("case", ["missing_api", "unsupported_api", "missing_counter", "zero"])
def test_unavailable_acceptance_is_null(fake_vllm, case):
    config, _, llm = fake_vllm
    config.update(draft_model="draft", num_speculative_tokens=5)
    if case == "missing_api":
        del llm.get_metrics
    elif case == "unsupported_api":
        llm.get_metrics.side_effect = NotImplementedError
    elif case == "missing_counter":
        llm.get_metrics.return_value = counters(4, 20, 12)[:-2]
    else:
        llm.get_metrics.return_value = counters(0, 0, 0)
    _, stats = generate.run_generation(["one", "two"], config)
    assert stats["mean_acceptance_length"] is None
    assert stats["acceptance_rate"] is None


# A real zero acceptance rate must remain distinguishable from missing metrics.
def test_all_draft_tokens_rejected(fake_vllm):
    config, _, llm = fake_vllm
    config.update(draft_model="draft", num_speculative_tokens=5)
    llm.get_metrics.side_effect = [[], counters(4, 20, 0)]
    _, stats = generate.run_generation(["one", "two"], config)
    assert stats["mean_acceptance_length"] == 1.0
    assert stats["acceptance_rate"] == 0.0


# Reject invalid input before importing vLLM or allocating model memory.
def test_empty_prompts(fake_vllm):
    config, module, _ = fake_vllm
    with pytest.raises(ValueError, match="at least one prompt"):
        generate.run_generation([], config)
    module.LLM.assert_not_called()


# Do not publish invalid acceptance rates if a metric resets during a run.
def test_counter_reset_fails(fake_vllm):
    config, _, llm = fake_vllm
    config.update(draft_model="draft", num_speculative_tokens=5)
    llm.get_metrics.side_effect = [counters(4, 20, 12), counters(1, 5, 3)]
    with pytest.raises(RuntimeError, match="counters reset"):
        generate.run_generation(["one", "two"], config)
