"""CPU coverage for custom benchmark configs and speculative evaluation."""

import json
import os
from unittest.mock import Mock
from types import SimpleNamespace
import sys

import pytest

import run


def test_eval_speculative_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config = {"target_model": "target", "draft_model": "draft", "n_prompts": 2,
              "seed": 0, "results_path": "results/results.json"}
    (tmp_path / "custom.json").write_text(json.dumps(config))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_gsm8k", Mock(return_value=[
        {"question": "one", "answer": "42"}, {"question": "two", "answer": "7"}]))
    monkeypatch.setattr(run, "format_prompt", lambda question, tokenizer: question)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda model: object())))
    generation = Mock(return_value=(["#### 42", "#### 8"], {"config": "specdec", "accuracy": None}))
    monkeypatch.setattr(run, "run_generation", generation)
    save = Mock()
    monkeypatch.setattr(run, "save_results", save)

    assert run.main(["eval", "--config", "custom.json", "--outputs", "results/outputs.json"]) == 0
    assert generation.call_args.args == (["one", "two"], config)
    assert save.call_args.args[0] == {"config": "specdec", "accuracy": 0.5}
    outputs = json.loads((tmp_path / "results/outputs.json").read_text())
    assert outputs["runtime"] == {"VLLM_BATCH_INVARIANT": "1"}
    assert outputs["examples"][1] == {"question": "two", "answer": "7", "completion": "#### 8"}
    assert json.loads((tmp_path / "custom.json").read_text()) == config


@pytest.mark.parametrize("target,config", [("baseline", {"draft_model": "draft"}), ("specdec", {})])
def test_reject_wrong_generation_mode(tmp_path, monkeypatch, target, config):
    (tmp_path / "custom.json").write_text(json.dumps(config))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    generation = Mock()
    monkeypatch.setattr(run, "run_generation", generation)
    with pytest.raises(ValueError, match="requires a config"):
        run.main([target, "--config", "custom.json"])
    generation.assert_not_called()


@pytest.mark.parametrize("original", [None, "0", "1"])
def test_check_uses_batch_invariance_without_changing_benchmark_environment(monkeypatch, original):
    if original is None:
        monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    else:
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", original)

    def compare(prompts, config, report_path):
        assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
        return 0

    monkeypatch.setattr(run, "check_correctness", compare)
    assert run.main(["check"]) == 0
    assert os.environ.get("VLLM_BATCH_INVARIANT") == original
