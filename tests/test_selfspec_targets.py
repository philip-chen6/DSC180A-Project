"""CPU tests for GRPO reward wiring, LoRA generation options, and new run.py targets."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run
from src import generate, train_grpo


def test_grpo_reward_function_scores_plain_and_conversational_completions():
    completions = ["work... #### 42", [{"role": "assistant", "content": "#### 8"}], "no answer"]
    assert train_grpo.gsm8k_correctness(completions, answer=["42", "7", "3"],
                                        prompts=["a", "b", "c"]) == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        train_grpo.gsm8k_correctness(["#### 1"], answer=[])
    with pytest.raises(TypeError):
        train_grpo.gsm8k_correctness([42], answer=["42"])


def test_draft_quantization_is_a_draft_setting():
    # Baseline configs are derived by removing these keys (check, eval_sample).
    assert "draft_quantization" in generate.DRAFT_CONFIG_KEYS
    assert "lora_adapter" not in generate.DRAFT_CONFIG_KEYS


def test_generation_serves_lora_policy_and_quantized_draft(tmp_path, monkeypatch):
    adapter = tmp_path / "checkpoint-25"
    adapter.mkdir()
    llm = Mock()
    llm.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="1", token_ids=[1])])]
    llm.get_metrics.return_value = []
    module = SimpleNamespace(__version__="test", LLM=Mock(return_value=llm), SamplingParams=Mock())
    request = Mock(return_value="lora-request")
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=request))
    config = {"target_model": "target", "draft_model": "target-awq", "num_speculative_tokens": 3,
              "draft_quantization": "awq_marlin", "temperature": 1.0, "max_tokens": 8, "seed": 0,
              "lora_adapter": str(adapter), "max_lora_rank": 32}

    generate.run_generation(["p"], config)
    args = module.LLM.call_args.kwargs
    assert args["enable_lora"] is True and args["max_lora_rank"] == 32
    assert args["speculative_config"] == {"model": "target-awq", "num_speculative_tokens": 3,
                                          "quantization": "awq_marlin"}
    request.assert_called_once_with("policy", 1, str(adapter))
    assert llm.generate.call_args.kwargs == {"use_tqdm": False, "lora_request": "lora-request"}

    with pytest.raises(RuntimeError, match="not found"):
        generate.run_generation(["p"], dict(config, lora_adapter=str(tmp_path / "missing")))


@pytest.mark.parametrize("target,module_name,function", [
    ("train", "src.train_grpo", "train_grpo"),
    ("acceptance", "src.selfspec", "run_acceptance_study"),
])
def test_study_targets_dispatch_with_default_configs(tmp_path, monkeypatch, target, module_name, function):
    config = {"output_dir": "checkpoints/grpo", "marker": target}
    path = tmp_path / run.STUDY_CONFIGS[target]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    called = Mock()
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{function: called}))
    assert run.main([target]) == 0
    received = called.call_args.args[0]
    assert received["marker"] == target
    if target == "train":
        assert received["output_dir"] == str(tmp_path / "checkpoints/grpo")
    else:
        assert called.call_args.kwargs == {"root": tmp_path}


def test_study_targets_reject_outputs():
    with pytest.raises(SystemExit):
        run.main(["acceptance", "--outputs", "x.json"])


def test_draftcost_times_each_model_and_writes_cost_ratios(tmp_path, monkeypatch):
    config = {
        "target": {"target_model": "target", "temperature": 1.0, "n_prompts": 2, "seed": 0},
        "lora_adapter": "checkpoints/grpo/checkpoint-200",
        "drafts": {"awq4_lora": {"model": "target-awq", "use_lora": True},
                   "small": {"model": "small", "quantization": "fp8"}},
        "results_path": "results/selfspec/draft_cost.json",
    }
    (tmp_path / "cost.json").write_text(json.dumps(config))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    speeds = {"target": 100.0, "target-awq": 160.0, "small": 200.0}
    seen = []

    def fake_child(command, cwd):
        child = json.loads(open(command[command.index("--config") + 1]).read())
        seen.append(child)
        assert command[2] == "baseline"
        stats = {"tokens_per_sec": speeds[child["target_model"]]}
        with open(command[command.index("--outputs") + 1], "w") as file:
            json.dump({"stats": stats}, file)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run.subprocess, "run", fake_child)
    assert run.main(["draftcost", "--config", "cost.json"]) == 0
    saved = json.loads((tmp_path / config["results_path"]).read_text())
    assert saved["cost_ratio"] == {"awq4_lora": pytest.approx(0.625), "small": 0.5}
    assert [child["target_model"] for child in seen] == ["target", "target-awq", "small"]
    assert seen[0]["lora_adapter"] == seen[1]["lora_adapter"] == config["lora_adapter"]
    assert "lora_adapter" not in seen[2] and seen[2]["quantization"] == "fp8"
    assert all("draft_model" not in child for child in seen)


def test_draftcost_rejects_draft_settings_in_target(tmp_path, monkeypatch):
    config = {"target": {"target_model": "t", "draft_model": "d"}, "drafts": {},
              "results_path": "r.json"}
    (tmp_path / "cost.json").write_text(json.dumps(config))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="draft settings"):
        run.main(["draftcost", "--config", "cost.json"])


def test_selfspec_configs_are_consistent():
    root = run.ROOT / "config"
    specdec = json.loads((root / "selfspec_sample.json").read_text())
    baseline = json.loads((root / "selfspec_baseline_sample.json").read_text())
    # eval_sample requires the pair to match except for draft settings.
    assert baseline == {k: v for k, v in specdec.items() if k not in generate.DRAFT_CONFIG_KEYS}
    acceptance = json.loads((root / "selfspec_acceptance.json").read_text())
    from src.selfspec import validate_variant
    for variant in acceptance["variants"]:
        validate_variant(variant, acceptance["target_model"])
    train = json.loads((root / "train_grpo.json").read_text())
    grpo = train["grpo"]
    generation_batch = grpo["per_device_train_batch_size"] * grpo["gradient_accumulation_steps"]
    assert generation_batch % grpo["num_generations"] == 0
    assert train["lora"]["r"] <= specdec["max_lora_rank"]
