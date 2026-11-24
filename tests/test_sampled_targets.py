"""Sampled CLI/config coverage, using explicit CPU-only provider stand-ins."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run


ROOT = Path(__file__).resolve().parents[1]


def _sample_pair(tmp_path):
    baseline = {"target_model": "target", "temperature": 1.0, "top_p": 1.0,
                "max_tokens": 512, "n_prompts": 2, "seed": 0,
                "results_path": str(tmp_path / "results" / "results.json")}
    speculative = dict(baseline, draft_model="draft", num_speculative_tokens=5)
    paths = []
    for mode, config in (("baseline_sample", baseline), ("specdec_sample", speculative)):
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        paths.append(path)
    return paths


@pytest.mark.parametrize("mode", ["baseline", "specdec"])
def test_sample_configs_preserve_greedy_settings(mode):
    greedy = json.loads((ROOT / "config" / f"{mode}.json").read_text())
    sampled = json.loads((ROOT / "config" / f"{mode}_sample.json").read_text())
    assert greedy["temperature"] == 0.0
    assert sampled == dict(greedy, temperature=1.0, top_p=1.0)


@pytest.mark.parametrize("target", ["baseline_sample", "specdec_sample"])
def test_sampled_target_selects_own_config(tmp_path, monkeypatch, target):
    paths = _sample_pair(tmp_path)
    (tmp_path / "config").mkdir()
    selected = paths[int(target == "specdec_sample")]
    config = json.loads(selected.read_text())
    (tmp_path / "config" / f"{target}.json").write_text(json.dumps(config))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_gsm8k", Mock(return_value=[{"question": "one", "answer": "42"}]))
    monkeypatch.setattr(run, "format_prompt", lambda question, tokenizer: question)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda model: object())))
    generate = Mock(return_value=(["#### 42"], {"config": target, "accuracy": None}))
    monkeypatch.setattr(run, "run_generation", generate)
    save = Mock()
    monkeypatch.setattr(run, "save_results", save)
    assert run.main([target]) == 0
    assert generate.call_args.args == (["one"], config)
    assert save.call_args.args[0]["config"] == target


@pytest.mark.parametrize("problem", ["temperature", "seed"])
def test_sampled_eval_rejects_unmatched_configs(tmp_path, monkeypatch, problem):
    baseline, speculative = _sample_pair(tmp_path)
    config = json.loads(speculative.read_text())
    config[problem] = 0.0 if problem == "temperature" else 1
    speculative.write_text(json.dumps(config))
    launch = Mock()
    monkeypatch.setattr(run.subprocess, "run", launch)
    with pytest.raises(ValueError, match="positive|must match"):
        run.main(["eval_sample", "--baseline-config", str(baseline),
                  "--specdec-config", str(speculative)])
    launch.assert_not_called()


def test_sampled_eval_aborts_after_failed_child(tmp_path, monkeypatch):
    baseline, speculative = _sample_pair(tmp_path)
    launch = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(run.subprocess, "run", launch)
    assert run.main(["eval_sample", "--baseline-config", str(baseline),
                     "--specdec-config", str(speculative)]) == 7
    assert launch.call_count == 1


@pytest.mark.parametrize("omit_baseline_top_p", [False, True])
@pytest.mark.parametrize("draft_method", [None, "probabilistic"])
def test_sampled_eval_cli_scores_both_and_releases_engines(tmp_path, omit_baseline_top_p, draft_method):
    baseline, speculative = _sample_pair(tmp_path)
    if draft_method is not None:
        config = json.loads(speculative.read_text())
        config["draft_sample_method"] = draft_method
        speculative.write_text(json.dumps(config))
    if omit_baseline_top_p:
        config = json.loads(baseline.read_text())
        config.pop("top_p")
        baseline.write_text(json.dumps(config))
    (tmp_path / "datasets.py").write_text('''
def load_dataset(name, config, split):
    return [{"question": "one", "answer": "work #### 42"},
            {"question": "two", "answer": "work #### 7"}]
''')
    (tmp_path / "transformers.py").write_text('''
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, model):
        return cls()
    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]
''')
    (tmp_path / "vllm.py").write_text('''
import atexit
import os
from pathlib import Path
from types import SimpleNamespace
__version__ = "fake-for-cpu-test"

class SamplingParams:
    def __init__(self, **kwargs):
        assert kwargs == {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "seed": 0, "n": 1}

class LLM:
    def __init__(self, **kwargs):
        lock = Path(os.environ["FAKE_ENGINE_LOCK"])
        with lock.open("x") as file:
            file.write(str(os.getpid()))
        atexit.register(lock.unlink)
        print(f"FAKE_ENGINE_PID={os.getpid()}", flush=True)
        self.speculative = "speculative_config" in kwargs
        self.calls = 0
    def generate(self, prompts, sampling, use_tqdm):
        self.calls += 1
        answer = "42" if prompts[0] == "one" else ("7" if self.speculative else "8")
        return [SimpleNamespace(outputs=[SimpleNamespace(text="#### " + answer, token_ids=[1, 2])])]
    def get_metrics(self):
        names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
                 "vllm:spec_decode_num_accepted_tokens")
        return [SimpleNamespace(name=name, value=value) for name, value in
                zip(names, (self.calls, self.calls * 5, self.calls * 3))]
''')
    archive = tmp_path / "pair.json"
    lock = tmp_path / "engine-active"
    env = dict(os.environ, PYTHONPATH=str(tmp_path), FAKE_ENGINE_LOCK=str(lock))
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "eval_sample",
         "--baseline-config", str(baseline), "--specdec-config", str(speculative),
         "--outputs", str(archive)], cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not lock.exists()
    pids = [line for line in result.stdout.splitlines() if line.startswith("FAKE_ENGINE_PID=")]
    assert len(pids) == len(set(pids)) == 2
    records = json.loads((tmp_path / "results/results.json").read_text())
    assert [record["config"] for record in records] == ["baseline_sample", "specdec_sample"]
    assert [record["accuracy"] for record in records] == [0.5, 1.0]
    assert records[0]["acceptance_rate"] is None
    assert records[1]["acceptance_rate"] == pytest.approx(0.6)
    pair = json.loads(archive.read_text())["runs"]
    assert pair["baseline_sample"]["examples"][1]["completion"] == "#### 8"
    assert pair["specdec_sample"]["examples"][1]["completion"] == "#### 7"


def test_plot_preserves_four_configuration_labels(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from src.metrics import plot_comparison, save_results

    labels = ["baseline", "specdec", "baseline_sample", "specdec_sample"]
    path = tmp_path / "results.json"
    for label in labels:
        save_results({"config": label, "n_prompts": 2, "tokens_per_sec": 1.0,
                      "mean_latency_s": 1.0, "mean_output_len": 1.0,
                      "mean_acceptance_length": None, "acceptance_rate": None,
                      "accuracy": None}, str(path))
    try:
        plot_comparison(str(path), str(tmp_path / "comparison.png"))
        assert [tick.get_text() for tick in plt.gcf().axes[0].get_xticklabels()] == labels
        assert (tmp_path / "comparison.png").stat().st_size > 0
    finally:
        plt.close("all")
