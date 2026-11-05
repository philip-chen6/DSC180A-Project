# Owner: Philip
"""CPU tests for comparison logic, not real speculative decoding correctness."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src import check


# Run the worker in-process with fake generation to test comparison and config flow.
@pytest.fixture
def fake_processes(monkeypatch):
    runs = []
    lifecycle = []

    class Process:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.exitcode = 0

        def start(self):
            lifecycle.append("start")
            self.target(*self.args)

        def join(self):
            lifecycle.append("join")

        def is_alive(self):
            return False

    monkeypatch.setattr(check, "find_spec", lambda name: object())
    monkeypatch.setattr(check.multiprocessing, "get_context",
                        lambda method: SimpleNamespace(Process=Process))
    return runs, lifecycle


# Count full-string differences; do not normalize away whitespace or explanations.
@pytest.mark.parametrize("spec_outputs, count", [(["42", "7"], 0), (["42", "7 "], 1)])
def test_correctness_count_and_configs(fake_processes, monkeypatch, capsys, spec_outputs, count):
    runs, lifecycle = fake_processes

    def generate(prompts, config):
        runs.append((prompts, config.copy()))
        return (spec_outputs if "draft_model" in config else ["42", "7"]), {}

    monkeypatch.setattr(check, "run_generation", generate)
    config = {"target_model": "target", "draft_model": "draft",
              "num_speculative_tokens": 5, "temperature": 0.8,
              "seed": 0, "max_tokens": 512, "n_prompts": 200}
    original = config.copy()
    assert check.check_correctness(["one", "two"], config) == count
    assert config == original
    assert lifecycle == ["start", "join", "start", "join"]
    baseline, speculative = [run_config for _, run_config in runs]
    assert baseline == {key: value for key, value in speculative.items()
                        if key not in ("draft_model", "num_speculative_tokens")}
    assert speculative == dict(original, temperature=0.0, n_prompts=2)
    assert all(prompts == ["one", "two"] for prompts, _ in runs)
    assert f"{count}/2 outputs differ" in capsys.readouterr().out


# Never report a successful comparison after an incomplete generation.
def test_missing_completion_aborts(fake_processes, monkeypatch, capsys):
    monkeypatch.setattr(check, "run_generation", lambda prompts, config: ([], {}))
    with pytest.raises(RuntimeError, match="wrong number of completions"):
        check.check_correctness(["one"], {"draft_model": "draft"})
    assert "outputs differ" not in capsys.readouterr().out


# Explain an absent vLLM environment before trying to launch a GPU worker.
def test_missing_vllm(monkeypatch):
    monkeypatch.setattr(check, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="vLLM is not installed"):
        check.check_correctness(["one"], {"draft_model": "draft"})


# Exercise real process spawning and CLI exit codes with an explicit vLLM stand-in.
@pytest.mark.parametrize("different", [False, True])
def test_check_cli_processes(tmp_path, different):
    (tmp_path / "vllm.py").write_text('''
import os
from types import SimpleNamespace
__version__ = "fake-for-cpu-test"

class SamplingParams:
    def __init__(self, **kwargs):
        assert kwargs["temperature"] == 0.0

class LLM:
    def __init__(self, **kwargs):
        self.speculative = "speculative_config" in kwargs
        print(f"FAKE_ENGINE_PID={os.getpid()}", flush=True)

    def generate(self, prompts, sampling, use_tqdm):
        text = "different" if self.speculative and os.environ["FAKE_DIFFERENT"] == "1" else "same"
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1])])]

    def get_metrics(self):
        return []
''', encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(tmp_path), FAKE_DIFFERENT=str(int(different)))
    result = subprocess.run(
        [sys.executable, str(root / "run.py"), "check"], cwd=tmp_path,
        env=environment, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == int(different), result.stdout + result.stderr
    assert f"{3 if different else 0}/3 outputs differ" in result.stdout
    pids = [line for line in result.stdout.splitlines() if line.startswith("FAKE_ENGINE_PID=")]
    assert len(pids) == len(set(pids)) == 2
