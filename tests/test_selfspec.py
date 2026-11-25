"""CPU tests for the quantized self-speculative draft study (no model downloads)."""

import json
from types import SimpleNamespace

import pytest

from src import selfspec

try:
    import torch
except ImportError:  # CPU-only setups without torch still run the pure-Python tests.
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


# ---------------------------------------------------------------- pure Python

def test_parse_quant():
    assert selfspec.parse_quant("none") == ("none", None)
    assert selfspec.parse_quant("nf4") == ("nf4", None)
    assert selfspec.parse_quant("rtn4") == ("rtn", 4)
    for bad in ("rtn1", "rtn9", "int4", "", None):
        with pytest.raises(ValueError):
            selfspec.parse_quant(bad)


def test_expected_speedup_matches_worked_examples():
    # Illustrative numbers quoted in the proposal discussion.
    assert selfspec.expected_speedup(0.9, 5, 0.6) == pytest.approx(4.6856 / 4.0, rel=1e-3)
    assert selfspec.expected_speedup(0.9, 3, 0.6) == pytest.approx(3.439 / 2.8, rel=1e-3)
    assert selfspec.expected_speedup(0.75, 5, 0.5) < 1.0
    assert selfspec.expected_tokens_per_step(1.0, 4) == 5.0
    assert selfspec.expected_tokens_per_step(0.0, 4) == 1.0
    # Free drafts with perfect acceptance give k + 1 tokens per target step.
    assert selfspec.expected_speedup(1.0, 3, 0.0) == 4.0
    with pytest.raises(ValueError):
        selfspec.expected_tokens_per_step(1.1, 3)
    with pytest.raises(ValueError):
        selfspec.expected_tokens_per_step(0.5, 0)
    with pytest.raises(ValueError):
        selfspec.expected_speedup(0.5, 2, -0.1)


def _make_checkpoints(tmp_path, steps):
    for step in steps:
        folder = tmp_path / f"checkpoint-{step}"
        folder.mkdir()
        (folder / "adapter_config.json").write_text("{}")
    (tmp_path / "final").mkdir()  # ignored: not a numbered checkpoint
    (tmp_path / "checkpoint-999").mkdir()  # ignored: no adapter inside


def test_discover_checkpoints(tmp_path):
    _make_checkpoints(tmp_path, [100, 25, 50])
    found = selfspec.discover_checkpoints(str(tmp_path))
    assert [step for step, _ in found] == [0, 25, 50, 100]
    assert found[0] == (0, None)
    assert found[2][1] == str(tmp_path / "checkpoint-50")
    assert [s for s, _ in selfspec.discover_checkpoints(str(tmp_path), [0, 50])] == [0, 50]
    with pytest.raises(ValueError, match="not found"):
        selfspec.discover_checkpoints(str(tmp_path), [75])
    assert selfspec.discover_checkpoints(str(tmp_path / "missing")) == [(0, None)]


def test_draft_adapter_live_none_and_refresh(tmp_path):
    _make_checkpoints(tmp_path, [25, 50, 75, 100])
    ckpts = selfspec.discover_checkpoints(str(tmp_path))
    path = {step: p for step, p in ckpts}
    live = {"name": "live", "adapter": "live"}
    frozen = {"name": "frozen", "adapter": "none"}
    refresh = {"name": "r", "adapter": {"refresh_every": 50}}
    assert selfspec.draft_adapter(live, 75, ckpts) == path[75]
    assert selfspec.draft_adapter(live, 0, ckpts) is None
    assert selfspec.draft_adapter(frozen, 100, ckpts) is None
    assert selfspec.draft_adapter(refresh, 25, ckpts) is None
    assert selfspec.draft_adapter(refresh, 75, ckpts) == path[50]
    assert selfspec.draft_adapter(refresh, 100, ckpts) == path[100]


def test_validate_variant():
    target = "Qwen/Qwen2.5-1.5B-Instruct"
    selfspec.validate_variant({"name": "ok", "quant": "rtn4", "adapter": "live"}, target)
    selfspec.validate_variant({"name": "small", "model": "small", "adapter": "none"}, target)
    with pytest.raises(ValueError, match="cannot load the policy LoRA"):
        selfspec.validate_variant({"name": "small", "model": "small"}, target)
    with pytest.raises(ValueError, match="refresh_every"):
        selfspec.validate_variant({"name": "r", "adapter": {"refresh_every": 0}}, target)
    with pytest.raises(ValueError, match="adapter must be"):
        selfspec.validate_variant({"name": "x", "adapter": "stale"}, target)
    with pytest.raises(ValueError, match="name"):
        selfspec.validate_variant({"quant": "rtn4"}, target)


def test_cost_ratios_from_draftcost_output(tmp_path):
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({"cost_ratio": {"awq4_lora": 0.6, "qwen0.5b": 0.5}}))
    config = {"variants": [{"name": "a", "cost_key": "awq4_lora"}, {"name": "b"},
                           {"name": "c", "cost_key": "qwen0.5b"}],
              "speedup": {"cost_ratio_path": str(cost), "cost_ratio": {"c": 0.4}}}
    assert selfspec._cost_ratios(config) == {"a": 0.6, "c": 0.4}


# ---------------------------------------------------------------- torch-based

@needs_torch
def test_rtn_quantize_weight_grid_and_error_bound():
    torch.manual_seed(0)
    weight = torch.randn(8, 256)
    for bits in (3, 4, 8):
        quantized = selfspec.rtn_quantize_weight(weight, bits, group_size=128)
        assert quantized.shape == weight.shape and quantized.dtype == weight.dtype
        for row in range(8):
            for group in range(2):
                values = quantized[row, group * 128:(group + 1) * 128]
                original = weight[row, group * 128:(group + 1) * 128]
                assert values.unique().numel() <= 2 ** bits
                step = (original.max() - original.min()) / (2 ** bits - 1)
                assert (values - original).abs().max() <= step + 1e-6
    error4 = (selfspec.rtn_quantize_weight(weight, 4) - weight).abs().mean()
    error8 = (selfspec.rtn_quantize_weight(weight, 8) - weight).abs().mean()
    assert error8 < error4 / 8


@needs_torch
def test_rtn_quantize_weight_falls_back_to_per_row_groups():
    weight = torch.randn(4, 100)  # 100 is not divisible by 128
    quantized = selfspec.rtn_quantize_weight(weight, 4, group_size=128)
    assert all(quantized[row].unique().numel() <= 16 for row in range(4))
    with pytest.raises(ValueError):
        selfspec.rtn_quantize_weight(torch.randn(3), 4)
    with pytest.raises(ValueError):
        selfspec.rtn_quantize_weight(weight, 1)


@needs_torch
def test_rtn_quantize_model_skips_lm_head_and_embeddings():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 128)
            self.proj = torch.nn.Linear(128, 128)
            self.lm_head = torch.nn.Linear(128, 10)

    model = Tiny()
    before = {name: p.clone() for name, p in model.named_parameters()}
    assert selfspec.rtn_quantize_model_(model, 4) == 1
    assert not torch.equal(model.proj.weight, before["proj.weight"])
    assert torch.equal(model.proj.bias, before["proj.bias"])
    assert torch.equal(model.lm_head.weight, before["lm_head.weight"])
    assert torch.equal(model.embed_tokens.weight, before["embed_tokens.weight"])
    with pytest.raises(RuntimeError):
        selfspec.rtn_quantize_model_(torch.nn.Embedding(2, 2), 4)


@needs_torch
def test_token_acceptance_definitions():
    p_logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    q_logits = torch.log(torch.tensor([[0.2, 0.7, 0.1]]))
    # sum min(p, q) = 0.2 + 0.2 + 0.1
    assert selfspec.token_acceptance(p_logits, q_logits, 1.0).item() == pytest.approx(0.5)
    # greedy proposals accept argmax q (token 1) with probability p(1)
    assert selfspec.token_acceptance(p_logits, q_logits, 1.0, "greedy").item() == pytest.approx(0.2)
    assert selfspec.token_acceptance(p_logits, q_logits, 0.0).item() == 0.0
    assert selfspec.token_acceptance(p_logits, p_logits, 1.0).item() == pytest.approx(1.0)
    assert selfspec.token_acceptance(p_logits, p_logits, 0.0).item() == 1.0
    with pytest.raises(ValueError):
        selfspec.token_acceptance(p_logits, q_logits[:, :2], 1.0)
    with pytest.raises(ValueError):
        selfspec.token_acceptance(p_logits, q_logits, 1.0, "beam")


class FakeLM:
    """Next-token logits depend only on the current token: logits = table[token]."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, input_ids, logits_to_keep):
        self.calls.append((input_ids.shape[1], logits_to_keep))
        logits = self.table[input_ids]
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


@needs_torch
def test_measure_acceptance_uses_completion_positions_only():
    torch.manual_seed(0)
    table = torch.randn(6, 6)
    perturbed = table.clone()
    perturbed[3] = torch.randn(6)  # only predictions made *from* token 3 differ
    target, draft = FakeLM(table), FakeLM(perturbed)
    sequences = [([1, 2], [3, 4, 5]), ([2], [1]), ([], [1]), ([1], [])]
    result = selfspec.measure_acceptance(target, draft, sequences, 1.0, device="cpu")
    assert result["n_sequences"] == 2 and result["n_tokens"] == 4
    assert target.calls == [(5, 4), (2, 2)]

    # Positions: tokens 2->3, 3->4, 4->5 in seq 1 and 2->1 in seq 2; only the
    # prediction from token 3 differs between the models.
    differing = selfspec.token_acceptance(table[3], perturbed[3], 1.0).item()
    assert result["alpha"] == pytest.approx((3 + differing) / 4)
    same = selfspec.measure_acceptance(target, FakeLM(table), sequences, 1.0, device="cpu")
    assert same["alpha"] == pytest.approx(1.0) and same["greedy_agreement"] == 1.0
    assert same["alpha_ci95"] == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        selfspec.measure_acceptance(target, draft, [([], [1])], 1.0, device="cpu")


@needs_torch
def test_plot_acceptance(tmp_path):
    pytest.importorskip("matplotlib")
    records = [
        {"step": step, "variant": name, "alpha": alpha, "policy_accuracy": 0.5,
         "alpha_ci95": [alpha - 0.01, alpha + 0.01] if name == "a" else None}
        for name, base in (("a", 0.9), ("b", 0.8)) for step, alpha in ((0, base), (50, base - 0.1))
    ]
    results = tmp_path / "acceptance.json"
    results.write_text(json.dumps({"records": records}))
    selfspec.plot_acceptance(str(results), str(tmp_path / "plot" / "a.png"))
    assert (tmp_path / "plot" / "a.png").stat().st_size > 0
