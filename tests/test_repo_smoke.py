"""CPU smoke tests for the extracted modules.

These tests are deliberately small and synthetic. They verify that the public
API is importable, that the fail-closed paths fail, and that the arithmetic
matches hand-computed values - not that any model checkpoint reproduces a
published number.
"""

from __future__ import annotations

import json

import pytest

from diagnostics.gradient_conflict import (
    cosine,
    decomposition_consistency,
    layer_conflict_table,
    norm_table,
    pair_records_with_skips,
    parse_lora_param,
    summarize_cosines,
)
from diagnostics.selective_abstention import (
    Decision,
    decide,
    evaluate_decisions,
    expected_calibration_error,
    fit_temperature,
    oracle_review_curve,
    progress_feedback,
    risk_coverage_curve,
    risk_scores,
    selective_accuracy,
)
from eval.leak_guard import (
    FrameFingerprint,
    Identity,
    ProtectedIndex,
    ScreenResult,
    dhash64,
    gray32_mean_abs,
    hamming,
    identity_overlap,
    is_near_duplicate,
    load_protected_manifest,
    propagate_group_rejection,
    screen_episode,
)
from eval.metrics_core import (
    CLASSES,
    auc,
    cluster_bootstrap_ci,
    condition_delta,
    full_report,
    hijack_threshold_metrics,
    macro_f1,
    mcnemar,
    validate_predictions,
)
from protocols.qwen_computer_use_v2 import (
    Geometry,
    ToolCallError,
    parse_tool_call,
    render_tool_call,
    to_screenshot_coords,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(sid: str, cls: str, scores: list[float], **extra) -> dict:
    row = {"sid": sid, "cls": cls, "scores": scores}
    row.update(extra)
    return row


def _perfect_rows() -> list[dict]:
    rows = []
    for index, cls in enumerate(CLASSES):
        scores = [0.02, 0.02, 0.02, 0.02, 0.02]
        scores[index] = 0.92
        rows.append(_row(f"s{index}", cls, scores))
    return rows


# ---------------------------------------------------------------------------
# fastPE
# ---------------------------------------------------------------------------


def _build_reference(torch, nn, bias: bool = True):
    class ReferencePatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = 3
            self.temporal_patch_size = 2
            self.patch_size = 3
            self.embed_dim = 7
            self.proj = nn.Conv3d(3, 7, (2, 3, 3), stride=(2, 3, 3), bias=bias).double()

        def forward(self, hidden_states):
            patches = hidden_states.view(-1, 3, 2, 3, 3)
            return self.proj(patches).view(-1, 7)

    return ReferencePatchEmbed()


def test_fastpe_forward_equivalence():
    torch = pytest.importorskip("torch")
    from torch import nn

    from fastpe.fast_patch_embed import FastPatchEmbed

    torch.manual_seed(19)
    for bias in (True, False):
        reference = _build_reference(torch, nn, bias=bias)
        fast = FastPatchEmbed(reference, mode="gemm")
        fallback = FastPatchEmbed(reference, mode="original")
        for shape in ((1, 54), (17, 54), (2, 3, 54)):
            x = torch.randn(*shape, dtype=torch.float64)
            expected = reference(x)
            assert torch.allclose(expected, fast(x), atol=1e-12, rtol=1e-12)
            assert torch.equal(expected, fallback(x))
        assert fast.original.proj.weight is reference.proj.weight


def test_fastpe_rejects_bad_configuration_and_shapes():
    torch = pytest.importorskip("torch")
    from torch import nn

    from fastpe.fast_patch_embed import FastPatchEmbed, apply_fastpe

    reference = _build_reference(torch, nn)
    with pytest.raises(ValueError):
        FastPatchEmbed(reference, mode="int8")
    fast = FastPatchEmbed(reference)
    with pytest.raises(ValueError):
        fast(torch.randn(2, 53))
    with pytest.raises(ValueError):
        fast.set_mode("turbo")

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = _build_reference(torch, nn)

    tower = Tower()
    assert apply_fastpe(tower) == 1
    assert apply_fastpe(tower) == 0  # idempotent
    tower.patch_embed.set_mode("original")
    x = torch.randn(4, 54, dtype=torch.float64)
    assert torch.equal(tower.patch_embed(x), tower.patch_embed.original(x))


def test_parity_audit_accepts_and_escalates():
    torch = pytest.importorskip("torch")

    from fastpe.fast_patch_embed import ParityBudget, audit_parity

    reference = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    same = reference.clone()
    report = audit_parity(reference, same, budget=ParityBudget())
    assert report.passed and report.verdict == "accept"

    flipped = reference.clone()
    flipped[1] = torch.tensor([0.7, 0.2, 0.1])
    report = audit_parity(
        reference,
        flipped,
        reference_risk=[0.2, 0.9],
        risk_threshold=0.5,
        budget=ParityBudget(max_flip_rate=0.4),
    )
    assert report.flips == 1
    assert report.risk_flips == 1
    assert report.verdict == "review_full_precision"


def test_benchmark_harness_runs():
    pytest.importorskip("torch")
    from fastpe.benchmark_parity import BenchmarkConfig, run

    result = run(BenchmarkConfig(patches=16, warmup=1, iterations=3))
    assert result["latency"]["original"]["p50_ms"] > 0
    assert result["latency"]["speedup_p50"] > 0
    assert result["parity_audit"]["n"] == 1


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_coordinate_conversion_is_single_scaled():
    screenshot = Geometry(498, 321)
    viewport = Geometry(332, 214)

    assert to_screenshot_coords([249, 150], screenshot, "pixel") == (249.0, 150.0)
    # 0-999 maps onto the screenshot, not onto the viewport.
    assert to_screenshot_coords([500, 500], screenshot, "normalized") == (249.0, 160.5)
    assert to_screenshot_coords([500, 500], screenshot, "normalized", viewport=viewport) == (
        249.0,
        160.5,
    )

    for bad in ([0, 321], [-1, 0], [0, float("nan")], [1000, 0], None, [1]):
        with pytest.raises(ToolCallError):
            to_screenshot_coords(bad, screenshot, "pixel" if bad in ([0, 321], [-1, 0]) else "normalized")
    with pytest.raises(ToolCallError):
        to_screenshot_coords([1, 1], screenshot, "viewport")


def test_parse_all_actions():
    screenshot = Geometry(498, 321)
    viewport = Geometry(332, 214)
    cases = {
        "left_click": {"action": "left_click", "coordinate": [249, 150]},
        "type": {"action": "type", "text": "hello"},
        "scroll": {"action": "scroll", "coordinate": [249, 150], "pixels": -100},
        "key": {"action": "key", "keys": ["Enter"]},
        "terminate": {"action": "terminate", "status": "success"},
    }
    for name, arguments in cases.items():
        parsed = parse_tool_call(
            render_tool_call(arguments) + "<|im_end|>",
            screenshot,
            space="pixel",
            viewport=viewport,
        )
        assert parsed.action == name
        assert parsed.arguments["action"] == name

    scroll = parse_tool_call(
        render_tool_call(cases["scroll"]), screenshot, space="pixel"
    )
    # Official positive means up; BrowserGym positive means down.
    assert scroll.command == "scroll_at(249.0, 150.0, 0, 100)"
    assert parse_tool_call(render_tool_call(cases["terminate"]), screenshot).terminates


def test_parse_rejects_malformed_calls():
    screenshot = Geometry(498, 321)
    bad_payloads = [
        {},
        "{}",
        "<tool_call>{}</tool_call>",
        '<tool_call>{"name":"bad","arguments":{}}</tool_call>',
        render_tool_call({"action": "left_click", "coordinate": [1, 2]}) + " trailing",
        render_tool_call({"action": "left_click", "coordinate": [1, 2]}) * 2,
        render_tool_call({"action": "key", "keys": ["F5"]}),
        render_tool_call({"action": "key", "keys": ["Enter", "Tab"]}),
        render_tool_call({"action": "type", "text": "x" * 201}),
        render_tool_call({"action": "type", "text": ""}),
        render_tool_call({"action": "scroll", "coordinate": [1, 2], "pixels": 0}),
        render_tool_call({"action": "scroll", "coordinate": [1, 2], "pixels": 900}),
        render_tool_call({"action": "terminate", "status": "maybe"}),
        render_tool_call({"action": "double_click", "coordinate": [1, 2]}),
    ]
    for raw in bad_payloads:
        with pytest.raises(ToolCallError):
            parse_tool_call(raw, screenshot, space="pixel")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_macro_f1_perfect_and_degraded():
    rows = _perfect_rows()
    assert macro_f1(rows) == pytest.approx(1.0)
    assert full_report(rows)["accuracy"] == 1.0

    rows[0]["scores"] = [0.1, 0.7, 0.1, 0.05, 0.05]
    report = full_report(rows, threshold=0.5)
    assert report["accuracy"] == pytest.approx(0.8)
    assert report["classes"][CLASSES[0]]["recall"] == 0.0
    assert 0.0 < report["macro_f1"] < 1.0


def test_auc_and_threshold_regime():
    rows = _perfect_rows()
    positives = [r["scores"][0] for r in rows if r["cls"] == CLASSES[0]]
    negatives = [r["scores"][0] for r in rows if r["cls"] != CLASSES[0]]
    assert auc(positives, negatives) == 1.0
    assert auc(negatives, positives) == 0.0
    assert auc([0.5, 0.5], [0.5]) == 0.5

    metrics = hijack_threshold_metrics(rows, 0.5)
    assert (metrics["tp"], metrics["fp"], metrics["recall"], metrics["fpr"]) == (1, 0, 1.0, 0.0)


def test_validation_and_label_gate():
    rows = _perfect_rows()
    rows[0]["pred"] = "B"
    with pytest.raises(ValueError):
        validate_predictions(rows, _perfect_rows())

    rows = _perfect_rows()
    rows[0]["label_status"] = "sensitivity_only"
    with pytest.raises(ValueError):
        full_report(rows)

    with pytest.raises(ValueError):
        validate_predictions([rows[0], rows[0]], _perfect_rows())


def test_cluster_bootstrap_and_mcnemar():
    values = [1.0, 0.0, 1.0, 1.0]
    keys = ["a", "a", "b", "b"]
    first = cluster_bootstrap_ci(values, keys, replicates=200, seed=7)
    second = cluster_bootstrap_ci(values, keys, replicates=200, seed=7)
    assert first == second
    assert first["n_clusters"] == 2
    assert first["ci95"][0] <= first["point"] <= first["ci95"][1]
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(values, ["a"] * 4)

    base = _perfect_rows()
    variant = _perfect_rows()
    variant[0]["scores"] = [0.1, 0.7, 0.1, 0.05, 0.05]
    result = mcnemar(base, variant)
    assert result["b_only_a"] == 1 and result["c_only_b"] == 0


def test_condition_delta_is_sensitivity_only():
    base = _perfect_rows()
    variant = _perfect_rows()
    variant[0]["scores"] = [0.05, 0.85, 0.04, 0.03, 0.03]
    result = condition_delta(base, variant)
    assert result["summary"]["n_pairs"] == 5
    assert result["summary"]["argmax_flip_rate"] == pytest.approx(0.2)
    assert result["summary"]["metric_class"] == "sensitivity_only"
    assert "accuracy" not in result["summary"]


# ---------------------------------------------------------------------------
# Leak guard
# ---------------------------------------------------------------------------


def _flat_gray(value: float = 128.0) -> list[float]:
    return [value] * (32 * 32)


def _fingerprint(sha: str, gray: list[float], bits: int = 0) -> FrameFingerprint:
    return FrameFingerprint(sha256=sha, dhash=bits, gray32=tuple(gray))


def test_dhash_and_near_duplicate_rule():
    flat = [[0] * 9 for _ in range(8)]
    assert dhash64(flat) == 0
    changed = [row[:] for row in flat]
    changed[0][8] = -1
    assert hamming(dhash64(flat), dhash64(changed)) == 1

    left = _fingerprint("0" * 64, _flat_gray(100.0))
    right = _fingerprint("1" * 64, _flat_gray(101.0))
    assert gray32_mean_abs(left.gray32, right.gray32) == pytest.approx(1.0)
    assert is_near_duplicate(left, right)
    far = _fingerprint("2" * 64, _flat_gray(200.0))
    assert not is_near_duplicate(left, far)


def test_protected_index_and_manifest(tmp_path):
    manifest = {
        "status": "protected_for_test_no_training",
        "episodes": [
            {"episode_id": "ep1", "task": "click-button", "seed": 1},
            {"episode_id": "ep2", "task": "click-button", "seed": 2},
        ],
    }
    path = tmp_path / "protected.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_protected_manifest(path)
    assert loaded.n_instances == 2
    assert len(loaded.sha256) == 64

    manifest["episodes"].append({"episode_id": "ep3", "task": "click-button", "seed": 1})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protected_manifest(path)

    manifest["episodes"].pop()
    manifest["status"] = "draft"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protected_manifest(path)

    index = ProtectedIndex()
    protected_frame = _fingerprint("a" * 64, _flat_gray(90.0))
    index.add(protected_frame)
    assert index.match(protected_frame) == "exact_protected_image"
    assert index.match(_fingerprint("b" * 64, _flat_gray(90.5))) == "near_protected_image"
    assert index.match(_fingerprint("c" * 64, _flat_gray(220.0))) is None


def test_identity_screen_and_group_propagation():
    identity = Identity("click-button", 7)
    protected = {Identity("click-button", 7)}
    index = ProtectedIndex()
    index.add(_fingerprint("d" * 64, _flat_gray(50.0)))

    result = screen_episode("ep-a", identity, [_fingerprint("e" * 64, _flat_gray(50.2))], index, protected)
    assert "protected_task_seed" in result.reasons
    assert "near_protected_image" in result.reasons

    clean = ScreenResult("ep-b", identity, set())
    groups = {identity: ["ep-a", "ep-b"]}
    rejections = propagate_group_rejection([result, clean], groups)
    assert {r["episode_id"] for r in rejections} == {"ep-a", "ep-b"}
    assert rejections[1]["reasons"] == ["related_task_seed_group"]

    assert identity_overlap([{"task": "click-button", "seed": 7}], protected) == [
        {"task": "click-button", "seed": 7}
    ]


# ---------------------------------------------------------------------------
# Selective abstention and calibration
# ---------------------------------------------------------------------------


def test_three_state_priority():
    assert decide(0.99, 0.9) is Decision.FLAG
    assert decide(0.4, 0.1) is Decision.REVIEW
    assert decide(0.9, 0.1) is Decision.ACCEPT
    with pytest.raises(ValueError):
        decide(1.2, 0.1)


def test_progress_feedback_and_risk_scores():
    feedback = progress_feedback([0.9, 0.05, 0.03, 0.02])
    assert feedback["label"] == "progress"
    assert feedback["reward"] == 1.0
    assert feedback["needs_review"] is False

    uncertain = progress_feedback([0.4, 0.3, 0.2, 0.1])
    assert uncertain["label"] is None and uncertain["needs_review"] is True

    with pytest.raises(ValueError):
        progress_feedback([0.5, 0.5])
    with pytest.raises(ValueError):
        risk_scores([0.9, 0.2, 0.0, 0.0, 0.0])

    assert risk_scores([0.5, 0.1, 0.1, 0.1, 0.2]) == {"p1": 0.5, "p1p5": 0.7}


def test_review_rows_are_not_counted_correct():
    labels = [0, 1, 2]
    predictions = [0, 2, 2]
    decisions = [Decision.ACCEPT, Decision.REVIEW, Decision.ACCEPT]
    report = evaluate_decisions(labels, predictions, decisions)
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["accept_accuracy"] == 1.0
    assert report["errors_in_reviewed"] == 1
    assert report["oracle_remaining_errors"] == 0
    assert report["hijack_total"] == 1


def test_risk_coverage_and_oracle_curves():
    errors = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.9, 0.8]
    points, aurc = risk_coverage_curve(errors, scores)
    assert points[0] == (0.25, 0.0)
    assert points[-1][1] == pytest.approx(0.5)
    assert 0.0 < aurc < 0.5
    assert selective_accuracy(errors, scores, 0.5)["selective_accuracy"] == 1.0

    oracle_points, mean_risk = oracle_review_curve(errors, scores)
    assert oracle_points[0][2] == 1  # one error resolved by reviewing the top row
    assert mean_risk < aurc


def test_calibration_metrics():
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0
    assert expected_calibration_error([1.0, 1.0], [0, 0]) == 1.0

    overconfident = [[10.0, 0.0]] * 4
    labels = [0, 0, 1, 1]
    fit = fit_temperature(overconfident, labels)
    assert fit["temperature"] > 1.0
    assert fit["nll_after"] <= fit["nll_before"] + 1e-12


# ---------------------------------------------------------------------------
# Gradient conflict probe
# ---------------------------------------------------------------------------


def test_cosine_and_parameter_parsing():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) is None
    with pytest.raises(ValueError):
        cosine([float("nan")], [1.0])

    name = "base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight"
    assert parse_lora_param(name) == (12, "q_proj", "lora_A")
    assert parse_lora_param("lm_head.weight") == (None, None, None)


def test_conflict_aggregation_and_norm_table():
    pairs = [
        {"a": "layers.0.self_attn.q_proj.lora_A.weight",
         "b": "layers.0.self_attn.q_proj.lora_A.weight", "cos": -0.4},
        {"a": "layers.0.self_attn.q_proj.lora_B.weight",
         "b": "layers.0.self_attn.q_proj.lora_B.weight", "cos": 0.2},
        {"a": "layers.1.self_attn.v_proj.lora_A.weight",
         "b": "layers.1.self_attn.v_proj.lora_A.weight", "cos": None},
    ]
    summary = summarize_cosines(pairs, group="all")
    assert summary["n"] == 2 and summary["n_skipped"] == 1
    assert summary["conflict_ratio"] == 0.5

    table = layer_conflict_table(pairs)
    assert table["by_layer"]["0"]["n"] == 2
    assert table["n_unparsed"] == 0

    norms = {
        "layers.0.self_attn.q_proj.lora_A.weight": 3.0,
        "layers.0.self_attn.q_proj.lora_B.weight": 4.0,
    }
    table = norm_table(norms)
    assert table["total"] == pytest.approx(5.0)
    assert table["by_layer"]["0"] == pytest.approx(5.0)

    check = decomposition_consistency([1.0, 2.0], [0.5, -0.5], [1.5, 1.5], 1.0)
    assert check["ok"] and check["max_abs_error"] == pytest.approx(0.0)

    records, skipped = pair_records_with_skips({"a": [1.0], "b": [1.0]}, {"a": [2.0]})
    assert len(records) == 1 and skipped == ["b"]
