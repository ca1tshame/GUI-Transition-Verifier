"""Unit tests covering previously untested branches and edge cases across core modules.

All tests are deterministic, pure-CPU, and synthetic.
"""

from __future__ import annotations

import pytest

from diagnostics.gradient_conflict import (
    cosine,
    decomposition_consistency,
    group_by,
    layer_conflict_table,
    lora_grad_vectors,
    pair_records,
    pair_records_with_skips,
    restore_requires_grad,
    snapshot_requires_grad,
)
from diagnostics.selective_abstention import (
    Decision,
    decide,
    evaluate_decisions,
)
from eval.leak_guard import (
    FrameFingerprint,
    is_near_duplicate,
)
from eval.metrics_core import (
    CLASSES,
    paired_cluster_bootstrap,
    recall_at_fpr,
    risk_auc_suite,
    threshold_sweep,
)
from protocols.qwen_computer_use_v2 import (
    MAX_SCROLL_PIXELS,
    Geometry,
    ProtocolError,
    ToolCallError,
    parse_tool_call,
    render_tool_call,
    to_screenshot_coords,
)


# ---------------------------------------------------------------------------
# 1. diagnostics.selective_abstention blind spots
# ---------------------------------------------------------------------------


def test_decide_simultaneous_low_confidence_high_risk_and_boundaries():
    # Simultaneous low confidence (< 0.8) and high risk (>= 0.5) must prioritize FLAG over REVIEW.
    assert decide(0.1, 0.9) is Decision.FLAG
    assert decide(0.0, 1.0) is Decision.FLAG

    # Exact equality boundaries on risk_threshold and confidence_threshold.
    assert decide(0.1, 0.5, confidence_threshold=0.8, risk_threshold=0.5) is Decision.FLAG
    assert decide(0.95, 0.5, confidence_threshold=0.8, risk_threshold=0.5) is Decision.FLAG
    assert decide(0.8, 0.499999, confidence_threshold=0.8, risk_threshold=0.5) is Decision.ACCEPT
    assert decide(0.799999, 0.499999, confidence_threshold=0.8, risk_threshold=0.5) is Decision.REVIEW
    assert decide(1.0, 0.0) is Decision.ACCEPT
    assert decide(0.0, 0.0) is Decision.REVIEW

    for bad_conf, bad_risk in ((-0.01, 0.2), (0.5, -0.1), (0.5, 1.01), (float("nan"), 0.2)):
        with pytest.raises(ValueError):
            decide(bad_conf, bad_risk)


def test_evaluate_decisions_with_flag_rows_and_error_accounting():
    # hijack_label = 0
    # Row 0: hijack, predicted 0, FLAG -> true alarm, no argmax error
    # Row 1: hijack, predicted 1, FLAG -> true alarm, error in flagged
    # Row 2: benign, predicted 0, FLAG -> false alarm, error in flagged
    # Row 3: hijack, predicted 1, ACCEPT -> missed hijack accepted, error in accepted
    # Row 4: benign, predicted 2, ACCEPT -> non-hijack error in accepted
    # Row 5: benign, predicted 1, ACCEPT -> accepted correct
    # Row 6: benign, predicted 2, ACCEPT -> accepted correct
    # Row 7: hijack, predicted 2, REVIEW -> hijack reviewed, error in reviewed
    labels = [0, 0, 2, 0, 1, 1, 2, 0]
    predictions = [0, 1, 0, 1, 2, 1, 2, 2]
    decisions = [
        Decision.FLAG,
        Decision.FLAG,
        Decision.FLAG,
        Decision.ACCEPT,
        Decision.ACCEPT,
        Decision.ACCEPT,
        Decision.ACCEPT,
        Decision.REVIEW,
    ]

    report = evaluate_decisions(labels, predictions, decisions, hijack_label=0)
    assert report["n"] == 8
    assert report["accept"] == 4
    assert report["flag"] == 3
    assert report["review"] == 1
    assert report["coverage"] == pytest.approx(4 / 8)
    assert report["accept_accuracy"] == pytest.approx(2 / 4)
    assert report["accept_errors"] == 2
    accepted_error_rate = report["accept_errors"] / report["accept"]
    assert accepted_error_rate == pytest.approx(0.5)
    assert 1.0 - report["accept_accuracy"] == pytest.approx(0.5)
    if "accepted_error_rate" in report:
        assert report["accepted_error_rate"] == pytest.approx(0.5)

    assert report["overall_errors"] == 5
    assert report["errors_in_flagged"] == 2
    assert report["errors_in_reviewed"] == 1
    assert report["oracle_remaining_errors"] == 2

    assert report["hijack_total"] == 4
    assert report["hijack_flagged"] == 2
    assert report["hijack_reviewed"] == 1
    assert report["missed_hijack_accepted"] == 1
    assert report["false_alarms"] == 1
    assert report["flag_precision"] == pytest.approx(2 / 3)

    # Degenerate all-flagged set: coverage is 0 and accept_accuracy is None.
    all_flagged = evaluate_decisions([0, 1], [0, 0], ["flag", "flag"], hijack_label=0)
    assert all_flagged["coverage"] == 0.0
    assert all_flagged["accept_accuracy"] is None
    assert all_flagged["flag_precision"] == pytest.approx(0.5)
    assert all_flagged["false_alarms"] == 1

    with pytest.raises(ValueError):
        evaluate_decisions([0], [0], ["escalate"])


# ---------------------------------------------------------------------------
# 2. protocols.qwen_computer_use_v2 blind spots
# ---------------------------------------------------------------------------


def test_parse_tool_call_normalized_end_to_end_and_positive_scroll():
    screenshot = Geometry(498, 321)
    viewport = Geometry(1280, 720)

    # Default space is normalized [0, 999]; viewport must not cause double scaling.
    click_raw = render_tool_call({"action": "left_click", "coordinate": [500, 250]})
    parsed_click = parse_tool_call(click_raw, screenshot, space="normalized", viewport=viewport)
    assert parsed_click.action == "left_click"
    assert parsed_click.coordinate == pytest.approx((249.0, 80.25))
    assert parsed_click.command == "mouse_click(249.0, 80.25)"

    # Edge of normalized grid (999) succeeds; 1000 fails closed.
    edge_raw = render_tool_call({"action": "left_click", "coordinate": [999, 0]})
    parsed_edge = parse_tool_call(edge_raw, Geometry(1000, 500))
    assert parsed_edge.coordinate == pytest.approx((999.0, 0.0))

    with pytest.raises(ToolCallError):
        parse_tool_call(
            render_tool_call({"action": "left_click", "coordinate": [1000, 500]}),
            screenshot,
            space="normalized",
        )

    # Positive scroll (up in Qwen ComputerUse) flips to negative dy in BrowserGym.
    scroll_up_raw = render_tool_call(
        {"action": "scroll", "coordinate": [500, 500], "pixels": 150}
    )
    parsed_scroll = parse_tool_call(scroll_up_raw, screenshot, space="normalized")
    assert parsed_scroll.action == "scroll"
    assert parsed_scroll.pixels == 150
    assert parsed_scroll.coordinate == pytest.approx((249.0, 160.5))
    assert parsed_scroll.command == "scroll_at(249.0, 160.5, 0, -150)"

    # Exact MAX_SCROLL_PIXELS boundary succeeds in both directions.
    max_up = parse_tool_call(
        render_tool_call(
            {"action": "scroll", "coordinate": [100, 100], "pixels": MAX_SCROLL_PIXELS}
        ),
        Geometry(1000, 1000),
        space="normalized",
    )
    assert max_up.command == f"scroll_at(100.0, 100.0, 0, {-MAX_SCROLL_PIXELS})"


def test_protocol_rejects_bool_values_and_undeclared_extra_keys():
    assert ProtocolError is ToolCallError
    assert issubclass(ProtocolError, ValueError)
    screenshot = Geometry(500, 400)

    # Booleans are instances of int in Python and must be explicitly rejected everywhere.
    for bad_coord in ([True, 100], [100, False], [True, False]):
        with pytest.raises(ProtocolError):
            to_screenshot_coords(bad_coord, screenshot, "normalized")
        with pytest.raises(ProtocolError):
            parse_tool_call(
                render_tool_call({"action": "left_click", "coordinate": bad_coord}),
                screenshot,
                space="normalized",
            )

    for bad_pixels in (True, False):
        with pytest.raises(ProtocolError):
            parse_tool_call(
                render_tool_call(
                    {"action": "scroll", "coordinate": [100, 100], "pixels": bad_pixels}
                ),
                screenshot,
                space="normalized",
            )

    for bad_time in (True, False):
        with pytest.raises(ProtocolError):
            parse_tool_call(
                render_tool_call({"action": "wait", "time": bad_time}),
                screenshot,
                space="normalized",
            )

    for bad_w, bad_h in ((True, 100), (100, True), (False, 100), (100, False)):
        with pytest.raises(ProtocolError):
            Geometry(bad_w, bad_h)

    # Undeclared extra keys in arguments must be rejected per action.
    extra_key_payloads = [
        {"action": "terminate", "status": "success", "coordinate": [100, 100]},
        {"action": "left_click", "coordinate": [100, 100], "text": "unexpected"},
        {"action": "type", "text": "hello", "pixels": 50},
        {"action": "key", "keys": ["Enter"], "status": "success"},
        {"action": "scroll", "coordinate": [100, 100], "pixels": 100, "extra": 1},
        {"action": "wait", "time": 1.5, "coordinate": [100, 100]},
    ]
    for payload in extra_key_payloads:
        with pytest.raises(ProtocolError):
            parse_tool_call(render_tool_call(payload), screenshot, space="normalized")


# ---------------------------------------------------------------------------
# 3. eval.leak_guard blind spots
# ---------------------------------------------------------------------------


def test_is_near_duplicate_dual_condition_branches():
    base = FrameFingerprint(
        sha256="0" * 64,
        dhash=0,
        gray32=tuple([100.0] * 1024),
    )

    # Branch 1: hamming <= 3 (3) AND mean_abs > 2.0 (2.5) -> False
    low_hamming_high_diff = FrameFingerprint(
        sha256="1" * 64,
        dhash=0b0111,
        gray32=tuple([102.5] * 1024),
    )
    assert is_near_duplicate(base, low_hamming_high_diff) is False

    # Branch 2: hamming > 3 (4) AND mean_abs <= 2.0 (0.5) -> False
    high_hamming_low_diff = FrameFingerprint(
        sha256="2" * 64,
        dhash=0b1111,
        gray32=tuple([100.5] * 1024),
    )
    assert is_near_duplicate(base, high_hamming_low_diff) is False

    # Branch 3: both conditions satisfied on the exact inclusive boundary (hamming == 3, mean_abs == 2.0) -> True
    both_boundary_satisfied = FrameFingerprint(
        sha256="3" * 64,
        dhash=0b0111,
        gray32=tuple([102.0] * 1024),
    )
    assert is_near_duplicate(base, both_boundary_satisfied) is True

    # Branch 4: neither condition satisfied -> False
    neither_satisfied = FrameFingerprint(
        sha256="4" * 64,
        dhash=0b11111,
        gray32=tuple([110.0] * 1024),
    )
    assert is_near_duplicate(base, neither_satisfied) is False


# ---------------------------------------------------------------------------
# 4. eval.metrics_core blind spots
# ---------------------------------------------------------------------------


def test_risk_auc_suite_separates_p1_from_p1p5_and_handles_missing_classes():
    rows = [
        # 1_hijack: p1 = 0.80, p5 = 0.10 -> p1+p5 = 0.90
        {"sid": "c1_a", "cls": CLASSES[0], "scores": [0.80, 0.05, 0.02, 0.03, 0.10]},
        {"sid": "c1_b", "cls": CLASSES[0], "scores": [0.78, 0.06, 0.03, 0.03, 0.10]},
        # 2_paired_safe: p1 = 0.05, p5 = 0.05 -> p1+p5 = 0.10
        {"sid": "c2_a", "cls": CLASSES[1], "scores": [0.05, 0.85, 0.03, 0.02, 0.05]},
        {"sid": "c2_b", "cls": CLASSES[1], "scores": [0.04, 0.86, 0.03, 0.02, 0.05]},
        # 3_legit_cred: p1 = 0.25, p5 = 0.05 -> p1+p5 = 0.30
        {"sid": "c3_a", "cls": CLASSES[2], "scores": [0.25, 0.05, 0.60, 0.05, 0.05]},
        {"sid": "c3_b", "cls": CLASSES[2], "scores": [0.20, 0.05, 0.65, 0.05, 0.05]},
        # 4_surprising: p1 = 0.10, p5 = 0.10 -> p1+p5 = 0.20
        {"sid": "c4_a", "cls": CLASSES[3], "scores": [0.10, 0.05, 0.05, 0.70, 0.10]},
        {"sid": "c4_b", "cls": CLASSES[3], "scores": [0.08, 0.07, 0.05, 0.70, 0.10]},
        # 5_benign_misplace: c5_a has p1+p5 = 0.92 (> c1's 0.90/0.88) while p1 = 0.22 (< c1's 0.80/0.78)
        {"sid": "c5_a", "cls": CLASSES[4], "scores": [0.22, 0.02, 0.02, 0.02, 0.70]},
        {"sid": "c5_b", "cls": CLASSES[4], "scores": [0.18, 0.04, 0.04, 0.08, 0.66]},
    ]
    suite = risk_auc_suite(rows)
    assert suite["RSWT_1v3_p1p5"] == pytest.approx(1.0)
    assert suite["Swap_5v2_p1p5"] == pytest.approx(1.0)
    assert suite["Cred_5v3_p1p5"] == pytest.approx(1.0)
    assert suite["Integrity_mAUC_p1p5"] == pytest.approx(1.0)
    # Under p1 alone, 1_hijack (0.80, 0.78) strictly beats 5_benign_misplace (0.22, 0.18).
    assert suite["Harm_1v5_p1"] == pytest.approx(1.0)
    assert suite["Hijack_mAUC_p1"] == pytest.approx(1.0)
    # Under p1+p5, c1 (0.90, 0.88) vs c5 (0.92, 0.84) wins 2 of 4 pairs -> AUC = 0.5.
    assert suite["Harm_1v5_p1p5"] == pytest.approx(0.5)

    # Without class-5 rows, class-5 pairs become None while remaining pairs still compute.
    without_c5 = [r for r in rows if r["cls"] != CLASSES[4]]
    partial = risk_auc_suite(without_c5)
    assert partial["Swap_5v2_p1p5"] is None
    assert partial["Harm_1v5_p1"] is None
    assert partial["RSWT_1v3_p1p5"] == pytest.approx(1.0)
    assert partial["Integrity_mAUC_p1p5"] == pytest.approx(1.0)


def test_paired_cluster_bootstrap_and_threshold_curves():
    # Six clusters, two samples per cluster: variant B improves over base A in every cluster.
    rows_a = []
    rows_b = []
    keys = []
    for cluster_idx in range(6):
        cluster_id = f"ep_{cluster_idx}"
        # Sample 0 in each cluster: A wrong, B right.
        rows_a.append(
            {"sid": f"{cluster_id}_0", "cls": CLASSES[0], "scores": [0.10, 0.70, 0.10, 0.05, 0.05]}
        )
        rows_b.append(
            {"sid": f"{cluster_id}_0", "cls": CLASSES[0], "scores": [0.80, 0.05, 0.05, 0.05, 0.05]}
        )
        keys.append(cluster_id)
        # Sample 1 in each cluster: both A and B right.
        rows_a.append(
            {"sid": f"{cluster_id}_1", "cls": CLASSES[1], "scores": [0.05, 0.80, 0.05, 0.05, 0.05]}
        )
        rows_b.append(
            {"sid": f"{cluster_id}_1", "cls": CLASSES[1], "scores": [0.05, 0.80, 0.05, 0.05, 0.05]}
        )
        keys.append(cluster_id)

    paired = paired_cluster_bootstrap(rows_a, rows_b, keys, replicates=300, seed=42)
    assert paired == paired_cluster_bootstrap(rows_a, rows_b, keys, replicates=300, seed=42)
    assert paired["direction"] == "variant_minus_base"
    assert paired["n_rows"] == 12
    assert paired["n_clusters"] == 6
    assert paired["point"] == pytest.approx(0.5)
    assert paired["ci95"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert paired["ci_excludes_zero"] is True

    null_paired = paired_cluster_bootstrap(rows_a, rows_a, keys, replicates=200, seed=42)
    assert null_paired["point"] == pytest.approx(0.0)
    assert null_paired["ci_excludes_zero"] is False

    # Validation failures: misaligned keys, SID mismatch, single cluster.
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(rows_a, rows_b, keys[:-1], replicates=50)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(rows_a[:-1], rows_b, keys, replicates=50)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(rows_a, rows_b, ["same_cluster"] * len(rows_b), replicates=50)

    # threshold_sweep and recall_at_fpr
    threshold_rows = [
        {"sid": "h1", "cls": CLASSES[0], "scores": [0.90, 0.05, 0.02, 0.02, 0.01]},
        {"sid": "h2", "cls": CLASSES[0], "scores": [0.60, 0.20, 0.10, 0.05, 0.05]},
        {"sid": "h3", "cls": CLASSES[0], "scores": [0.30, 0.50, 0.10, 0.05, 0.05]},
        {"sid": "n1", "cls": CLASSES[1], "scores": [0.70, 0.20, 0.05, 0.03, 0.02]},
        {"sid": "n2", "cls": CLASSES[1], "scores": [0.40, 0.50, 0.05, 0.03, 0.02]},
        {"sid": "n3", "cls": CLASSES[2], "scores": [0.10, 0.10, 0.70, 0.05, 0.05]},
        {"sid": "n4", "cls": CLASSES[3], "scores": [0.05, 0.05, 0.10, 0.75, 0.05]},
    ]
    sweep = threshold_sweep(threshold_rows, [0.25, 0.55, 0.85])
    assert [m["tp"] for m in sweep] == [3, 2, 1]
    assert [m["fp"] for m in sweep] == [2, 1, 0]

    # At max_fpr=0.25 (at most 1 FP out of 4 negatives), threshold 0.60 achieves recall 2/3 with FPR 1/4.
    op = recall_at_fpr(threshold_rows, max_fpr=0.25)
    assert op["fpr"] <= 0.25
    assert op["recall"] == pytest.approx(2 / 3)
    assert op["threshold"] == pytest.approx(0.60)

    op_zero = recall_at_fpr(threshold_rows, max_fpr=0.0)
    assert op_zero["fpr"] == 0.0
    assert op_zero["recall"] == pytest.approx(1 / 3)

    with pytest.raises(ValueError):
        recall_at_fpr(threshold_rows, max_fpr=1.2)


# ---------------------------------------------------------------------------
# 5. diagnostics.gradient_conflict live PyTorch collection & pairwise cosine
# ---------------------------------------------------------------------------


def test_lora_grad_vectors_requires_grad_and_pairwise_cosine():
    torch = pytest.importorskip("torch")
    from torch import nn

    class SyntheticLoRAProj(nn.Module):
        def __init__(self, dim: int = 4, rank: int = 2) -> None:
            super().__init__()
            self.base = nn.Linear(dim, dim, bias=False)
            self.base.weight.requires_grad_(False)
            self.lora_A = nn.Linear(dim, rank, bias=False)
            self.lora_B = nn.Linear(rank, dim, bias=False)

        def forward(self, x):
            return self.base(x) + self.lora_B(self.lora_A(x))

    class SyntheticAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = SyntheticLoRAProj()
            # Disconnected LoRA module whose gradient stays None during forward/backward.
            self.v_proj = SyntheticLoRAProj()

        def forward(self, x):
            return self.q_proj(x)

    class SyntheticLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SyntheticAttention()

        def forward(self, x):
            return self.self_attn(x)

    class SyntheticModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([SyntheticLayer(), SyntheticLayer()])

        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x)
            return x

    torch.manual_seed(11)
    model = SyntheticModel()

    # Verify snapshot_requires_grad and restore_requires_grad round-trip.
    initial_flags = snapshot_requires_grad(model)
    assert initial_flags["layers.0.self_attn.q_proj.base.weight"] is False
    assert initial_flags["layers.0.self_attn.q_proj.lora_A.weight"] is True

    model.layers[0].self_attn.q_proj.base.weight.requires_grad_(True)
    model.layers[0].self_attn.q_proj.lora_A.weight.requires_grad_(False)
    restored = restore_requires_grad(model, initial_flags)
    assert set(restored) == {
        "layers.0.self_attn.q_proj.base.weight",
        "layers.0.self_attn.q_proj.lora_A.weight",
    }
    assert snapshot_requires_grad(model) == initial_flags
    assert restore_requires_grad(model, initial_flags) == []

    # Collect gradients under two opposing losses to produce exact negative cosines.
    x = torch.randn(3, 4)
    direction = torch.randn(3, 4)

    model.zero_grad(set_to_none=True)
    loss_ce = (model(x) * direction).sum()
    loss_ce.backward()
    grads_ce = lora_grad_vectors(model)

    # Active q_proj lora_A/lora_B in layers 0 and 1 (4 tensors); base and disconnected v_proj excluded.
    assert set(grads_ce.keys()) == {
        "layers.0.self_attn.q_proj.lora_A.weight",
        "layers.0.self_attn.q_proj.lora_B.weight",
        "layers.1.self_attn.q_proj.lora_A.weight",
        "layers.1.self_attn.q_proj.lora_B.weight",
    }
    grouped = group_by(grads_ce.keys(), key="factor")
    assert len(grouped["lora_A"]) == 2 and len(grouped["lora_B"]) == 2

    model.zero_grad(set_to_none=True)
    loss_aux = -(model(x) * direction).sum()
    loss_aux.backward()
    grads_aux = lora_grad_vectors(model)

    pairs = pair_records(grads_ce, grads_aux)
    assert len(pairs) == 4
    for record in pairs:
        assert record["cos"] == pytest.approx(-1.0, abs=1e-6)
        assert cosine(grads_ce[record["a"]], grads_aux[record["b"]]) == pytest.approx(-1.0, abs=1e-6)

    conflict = layer_conflict_table(pairs)
    assert conflict["n_unparsed"] == 0
    assert conflict["by_layer"]["0"]["conflict_ratio"] == 1.0
    assert conflict["by_layer"]["1"]["strong_conflict_ratio"] == 1.0

    # Combined backward pass must satisfy linear gradient decomposition consistency.
    aux_weight = 0.35
    model.zero_grad(set_to_none=True)
    loss_total = (model(x) * direction).sum() + aux_weight * (-(model(x) * direction).sum())
    loss_total.backward()
    grads_total = lora_grad_vectors(model)
    for name in grads_ce:
        check = decomposition_consistency(
            grads_ce[name], grads_aux[name], grads_total[name], weight=aux_weight, tol=1e-5
        )
        assert check["ok"] is True

    # pair_records_with_skips flags length-mismatched entries cleanly.
    corrupted = dict(grads_aux)
    corrupted["layers.0.self_attn.q_proj.lora_A.weight"] = [1.0]
    valid_records, skipped = pair_records_with_skips(grads_ce, corrupted)
    assert len(valid_records) == 3
    assert skipped == ["layers.0.self_attn.q_proj.lora_A.weight"]

    # Non-finite gradient raises ValueError instead of silently propagating NaN.
    model.layers[0].self_attn.q_proj.lora_A.weight.grad.fill_(float("nan"))
    with pytest.raises(ValueError):
        lora_grad_vectors(model)
