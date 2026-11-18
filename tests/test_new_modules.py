"""Unit tests for newly added training, adapter fusion, cascade, PRM, and calibration modules.

Covers:
1. ``training.losses``: ``class_weights``, ``full_vocab_loss`` sample-weight preservation at
   ``batch_size=1`` (preventing PyTorch ``F.cross_entropy`` weight cancellation), and
   ``pos_only`` unilateral margin loss vs bilateral ``both``.
2. ``training.adapter_soup``: ``merge_lora_adapters`` (``r=16+16->32``, ``alpha=32+32->64``)
   single-pass forward equivalence vs weighted sum of two LoRA passes (``< 1e-6`` error)
   and contrast against naive factor-averaging cross-term error.
3. ``fastpe.cascade``: ``ALERT_SAFE_CONFIG`` vs ``FULL_PARITY_CONFIG`` in ``should_fallback``,
   ``CascadeRunner`` two-stage ``"gemm" -> "original"`` switching reusing the exact same
   ``batch`` object (``is`` identity), thread-lock protection, and ``try...finally``
   restoration to ``"gemm"`` when Stage 2 raises an exception.
4. ``prm.scoring`` & ``prm.controller``: single-token ``ABCD`` / ``AB`` last-position logits
   extraction across layouts, 4-class PRM & dual binary ``change``/``goal`` Action Judge V3
   scoring/loss, pure-stdlib ``Feedback`` (excluding DOM Oracle fields), and 7-state
   idempotent fail-closed controller (non-idempotent stale-failure guard, retry/observe/recover
   budgets, cycle detection, and pinned ``original_bf16`` safety escalation).
5. ``diagnostics.selective_abstention``: 5-class logits audit proving multiclass Temperature
   Scaling preserves per-sample ``argmax`` while altering cross-sample ``p1`` ROC-AUC
   (``ts_roc_auc != raw_roc_auc``), whereas positive-slope (``a > 0``) ``binary_platt`` /
   ``apply_binary_platt`` strictly preserves cross-sample ``p1`` ROC-AUC, plus
   ``brier_multiclass`` and ``brier_binary``.
"""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from diagnostics.selective_abstention import (
    Decision as AbstentionDecision,
    apply_binary_platt,
    audit_temperature_vs_platt_auc,
    binary_platt,
    brier_binary,
    brier_multiclass,
    roc_auc,
)
from prm.controller import (
    ACTIONS,
    ControllerAction,
    ControllerContext,
    Feedback,
    ModelSignal,
    StateEvidence,
    conservative_reward,
    decide_action,
    evaluate_progress_feedback,
    has_state_action_loop,
    is_non_idempotent_action,
    require_original_bf16_safety_review,
)


# ---------------------------------------------------------------------------
# 1. training.losses
# ---------------------------------------------------------------------------


def test_class_weights_effective_number_normalization_and_cap():
    torch = pytest.importorskip("torch")
    from training.losses import class_weights

    # Extreme class imbalance: class 0 occurs once, classes 1..4 occur 500 times each.
    labels = [0] + [1] * 500 + [2] * 500 + [3] * 500 + [4] * 500
    weights = class_weights(labels, beta=0.999, cap=3.0)
    assert weights.shape == (5,)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    # Class 0 must carry the largest weight, bounded by cap=3.0 prior to final renormalization.
    assert float(weights[0]) > float(weights[1])
    for idx in (2, 3, 4):
        assert float(weights[idx]) == pytest.approx(float(weights[1]), abs=1e-6)

    # beta == 0 reduces to uniform weights.
    uniform = class_weights([0, 1, 2, 3, 4], beta=0.0, cap=3.0)
    assert torch.allclose(uniform, torch.ones(5))

    # Missing class, out-of-range beta/cap, or boolean labels must fail closed.
    with pytest.raises(ValueError):
        class_weights([0, 1, 2, 3])
    with pytest.raises(ValueError):
        class_weights([0, 1, 2, 3, 4], beta=1.0)
    with pytest.raises(ValueError):
        class_weights([0, 1, 2, 3, 4], cap=0.5)
    with pytest.raises(ValueError):
        class_weights([True, False, True, False, True])


def test_full_vocab_loss_preserves_sample_weight_at_batch_size_one():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from training.losses import full_vocab_loss

    torch.manual_seed(7)
    vocab_size = 64
    token_ids = [10, 11, 12, 13, 14]
    raw_logits = torch.randn(1, vocab_size, dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)  # single sample of class 0 (1_hijack)

    unit_weights = torch.ones(5, dtype=torch.float32)
    boosted_weights = torch.tensor([2.75, 0.8, 0.8, 1.2, 0.9], dtype=torch.float32)

    # Built-in PyTorch F.cross_entropy(..., weight=...) self-normalizes by sum(w[y_i]),
    # so at batch_size=1 the scalar weight cancels out completely.
    cand_logits = raw_logits[:, token_ids]
    builtin_unit = F.cross_entropy(cand_logits, labels, weight=unit_weights)
    builtin_boosted = F.cross_entropy(cand_logits, labels, weight=boosted_weights)
    assert float(builtin_boosted) == pytest.approx(float(builtin_unit), rel=1e-6)

    # full_vocab_loss multiplies per-sample CE prior to .mean(), preserving exact weight scaling
    # in both loss value and parameter gradients even at batch_size=1.
    logits_u = raw_logits.clone().requires_grad_(True)
    loss_u, parts_u = full_vocab_loss(logits_u, labels, token_ids, weight=unit_weights)
    loss_u.backward()

    logits_w = raw_logits.clone().requires_grad_(True)
    loss_w, parts_w = full_vocab_loss(logits_w, labels, token_ids, weight=boosted_weights)
    loss_w.backward()

    assert float(loss_w.detach()) == pytest.approx(2.75 * float(loss_u.detach()), rel=1e-6)
    assert parts_w["ce"] == pytest.approx(2.75 * parts_u["ce"], rel=1e-6)
    assert torch.allclose(logits_w.grad, 2.75 * logits_u.grad, rtol=1e-6, atol=1e-6)

    # Auxiliary margin + KD terms combine cleanly on top of weighted full-vocab CE.
    batch_logits = torch.randn(3, vocab_size, requires_grad=True)
    batch_labels = torch.tensor([0, 3, 1], dtype=torch.long)
    teacher = torch.randn(3, 5)
    total_aux, parts_aux = full_vocab_loss(
        batch_logits,
        batch_labels,
        token_ids,
        weight=boosted_weights,
        margin=0.5,
        margin_weight=0.2,
        teacher_logits=teacher,
        kd_weight=0.1,
        margin_mode="pos_only",
    )
    assert parts_aux["margin"] > 0.0
    assert parts_aux["kd"] >= 0.0
    assert parts_aux["kd_n"] == 2  # class 3 is excluded from protected-class KD
    total_aux.backward()
    assert torch.isfinite(batch_logits.grad).all()


def test_pos_only_unilateral_margin_has_zero_reverse_force_on_negatives():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from training.losses import m1_loss, pos_only_margin_loss

    # Batch containing ONLY negative classes (1: 2_paired_safe, 4: 5_benign_misplace) and NO class 3.
    neg_only_base = torch.tensor(
        [
            [0.1, 1.0, 0.0, 2.5, -0.5],  # label 1, high class-3 logit 2.5
            [0.0, -0.2, 0.1, 2.0, 0.8],  # label 4, high class-3 logit 2.0
        ],
        dtype=torch.float32,
    )
    neg_labels = torch.tensor([1, 4], dtype=torch.long)

    # Unilateral pos_only margin must emit exact 0 loss and 0 margin gradient on negative-only batches.
    logits_pos_only = neg_only_base.clone().requires_grad_(True)
    m_pos = pos_only_margin_loss(logits_pos_only, neg_labels, margin=0.5)
    assert float(m_pos.detach()) == 0.0
    m_pos.backward()
    assert torch.count_nonzero(logits_pos_only.grad) == 0

    _, parts_pos = m1_loss(
        neg_only_base, neg_labels, margin=0.5, margin_weight=1.0, margin_mode="pos_only"
    )
    _, parts_both = m1_loss(
        neg_only_base, neg_labels, margin=0.5, margin_weight=1.0, margin_mode="both"
    )
    assert parts_pos["margin"] == 0.0
    assert parts_pos["margin_mode_pos_only"] == 1.0
    assert parts_both["margin"] > 0.0  # bilateral mode penalizes z_3 on negative classes

    # Mixed batch with one class-3 sample and one class-1 sample:
    # pos_only margin gradient touches ONLY the class-3 row (row 0), leaving row 1 untouched.
    mixed_logits = torch.tensor(
        [
            [0.0, 1.5, -1.0, 0.5, 1.2],  # row 0: positive class 3 (hardest neg is z_1 = 1.5)
            [0.0, 0.8, -0.5, 2.2, 0.1],  # row 1: negative class 1
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    mixed_labels = torch.tensor([3, 1], dtype=torch.long)
    loss_margin = pos_only_margin_loss(mixed_logits, mixed_labels, margin=0.25)
    expected = F.softplus(torch.tensor(0.25 + 1.5 - 0.5))
    assert float(loss_margin.detach()) == pytest.approx(float(expected), rel=1e-6)
    loss_margin.backward()
    assert float(mixed_logits.grad[0, 3]) < 0.0  # pushes z_3 up on positive sample
    assert float(mixed_logits.grad[0, 1]) > 0.0  # pushes hardest negative z_1 down on positive sample
    assert torch.all(mixed_logits.grad[1] == 0.0)  # zero reverse margin force on negative sample


# ---------------------------------------------------------------------------
# 2. training.adapter_soup
# ---------------------------------------------------------------------------


def test_adapter_soup_exact_delta_concatenation_vs_naive_factor_average():
    torch = pytest.importorskip("torch")
    from training.adapter_soup import (
        apply_lora_delta,
        merge_lora_adapters,
        merged_lora_config,
        verify_adapter_soup_equivalence,
    )

    cfg = merged_lora_config(rank=16, alpha=32.0, num_adapters=2)
    assert cfg == {"r": 32, "lora_alpha": 64, "alpha_over_r": 2.0}

    torch.manual_seed(2026)
    in_features, out_features, rank = 24, 18, 16
    alpha_over_r = 32.0 / 16.0  # == 64.0 / 32.0 == 2.0

    state_a = {
        "layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, in_features) * 0.2,
        "layers.0.self_attn.q_proj.lora_B.weight": torch.randn(out_features, rank) * 0.2,
        "layers.1.self_attn.v_proj.lora_A.weight": torch.randn(rank, in_features) * 0.2,
        "layers.1.self_attn.v_proj.lora_B.weight": torch.randn(out_features, rank) * 0.2,
    }
    state_b = {
        "layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, in_features) * 0.2,
        "layers.0.self_attn.q_proj.lora_B.weight": torch.randn(out_features, rank) * 0.2,
        "layers.1.self_attn.v_proj.lora_A.weight": torch.randn(rank, in_features) * 0.2,
        "layers.1.self_attn.v_proj.lora_B.weight": torch.randn(out_features, rank) * 0.2,
    }

    for weights in ((0.5, 0.5), (0.35, 0.65)):
        merged = merge_lora_adapters(state_a, state_b, weights=weights)
        assert merged["layers.0.self_attn.q_proj.lora_A.weight"].shape == (32, in_features)
        assert merged["layers.0.self_attn.q_proj.lora_B.weight"].shape == (out_features, 32)

        raw_x = torch.randn(5, in_features)
        x = raw_x / torch.linalg.vector_norm(raw_x, dim=-1, keepdim=True)
        key_a = "layers.0.self_attn.q_proj.lora_A.weight"
        key_b = "layers.0.self_attn.q_proj.lora_B.weight"

        # Merged single forward pass with alpha_merged / r_merged == 64 / 32 == 2.0
        merged_delta = apply_lora_delta(
            x, merged[key_a], merged[key_b], alpha_over_r=cfg["lora_alpha"] / cfg["r"]
        )
        # Weighted sum of two independent LoRA forward passes with alpha / r == 32 / 16 == 2.0
        indep_delta = (
            weights[0] * apply_lora_delta(x, state_a[key_a], state_a[key_b], alpha_over_r=alpha_over_r)
            + weights[1] * apply_lora_delta(x, state_b[key_a], state_b[key_b], alpha_over_r=alpha_over_r)
        )
        max_err = float((merged_delta - indep_delta).abs().max())
        assert max_err < 1e-6

        # Naive factor averaging introduces cross-term error B1@A2 + B2@A1.
        a_naive = weights[0] * state_a[key_a] + weights[1] * state_b[key_a]
        b_naive = weights[0] * state_a[key_b] + weights[1] * state_b[key_b]
        naive_delta = apply_lora_delta(x, a_naive, b_naive, alpha_over_r=alpha_over_r)
        naive_err = float((naive_delta - indep_delta).abs().max())
        assert naive_err > 1e-2

        audit = verify_adapter_soup_equivalence(
            state_a, state_b, merged_state=merged, weights=weights, alpha_over_r=2.0, num_probes=12, seed=9
        )
        assert audit["num_modules"] == 2
        assert audit["max_abs_diff"] < 1e-6
        assert audit["max_rel_l2"] < 1e-6
        assert audit["naive_avg_max_abs_diff"] > 1e-2

    # Validation errors on mismatched keys or missing lora_B partner.
    with pytest.raises(ValueError):
        merge_lora_adapters({"layers.0.q_proj.lora_A.weight": torch.randn(4, 8)}, state_b)
    with pytest.raises(ValueError):
        merge_lora_adapters(
            {"layers.0.q_proj.lora_A.weight": torch.randn(4, 8)},
            {"layers.0.q_proj.lora_A.weight": torch.randn(4, 8)},
        )


# ---------------------------------------------------------------------------
# 3. fastpe.cascade
# ---------------------------------------------------------------------------


def _build_fastpe_tower(torch, nn):
    from fastpe.fast_patch_embed import apply_fastpe

    class PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = 3
            self.temporal_patch_size = 2
            self.patch_size = 2
            self.embed_dim = 6
            self.proj = nn.Conv3d(3, 6, (2, 2, 2), stride=(2, 2, 2))

        def forward(self, hidden_states):
            return self.proj(hidden_states.view(-1, 3, 2, 2, 2)).view(-1, 6)

    class VisionTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = PatchEmbed()

    tower = VisionTower()
    assert apply_fastpe(tower, mode="gemm") == 1
    return tower


def test_should_fallback_alert_safe_vs_full_parity_operating_points():
    pytest.importorskip("torch")
    from fastpe.cascade import ALERT_SAFE_CONFIG, FULL_PARITY_CONFIG, CascadeConfig, should_fallback

    # Inside p1 boundary [0.30, 0.75]: both operating points trigger "p1_boundary".
    for boundary_probs in (
        [0.30, 0.60, 0.04, 0.03, 0.03],
        [0.52, 0.30, 0.08, 0.05, 0.05],
        [0.75, 0.15, 0.04, 0.03, 0.03],
    ):
        assert should_fallback(boundary_probs, ALERT_SAFE_CONFIG) == (True, "p1_boundary")
        assert should_fallback(boundary_probs, FULL_PARITY_CONFIG) == (True, "p1_boundary")

    # Low p1 (0.05 < 0.30) AND low confidence (max=0.45 < 0.70):
    # ALERT_SAFE_CONFIG stays on fast gemm, whereas FULL_PARITY_CONFIG triggers "low_confidence".
    low_conf_safe_p1 = [0.05, 0.45, 0.25, 0.15, 0.10]
    assert should_fallback(low_conf_safe_p1, ALERT_SAFE_CONFIG) == (False, "none")
    assert should_fallback(low_conf_safe_p1, FULL_PARITY_CONFIG) == (True, "low_confidence")

    # Confident safe (p1=0.02, conf=0.90) and confident hijack (p1=0.90 > 0.75): neither falls back.
    assert should_fallback([0.02, 0.90, 0.03, 0.03, 0.02], ALERT_SAFE_CONFIG) == (False, "none")
    assert should_fallback([0.02, 0.90, 0.03, 0.03, 0.02], FULL_PARITY_CONFIG) == (False, "none")
    assert should_fallback([0.90, 0.04, 0.02, 0.02, 0.02], FULL_PARITY_CONFIG) == (False, "none")

    # upper_bound_open=True treats p1 == p1_high as outside the boundary.
    open_cfg = CascadeConfig(gate_mode="alert_safe", upper_bound_open=True)
    assert should_fallback([0.75, 0.15, 0.04, 0.03, 0.03], open_cfg) == (False, "none")


def test_cascade_runner_batch_identity_exception_recovery_and_thread_lock():
    torch = pytest.importorskip("torch")
    from torch import nn
    from fastpe.cascade import ALERT_SAFE_CONFIG, CascadeRunner

    tower = _build_fastpe_tower(torch, nn)
    observed_calls: list[tuple[str, object]] = []

    def score_batch(batch):
        observed_calls.append((tower.patch_embed.mode, batch))
        if tower.patch_embed.mode == "gemm":
            # Stage 1 lands in p1 boundary [0.30, 0.75], triggering Stage 2.
            return [0.48, 0.40, 0.04, 0.04, 0.04]
        # Stage 2 original BF16 resolves to high-risk hijack above risk_threshold (~0.6703).
        return [0.82, 0.10, 0.03, 0.03, 0.02]

    shared_lock = threading.Lock()
    runner = CascadeRunner(tower, score_batch, lock=shared_lock, cfg=ALERT_SAFE_CONFIG)
    assert runner.lock is shared_lock

    resident_batch = {"pixel_values": object(), "input_ids": [1, 2, 3]}
    out = runner(resident_batch)

    assert len(observed_calls) == 2
    assert observed_calls[0][0] == "gemm"
    assert observed_calls[1][0] == "original"
    # Stage 1 and Stage 2 MUST reuse the exact same batch object reference (`is` identity).
    assert observed_calls[0][1] is resident_batch
    assert observed_calls[1][1] is resident_batch
    assert tower.patch_embed.mode == "gemm"
    assert out["mode_used"] == "original"
    assert out["fallback_triggered"] is True
    assert out["fallback_reason"] == "p1_boundary"
    assert out["fast_probabilities"] == pytest.approx([0.48, 0.40, 0.04, 0.04, 0.04])
    assert out["probabilities"] == pytest.approx([0.82, 0.10, 0.03, 0.03, 0.02])
    assert out["decision"] is AbstentionDecision.FLAG

    # Non-triggered path runs Stage 1 only and omits fast_probabilities.
    observed_calls.clear()
    fast_only_runner = CascadeRunner(
        tower,
        lambda b: (observed_calls.append((tower.patch_embed.mode, b)) or [0.02, 0.90, 0.03, 0.03, 0.02]),
        cfg=ALERT_SAFE_CONFIG,
    )
    fast_out = fast_only_runner.run(resident_batch)
    assert len(observed_calls) == 1 and observed_calls[0][0] == "gemm"
    assert fast_out["mode_used"] == "gemm"
    assert fast_out["fallback_triggered"] is False
    assert fast_out["fallback_reason"] == "none"
    assert "fast_probabilities" not in fast_out
    assert fast_out["decision"] is AbstentionDecision.ACCEPT

    # Exception in Stage 2 MUST still restore FastPatchEmbed mode back to "gemm" via try...finally.
    def failing_stage2(batch):
        _ = batch
        if tower.patch_embed.mode == "gemm":
            return [0.50, 0.35, 0.05, 0.05, 0.05]
        raise RuntimeError("simulated Stage 2 OOM or device error")

    err_runner = CascadeRunner(tower, failing_stage2, cfg=ALERT_SAFE_CONFIG)
    with pytest.raises(RuntimeError, match="simulated Stage 2"):
        err_runner.run(resident_batch)
    assert tower.patch_embed.mode == "gemm"

    # Thread-lock protection: concurrent calls must serialize without interleaving mode changes.
    active_workers = 0
    max_concurrent = 0
    counter_guard = threading.Lock()
    thread_errors: list[BaseException] = []

    def concurrent_score_fn(batch):
        nonlocal active_workers, max_concurrent
        with counter_guard:
            active_workers += 1
            max_concurrent = max(max_concurrent, active_workers)
        try:
            time.sleep(0.005)
            if tower.patch_embed.mode == "gemm" and batch["trigger"]:
                return [0.45, 0.40, 0.05, 0.05, 0.05]
            return [0.02, 0.90, 0.03, 0.03, 0.02]
        finally:
            with counter_guard:
                active_workers -= 1

    threaded_runner = CascadeRunner(tower, concurrent_score_fn, cfg=ALERT_SAFE_CONFIG)

    def worker_fn(idx: int) -> None:
        try:
            res = threaded_runner.run({"trigger": idx % 2 == 0})
            expected_mode = "original" if idx % 2 == 0 else "gemm"
            assert res["mode_used"] == expected_mode
        except BaseException as exc:  # pragma: no cover
            thread_errors.append(exc)

    threads = [threading.Thread(target=worker_fn, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not thread_errors
    assert max_concurrent == 1
    assert tower.patch_embed.mode == "gemm"


# ---------------------------------------------------------------------------
# 4. prm.scoring & prm.controller
# ---------------------------------------------------------------------------


def test_prm_scoring_single_token_abcd_extraction_and_dual_action_judge():
    torch = pytest.importorskip("torch")
    import prm.scoring as prm_scoring
    from prm.scoring import (
        DEFAULT_AB_TOKEN_IDS,
        DEFAULT_ABCD_TOKEN_IDS,
        PRM_CLASSES,
        extract_ab_logits,
        extract_abcd_logits,
        judge_loss,
        prm_loss,
        score_action_judge,
        score_prm,
    )

    assert DEFAULT_ABCD_TOKEN_IDS == (32, 33, 34, 35)
    assert DEFAULT_AB_TOKEN_IDS == (32, 33)
    assert "M1-A" in (prm_scoring.__doc__ or "")
    assert "ABCD" in (prm_scoring.__doc__ or "")

    # Construct [B=2, T=4, V=48] logits where t=0..2 hold distracting peaks and ONLY t=-1 holds target ABCD values.
    vocab_logits = torch.full((2, 4, 48), -5.0, dtype=torch.float32)
    vocab_logits[:, :-1, 34] = 50.0  # earlier sequence tokens must be ignored!
    # Row 0 last token: 'A' (32) -> progress (+1.0) with high confidence.
    vocab_logits[0, -1, list(DEFAULT_ABCD_TOKEN_IDS)] = torch.tensor([8.0, 0.0, -1.0, 0.0])
    # Row 1 last token: flat ABCD -> confidence ~0.25 < 0.8 -> abstains to needs_review=True.
    vocab_logits[1, -1, list(DEFAULT_ABCD_TOKEN_IDS)] = torch.tensor([0.2, 0.1, 0.0, 0.0])

    extracted = extract_abcd_logits(vocab_logits)
    assert extracted.shape == (2, 4)
    assert torch.allclose(extracted[0], torch.tensor([8.0, 0.0, -1.0, 0.0]))

    # Verify [T, V] unbatched layout extraction and [B, V] layout extraction match.
    tv_extracted = extract_abcd_logits(vocab_logits[0], input_layout="tv")
    assert tv_extracted.shape == (1, 4)
    assert torch.allclose(tv_extracted[0], extracted[0])
    ab_extracted = extract_ab_logits(vocab_logits[:, -1, :], input_layout="bv")
    assert ab_extracted.shape == (2, 2)

    prm_out = score_prm(vocab_logits, minimum_confidence=0.8)
    assert prm_out["raw_labels"] == [PRM_CLASSES[0], PRM_CLASSES[0]]
    assert prm_out["labels"] == ["progress", None]
    assert prm_out["model_rewards"] == [1.0, None]
    assert prm_out["needs_review"].tolist() == [False, True]

    # PRM weighted CE loss works at batch_size=1 without weight cancellation.
    loss_1 = prm_loss(vocab_logits[:1], [2], class_weight=torch.ones(4))
    loss_3 = prm_loss(vocab_logits[:1], [2], class_weight=torch.tensor([1.0, 1.0, 3.0, 1.0]))
    assert float(loss_3) == pytest.approx(3.0 * float(loss_1), rel=1e-6)

    # Dual binary Action Judge V3: change="no" (token 33) while goal="yes" (token 32)
    # models an already-achieved target where a redundant click caused no state change.
    change_vocab = torch.zeros((1, 3, 48), dtype=torch.float32)
    change_vocab[0, -1, list(DEFAULT_AB_TOKEN_IDS)] = torch.tensor([-3.0, 4.0])  # B -> "no" (False)
    goal_vocab = torch.zeros((1, 3, 48), dtype=torch.float32)
    goal_vocab[0, -1, list(DEFAULT_AB_TOKEN_IDS)] = torch.tensor([4.5, -2.5])  # A -> "yes" (True)

    judge_out = score_action_judge(change_vocab, goal_vocab, minimum_confidence=0.8)
    assert judge_out["change"]["labels"] == ["no"]
    assert judge_out["change"]["verdicts"] == [False]
    assert judge_out["goal"]["labels"] == ["yes"]
    assert judge_out["goal"]["verdicts"] == [True]

    j_total, j_parts = judge_loss(
        change_vocab, [1], goal_logits=goal_vocab, goal_labels=[0], goal_weight=0.5
    )
    assert float(j_total.detach()) == pytest.approx(
        j_parts["change_ce"] + 0.5 * j_parts["goal_ce"], rel=1e-6
    )


def test_prm_controller_seven_states_idempotency_budgets_loops_and_bf16_escalation():
    assert len(ACTIONS) == 7
    assert set(ACTIONS) == {"continue", "observe", "retry", "recover", "review", "stop", "block"}

    # Feedback contract must strictly exclude privileged DOM Oracle fields.
    feedback_fields = {f.name for f in dataclasses.fields(Feedback)}
    assert all("oracle" not in name.lower() and "dom" not in name.lower() for name in feedback_fields)
    with pytest.raises(TypeError):
        Feedback(label="progress", model_reward=1.0, confidence=0.9, needs_review=False, oracle_reward=1.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Feedback(label="progress", model_reward=1.0, confidence=0.5, needs_review=True)

    fb = evaluate_progress_feedback(
        [0.85, 0.05, 0.05, 0.05],
        safety_probabilities=[0.10, 0.70, 0.10, 0.05, 0.05],
        change_probabilities=[0.90, 0.10],
        goal_probabilities=[0.10, 0.90],
    )
    assert fb.label == "progress" and fb.changed is True and fb.goal_achieved is False

    base_ctx = ControllerContext(task="checkout-shipping", action_type="left_click", idempotent=True)

    # State 1: BLOCK on trusted confirmed_block (untrusted must NOT block).
    res_block = decide_action(
        base_ctx,
        StateEvidence(trusted=True, evidence_id="ev-block", safety_event="confirmed_block"),
    )
    assert res_block.decision is ControllerAction.BLOCK
    untrusted_block = decide_action(
        base_ctx,
        StateEvidence(trusted=False, safety_event="confirmed_block"),
    )
    assert untrusted_block.decision is ControllerAction.OBSERVE

    # State 2: STOP on trusted terminal success (untrusted must NOT stop).
    res_stop = decide_action(
        base_ctx,
        StateEvidence(trusted=True, evidence_id="ev-done", terminal="success"),
    )
    assert res_stop.decision is ControllerAction.STOP

    # State 3: CONTINUE on verified local goal achieved even when stale failure is reported on checkbox.
    checkbox_ctx = ControllerContext(task="enable-newsletter", action_type="checkbox", idempotent=False)
    assert is_non_idempotent_action("checkbox", declared_idempotent=True) is True
    res_goal = decide_action(
        checkbox_ctx,
        StateEvidence(trusted=True, evidence_id="ev-goal", goal_achieved=True, action_failed=True),
    )
    assert res_goal.decision is ControllerAction.CONTINUE

    # State 4: RECOVER on verified regression when safe recovery budget remains; REVIEW when exhausted.
    res_recover = decide_action(
        ControllerContext(task="form", action_type="left_click", idempotent=True, recoveries_used=0, recovery_budget=1),
        StateEvidence(trusted=True, evidence_id="ev-reg", regression=True, safe_recovery_available=True),
    )
    assert res_recover.decision is ControllerAction.RECOVER
    res_recover_exhausted = decide_action(
        ControllerContext(task="form", action_type="left_click", idempotent=True, recoveries_used=1, recovery_budget=1),
        StateEvidence(trusted=True, evidence_id="ev-reg", regression=True, safe_recovery_available=True),
    )
    assert res_recover_exhausted.decision is ControllerAction.REVIEW

    # State 5: RETRY on verified idempotent failure with unchanged state and remaining retry budget.
    clean_fail_ev = StateEvidence(
        trusted=True,
        evidence_id="ev-fail",
        action_failed=True,
        safe_to_retry=True,
        task_state_changed=False,
    )
    res_retry = decide_action(base_ctx, clean_fail_ev)
    assert res_retry.decision is ControllerAction.RETRY

    # Non-idempotent actions (submit / checkbox / idempotent=False) MUST NEVER auto-retry -> REVIEW.
    for non_idem_ctx in (
        ControllerContext(task="order", action_type="submit", idempotent=True),
        ControllerContext(task="toggle-opt", action_type="checkbox", idempotent=False),
        ControllerContext(task="custom", action_type="left_click", idempotent=False),
    ):
        res_non_idem = decide_action(non_idem_ctx, clean_fail_ev)
        assert res_non_idem.decision is ControllerAction.REVIEW

    # Stale failure / stale signal guards:
    # - Stale signal on non-idempotent action observes once (without re-executing), then escalates to REVIEW.
    stale_fb = evaluate_progress_feedback([0.10, 0.82, 0.04, 0.04], stale_signal=True)
    stale_obs = decide_action(
        ControllerContext(task="order", action_type="submit", observations_used=0, observation_budget=1),
        StateEvidence(trusted=False),
        stale_fb,
    )
    assert stale_obs.decision is ControllerAction.OBSERVE
    stale_rev = decide_action(
        ControllerContext(task="order", action_type="submit", observations_used=1, observation_budget=1),
        StateEvidence(trusted=False),
        stale_fb,
    )
    assert stale_rev.decision is ControllerAction.REVIEW

    # - Stale failure on idempotent action observes current GUI state first instead of blind retry.
    stale_idem_fail = decide_action(base_ctx, clean_fail_ev, stale_fb)
    assert stale_idem_fail.decision is ControllerAction.OBSERVE

    # Retry budget, repetition budget, observation budget, and state-action loop detection.
    assert decide_action(
        ControllerContext(task="t", action_type="left_click", idempotent=True, retries_used=1, retry_budget=1),
        clean_fail_ev,
    ).decision is ControllerAction.REVIEW
    assert decide_action(
        ControllerContext(
            task="t", action_type="left_click", idempotent=True, same_action_attempts=2, max_same_action_attempts=2
        ),
        clean_fail_ev,
    ).decision is ControllerAction.REVIEW

    loop_seq = ("shaA:click(10,20)", "shaB:scroll", "shaA:click(10,20)")
    assert has_state_action_loop(loop_seq) is True
    assert decide_action(
        ControllerContext(task="t", action_type="left_click", idempotent=True, recent_state_action=loop_seq),
        clean_fail_ev,
    ).decision is ControllerAction.REVIEW
    assert decide_action(
        ControllerContext(task="t", action_type="left_click", idempotent=True, observations_used=1, observation_budget=1),
        StateEvidence(trusted=False),
    ).decision is ControllerAction.REVIEW

    # Original-precision BF16 safety review escalation on high_risk_action or elevated safety_p1.
    high_p1_fb = evaluate_progress_feedback(
        [0.85, 0.05, 0.05, 0.05],
        safety_probabilities=[0.72, 0.15, 0.05, 0.04, 0.04],
    )
    bf16_escalation = decide_action(base_ctx, StateEvidence(trusted=False), high_p1_fb)
    assert bf16_escalation.decision is ControllerAction.REVIEW
    assert bf16_escalation.requires_original_bf16 is True

    # require_original_bf16_safety_review fails closed on None, exception, gemm backend, or wrong identity.
    assert require_original_bf16_safety_review(None, {}, "m1a-bf16")["allow"] is False
    assert (
        require_original_bf16_safety_review(
            lambda req: (_ for _ in ()).throw(RuntimeError("rpc error")), {}, "m1a-bf16"
        )["status"]
        == "review"
    )
    assert (
        require_original_bf16_safety_review(
            lambda req: {"backend": "gemm", "identity": "m1a-bf16", "allow": True}, {}, "m1a-bf16"
        )["allow"]
        is False
    )
    assert (
        require_original_bf16_safety_review(
            lambda req: {"backend": "original_bf16", "identity": "wrong-id", "allow": True},
            {},
            "m1a-bf16",
        )["allow"]
        is False
    )
    verified_bf16 = require_original_bf16_safety_review(
        lambda req: {"backend": "original_bf16", "identity": "m1a-bf16", "allow": True, "p1": 0.04},
        {"step": 1},
        "m1a-bf16",
    )
    assert verified_bf16["status"] == "verified" and verified_bf16["allow"] is True

    # conservative_reward ignores advisory model reward unless trusted evidence confirms progress.
    assert conservative_reward(StateEvidence(trusted=False, progress=True), ModelSignal("progress", 0.99)) is None
    assert conservative_reward(StateEvidence(trusted=True, evidence_id="ev1", progress=True), fb) == 1.0
    assert conservative_reward(StateEvidence(trusted=True, evidence_id="ev2", regression=True), fb) == -1.0
    assert conservative_reward(StateEvidence(trusted=True, evidence_id="ev3", task_state_changed=False), fb) == 0.0


# ---------------------------------------------------------------------------
# 5. diagnostics.selective_abstention: TS vs Platt p1 ROC-AUC & Brier scores
# ---------------------------------------------------------------------------


def test_temperature_scaling_alters_cross_sample_p1_auc_while_platt_preserves_order():
    # Constructed 5-class dataset demonstrating softmax denominator coupling:
    # Compare Sample 2 (positive, label 0) vs Sample 3 (negative, label 1):
    # - Sample 2 has evenly spread rival logits: [0.0, 0.95, 0.90, 0.90, 0.90]
    #   At T=1.0: sum_{k>0} exp(z_k) = exp(0.95) + 3*exp(0.90) = 9.9645 -> p1 = 0.09120
    # - Sample 3 has one concentrated rival and three suppressed classes: [0.0, 2.40, -8.0, -8.0, -8.0]
    #   At T=1.0: sum_{k>0} exp(z_k) = exp(2.40) + 3*exp(-8.0) = 11.0242 -> p1 = 0.08317
    # Thus at T=1.0, positive Sample 2 outranks negative Sample 3 (p1: 0.09120 > 0.08317).
    # For T > 1 (e.g., T = 2.0 or fitted T on overconfident errors), exp(2.40/T) shrinks much
    # faster than 4 moderate exponentials:
    # - At T=2.0: Sample 2 has sum = exp(0.475) + 3*exp(0.45) = 6.313 -> p1 = 0.13674
    #             Sample 3 has sum = exp(1.20) + 3*exp(-4.0) = 3.375  -> p1 = 0.22857
    # Sample 3 overtakes Sample 2 on p1, strictly reducing cross-sample p1 ROC-AUC while
    # every single row's argmax stays identical!
    logits = [
        [4.5, 0.2, -1.0, -1.0, -1.0],    # 0: clear hijack (label 0, argmax 0)
        [3.8, 0.1, -0.5, -0.5, -0.5],    # 1: clear hijack (label 0, argmax 0)
        [0.0, 0.95, 0.90, 0.90, 0.90],   # 2: hijack with spread rivals (label 0, argmax 1)
        [0.0, 2.40, -8.0, -8.0, -8.0],   # 3: benign with concentrated rival (label 1, argmax 1)
        [-3.0, 3.5, 0.0, -1.0, -1.0],    # 4: clear benign (label 1, argmax 1)
        [-2.5, 0.0, 3.2, -1.0, -1.0],    # 5: clear benign (label 2, argmax 2)
        [-2.0, 5.0, -1.0, -1.0, -1.0],   # 6: overconfident error driving fitted T > 1 (label 3, argmax 1)
    ]
    labels = [0, 0, 0, 1, 1, 2, 3]

    # 1) Explicit T = 2.0 audit:
    explicit_audit = audit_temperature_vs_platt_auc(logits, labels, temperature=2.0)
    assert explicit_audit["argmax_preserved"] is True
    assert explicit_audit["raw_roc_auc"] == pytest.approx(1.0)
    assert explicit_audit["ts_roc_auc"] == pytest.approx(11 / 12)
    assert explicit_audit["ts_roc_auc"] != explicit_audit["raw_roc_auc"]
    assert explicit_audit["ts_auc_changed"] is True
    assert explicit_audit["ts_pair_inversions"] >= 1
    assert explicit_audit["platt_a"] > 0.0
    assert explicit_audit["platt_roc_auc"] == pytest.approx(explicit_audit["raw_roc_auc"], abs=1e-12)
    assert explicit_audit["platt_order_preserving"] is True

    # 2) End-to-end fitted temperature (via fit_temperature) on the same 5-class dataset:
    fitted_audit = audit_temperature_vs_platt_auc(logits, labels)
    assert fitted_audit["temperature"] > 1.3
    assert fitted_audit["argmax_preserved"] is True
    assert fitted_audit["ts_roc_auc"] != fitted_audit["raw_roc_auc"]
    assert fitted_audit["ts_auc_changed"] is True
    assert fitted_audit["platt_a"] > 0.0
    assert fitted_audit["platt_roc_auc"] == pytest.approx(fitted_audit["raw_roc_auc"], abs=1e-12)
    assert fitted_audit["platt_order_preserving"] is True

    # Direct check on binary_platt + apply_binary_platt strict monotonicity when a > 0.
    raw_scores = [0.05, 0.12, 0.25, 0.60, 0.88, 0.95]
    bin_targets = [0, 0, 1, 0, 1, 1]
    slope, intercept = binary_platt(raw_scores, bin_targets)
    assert slope > 0.0
    calibrated = apply_binary_platt(raw_scores, slope, intercept)
    assert all(calibrated[i] < calibrated[i + 1] for i in range(len(calibrated) - 1))
    assert roc_auc(calibrated, bin_targets) == pytest.approx(roc_auc(raw_scores, bin_targets), abs=1e-12)


def test_brier_multiclass_and_brier_binary():
    # Perfect predictions -> 0.0 Brier score.
    assert brier_multiclass([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0, 1]) == pytest.approx(0.0)
    assert brier_binary([0.0, 1.0], [0, 1]) == pytest.approx(0.0)

    # Hand-computed 3-class Brier:
    # Row 0: p=[0.7, 0.2, 0.1], y=0 -> (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    # Row 1: p=[0.2, 0.5, 0.3], y=2 -> 0.2^2 + 0.5^2 + (0.3-1)^2 = 0.04 + 0.25 + 0.49 = 0.78
    # Mean = (0.14 + 0.78) / 2 = 0.46
    mc_brier = brier_multiclass([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]], [0, 2])
    assert mc_brier == pytest.approx(0.46)

    # Hand-computed binary Brier: ((0.2 - 0)^2 + (0.8 - 1)^2) / 2 = 0.04
    assert brier_binary([0.2, 0.8], [0, 1]) == pytest.approx(0.04)

    with pytest.raises(ValueError):
        brier_multiclass([[0.6, 0.6]], [0])  # non-simplex
    with pytest.raises(ValueError):
        brier_binary([1.2], [1])


def test_two_cascade_runners_cannot_interleave_patch_modes():
    import threading
    from fastpe.cascade import CascadeRunner

    class Target:
        def __init__(self):
            self.mode = "gemm"
        def set_mode(self, mode):
            self.mode = mode

    target = Target()
    entered = threading.Event()
    release = threading.Event()
    second_scored = threading.Event()
    seen = []

    def first_score(_):
        if target.mode == "original":
            entered.set()
            assert release.wait(5)
            seen.append(target.mode)
            return [.1, .3, .2, .2, .2]
        return [.5, .2, .1, .1, .1]

    def second_score(_):
        second_scored.set()
        return [.1, .3, .2, .2, .2]

    first = CascadeRunner(target, first_score)
    second = CascadeRunner(target, second_score)
    t1 = threading.Thread(target=lambda: first.run(None))
    t2 = threading.Thread(target=lambda: second.run(None))
    t1.start()
    assert entered.wait(5)
    t2.start()
    assert not second_scored.wait(.15)
    release.set()
    t1.join(5)
    t2.join(5)
    assert not t1.is_alive() and not t2.is_alive()
    assert seen == ["original"]
    assert second_scored.is_set()
