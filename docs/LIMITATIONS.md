# Known Limitations, Negative Results & Root-Cause Audits

This document preserves negative experiments, statistical bounds, and numerical edge cases discovered during development and post-hoc verification.

---

## 1. Why Online Verifier Feedback (`Arm C`) Did Not Improve End-to-End Success

- **Observed Result**: On Dev-19, `Arm C (+ adapted_v5)` tied pure policy `Arm A` at `10/19` (`383` verifier calls, `0` interventions, `+0.76s/step` latency). On one-shot Heldout-12, `Arm A/B` achieved `7/12` (`58.33%`) while `Arm C` scored `6/12` (`50.00%`, `McNemar p=1.0`, `268` calls, `1` intervention). On the single diverged task, the Verifier intervened **0 times** across all 25 steps (step-6 dock click variance under dual-LoRA execution rather than an erroneous intervention).
- **Low Semantic-Error Recall**: While `adapted_v5` cuts Dev-936 valid-progress false alarms from `66%–90%` to `0.14%` (`1/718`) and detects Class-B stagnation at `88.37%` (`38/43`), **Class-C semantic-error recall is only `7.50%` (`12/160`)**, leaving conservative `V-Val` thresholds (`tau_c=0.65, tau_b=0.65`) rarely triggered against a strong `M3` policy (`0` invalid actions).
- **Root-Cause Code Audit (`batch=1` Class-Weight Cancellation)**: Post-hoc audit of `policy/train_verifier_sft.py` revealed that calling `F.cross_entropy(logits, tgt, weight=cls_weights)` at `micro_batch_size=1` (chosen to preserve Qwen3-VL M-RoPE parity across paired screenshots) **mathematically cancels class weights** in PyTorch's default `reduction="mean"` (`w_y * nll / w_y = nll`), even with gradient accumulation. While `training/losses.py` implements and verifies the sample-level weighting fix on 5-class RSWT (`47.09% → 68.77%`), re-training the 4-class OpenApps Verifier with that fix remains an open ablation.

---

## 2. Small-Sample Evaluation Bounds

- OpenApps Dev (`n=19`) and one-shot frozen Heldout (`n=12`, `6` task components) carry wide 95% Wilson confidence intervals (`> ±20 pp`: `[31.71%, 72.67%]` on Dev `M3`, `[31.95%, 80.67%]` on Heldout `Arm A`).
- These sample sizes support statistically significant gains over zero-shot `M0` (`p=0.00195` on Dev, `p=0.01563` on Heldout) rather than fine-grained `±1`-task comparisons or claims of universal GUI benchmark dominance.
- For Heldout-12, per-task vectors and per-task subtask counts are withheld to avoid combinatorial re-identification, while aggregate totals and paired 2×2 contingency tables allow exact recomputation in `eval/v5_recompute.py`.

---

## 3. BF16 Numerical Drift in `FastPatchEmbed`

- `FastPatchEmbed` (`Conv3d → F.linear`) is algebraically exact in FP64 (`<1e-12` error), but BF16 GEMM reduction-order differences introduce small forward drift (`max_abs=0.007812`, `1` BF16 ULP; `grad_cos_sim=0.976824`).
- Empirically, this causes `15/966` (`1.55%`) argmax flips (`3` missed `1_hijack` alerts) under unconditional `fastPE` on RSWT, and `18/20` exact raw match (`20/20` action-type parity) on greedy policy inference.
- `fastpe/cascade.py` resolves this via empirical `Alert-Safe` (`24.12×`, `0` missed alerts) and `Full-Parity` (`4.62×`, `0/966` flips) fallback gates.

---

## 4. Calibration, Gradient Conflict & Leakage Audit Findings

- **Data Leakage (`eval/leak_guard.py`)**: Near-duplicate screening via `dHash+gray32` identified and filtered `62.51%` near-duplicate transitions in legacy `frozen_v1` splits.
- **Layer Gradient Conflict (`diagnostics/gradient_conflict.py`)**: Probing `144` LoRA modules revealed `-0.8018` Class-4 gradient opposition at layer 28 (alongside `11/12` local visual ambiguity on Class-4 Dev triples).
- **Calibration Monotonicity (`diagnostics/selective_abstention.py`)**: Temperature Scaling applied to multi-class probabilities flipped `15/22` cross-sample `p1` ROC-AUC ranking pairs on RSWT, whereas binary Platt scaling preserved strict rank monotonicity.
- **Modality Ablation**: Removing action text on RSWT (`E4`) lowered Macro-F1 by only `-0.7 pp`; MiniWoB++ `0/80 → 59/80` represents unpaired V1/V2 zero-shot alignment runs.

---

## 5. Licensing & Withheld Assets

- Source code in this repository is licensed under **Apache-2.0** ([`LICENSE`](../LICENSE)).
- No model checkpoints, screenshots, private paths, or third-party benchmark suites are redistributed. Experiments on **OpenApps** rely on its upstream repository (**CC BY-NC 4.0**), and `Qwen3-VL-8B-Instruct` remains governed by its upstream license.
