# Experiments & Full Ablation Records

> Complete benchmark tables, training configurations, and statistical verification records backed by [`results/`](../results/) and [`eval/v5_recompute.py`](../eval/v5_recompute.py).

---

## 1. OpenApps Policy LoRA Post-Training (`results/training_configs.json`)

- **Backbone**: `Qwen/Qwen3-VL-8B-Instruct` (`8.81B` total params; vision tower frozen; LM-only LoRA `r=16, alpha=32, dropout=0.05` across `252` modules = `43,646,976` trainable params, `0.4954%`).
- **M3 Curated Dataset (`4,534` rows)**: `3,291` CPU expert traces + `47` clean multi-step visual teacher traces (`209` short/1080p noisy fragments dropped) + `1,196` Train-only skill-gap fill rows (`163` unique tasks, `607` episodes).
- **Leakage Isolation (`eval/leak_guard.py`)**: `task_overlap = 0`, `episode_overlap = 0`, `image_overlap = 0` against Dev-19 and Heldout-12.
- **Training Budget**: `2` epochs (`568` optimizer steps, effective batch size `16`, `lr=1e-4`, cosine schedule, `fastPE` enabled, `7,416.35` GPU seconds, `30.00 GB` peak VRAM).

### Epoch × Data Quality Double Dissociation on OpenApps Dev-19

All runs share identical inference settings (`131072` px, `1024×576` desktop, `seed=911`, `max_steps=25`, `fastPE` enabled, official per-app-group `CompositeTask` evaluation in `policy/eval_openapps.py`):

| Policy Checkpoint | Train Mix | Dev-380 NLL | Dev-19 Whole-Task | Wilson 95% CI (`n=19`) | Official Subtasks (`n=55`) |
| --- | --- | ---: | ---: | --- | ---: |
| `m0_zeroshot` (V4 baseline) | Zero-shot base | — | `0/19` (`0.00%`) | `[0.0000, 0.1682]` | `0/55` |
| `dev_m1_ep1` (Step 194) | `3,103` expert-only | `0.16664` | `2/19` (`10.53%`) | `[0.0294, 0.3139]` | `17/55` (`30.91%`) |
| `dev_m2_ep1` (Step 194) | `2,877` expert + `226` visual teacher | `0.15424` | `8/19` (`42.11%`) | `[0.2314, 0.6372]` | `30/55` (`54.55%`) |
| `dev_m1_ep2` (Step 388) | `3,103` expert-only | `0.21898` | `9/19` (`47.37%`) | `[0.2733, 0.6829]` | `34/55` (`61.82%`) |
| `m2_ep2_v4` (Step 388, V4) | `2,877` expert + `226` visual teacher | `0.19614` | `3/19` (`15.79%`) | `[0.0552, 0.3757]` | `37/55` (V4 heuristic) |
| `dev_m3_ep1` (Step 284) | `4,534` skill-gap M3 | `0.02739` | `8/19` (`42.11%`) | `[0.2314, 0.6372]` | `32/55` (`58.18%`) |
| **`dev_m3_ep2` (`BEST_POLICY`, Step 568)** | **`4,534` skill-gap M3** | **`0.03287`** | **`10/19` (`52.63%`)** | **`[0.3171, 0.7267]`** | **`37/55` (`67.27%`)** |

* **Offline NLL vs. Online Execution**: `dev_m1_ep1` has lower offline NLL (`0.16664`) than `dev_m1_ep2` (`0.21898`) yet solves `2/19` vs `9/19` (`McNemar p = 0.015625`), as Ep1 under-converges on submit clicks. `M2` peaks at Ep1 (`8/19`) and drops to `3/19` at Ep2 with form-abandonment events (`4` vs `0` for `M1`).
* **Click Calibration (`results/performance_benchmarks.json`)**: On matched `M1 ∩ M2` OpenApps Dev clicks (`n=139`, `[0, 1000]^2` grid), `M1` maintains `37.17` mean Euclidean L2 error (`p90 = 39.32`) vs `M2`'s `74.20` (`p90 = 170.36`).

---

## 2. Controlled 3-Arm Dev-19 & One-Shot Frozen Heldout-12

Using frozen `M3 Ep2`, frozen `adapted_v5`, and `V-Val`-locked thresholds (`tau_c = 0.65`, `tau_b = 0.65`, zero tuning on Dev or Heldout; Heldout `seed=915` across 6 task components, run strictly once):

| Split & Arm | Mode | Whole-Task Pass | Wilson 95% CI | Official Subtasks | Wall Time / Steps | Interventions (`Rule / Verifier`) |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Dev-19 Arm A (`dev_m3_ep2`) | `policy_only` | `10/19` (`52.63%`) | `[0.3171, 0.7267]` | `37/55` (`67.27%`) | `2067.8s` / 392 | `0 / 0` |
| Dev-19 Arm B (`dev_arm_b_rules`) | `rules` | `10/19` (`52.63%`) | `[0.3171, 0.7267]` | `37/55` (`67.27%`) | `2077.6s` / 393 | `2 / 0` |
| Dev-19 Arm C (`dev_arm_c_verifier`) | `verifier` | `10/19` (`52.63%`) | `[0.3171, 0.7267]` | `37/55` (`67.27%`) | `2262.1s` / 393 | `2 / 0` (`383` calls) |
| Heldout-12 (`heldout_m0`) | Zero-shot `policy_only` | `0/12` (`0.00%`) | `[0.0000, 0.2425]` | `1/36` (`2.78%`) | `480.4s` / 92 | `0 / 0` |
| **Heldout-12 (`heldout_arm_a`)** | `policy_only` (`M3 Ep2`) | **`7/12` (`58.33%`)** | **`[0.3195, 0.8067]`** | **`27/36` (`75.00%`)** | `1338.4s` / 257 | `0 / 0` (`vs M0 p=0.015625`) |
| Heldout-12 (`heldout_arm_b`) | `rules` | `7/12` (`58.33%`) | `[0.3195, 0.8067]` | `27/36` (`75.00%`) | `1352.7s` / 258 | `0 / 0` (`vs Arm A p=1.0`) |
| Heldout-12 (`heldout_arm_c`) | `verifier` | `6/12` (`50.00%`) | `[0.2538, 0.7462]` | `26/36` (`72.22%`) | `1556.0s` / 274 | `0 / 1` (`268` calls, `p=1.0`) |

---

## 3. Semantic Verifier, Safety Classification & Process Control

### OpenApps Dev-936 4-Class Semantic Verifier (`results/verifier_metrics.json`)

Read-only `936`-step benchmark (`A=718` valid progress, `B=43` stagnation, `C=160` semantic error, `D=15` observe/recover) labeled from observable transition evidence (`policy/step_labeler.py`). `adapted_v5` was trained on disjoint Train transitions (`V-Train=713`, `V-Val=582`) at `micro_batch_size=1` for Qwen3-VL M-RoPE parity:

| Verifier (Dev-936) | Accuracy | Macro-F1 (4-cls) | Class-B Stagnation Rec | Class-C Semantic Err Prec / Rec | Valid-`A` False Alarm | Latency (ms/step) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reward_v3` (MiniWoB 0-shot) | `0.1261` | `0.0987` | `0.9767` | `0.1415` / `0.0938` | `0.7702` | `301.37` |
| `reward_biz` (MiniWoB+biz 0-shot) | `0.1891` | `0.1313` | `0.9535` | `0.1143` / `0.0750` | `0.6602` | `300.71` |
| `judge_v3` (MiniWoB binary 0-shot) | `0.1250` | `0.0704` | `1.0000` | `0.0000` / `0.0000` | `0.8969` | `298.01` |
| `rswt_m1a` (RSWT safety 0-shot) | `0.7329` | `0.2148` | `0.0000` | `1.0000` / `0.0063` (`1/160`) | `0.0000` | `297.04` |
| **`adapted_v5` (Zero-Leakage LoRA)** | **`0.8194`** | **`0.4510`** | **`0.8837` (`38/43`)** | **`0.9231` (`12/13`) / `0.0750` (`12/160`)** | **`0.0014` (`1/718`)** | `382.99` |

### RSWT Safety Verification, Protocol Alignment & PRM Control

| Pillar | Benchmark & Result |
| --- | --- |
| **RSWT 5-Class Safety (`training/`)** | Sample-level `batch=1` class-weight fix + cross-term-free rank-concatenated `adapter_soup` (`3.73e-9` merge error). On 966-row RSWT Test: `B0 47.09% → M1-A 68.57% → SOUP-AE 68.77%` Macro-F1; `0/265` benign false alarms (`1.12%` Clopper-Pearson upper bound). |
| **Tool Protocol (`protocols/`)** | Versioned `<tool_call>` schema + single `[0, 1000]^2` coordinate scaling: `0/80 → 59/80` unpaired MiniWoB++ V2 task completion. |
| **PRM & 7-State Guard (`prm/`)** | Judge V3 `136/136` on controlled paired state-change split; 4-class PRM `81.04%` Macro-F1; fail-closed 7-state controller `24/24` controlled multi-step tasks (`10/10` non-idempotent guard). |

---

## 4. `fastPE` Operator Speedup & Parity (`results/performance_benchmarks.json`)

Replacing Qwen3-VL's non-overlapping 3D patch `nn.Conv3d` (`kernel == stride == (2, 16, 16)`) with parameter-shared `F.linear` across two independent benchmark configurations:

1. **OpenApps Policy SFT + Inference (`Qwen3-VL-8B` on A800)**:
   - Tensor forward (`n=16`): `mean_abs=0.0`, `max_abs=0.007812` (1 BF16 ULP), `13,516.68 ms → 2.46 ms`.
   - LoRA gradient (`504` tensors): `loss_abs_diff=0.00615`, `global_grad_cos_sim=0.976824`, **`5.75×`** end-to-end training microbatch speedup (`32.909s → 5.726s`).
   - Greedy policy inference (`n=20` Dev samples): **`2.21×`** speedup (`6.638s → 3.005s`), `20/20` action-type parity, `18/20` exact raw match (`coord_l2_mean=5.116`).
2. **RSWT Safety Scoring + `CascadeRunner` (966-row Test)**:
   - Unconditional `fastPE`: **`50.8×`** E2E P50 speedup (`7,524.8 → 148.0 ms`), `15/966` (`1.55%`) argmax flips, `3` missed `1_hijack` alerts.
   - **Alert-Safe Cascade** (`p1 ∈ [0.30, 0.75]`): **`24.12×`** speedup (`313.06 ms` mean, `2.17%` fallback), **0 missed `1_hijack` alerts**.
   - **Full-Parity Cascade** (`+ conf < 0.70`): **`4.62×`** speedup (`19.77%` fallback), **`0/966`** argmax and risk-band flips.

---

## 5. Full GPU Training, Serving & Demo Recording Commands

Requires local `Qwen/Qwen3-VL-8B-Instruct` weights, `pip install ".[all]"`, and an external Playwright + BrowserGym + OpenApps setup:

```bash
# 1. Policy LoRA SFT (M3)
python -B -m policy.train_policy_sft --model-path /path/to/Qwen3-VL-8B-Instruct \
  --split /path/to/train.jsonl --manifest /path/to/manifest.json \
  --prompt-config /path/to/prompt_cfg.json --output-dir runs/m3_policy

# 2. Semantic Verifier LoRA SFT (micro_batch_size=1 for M-RoPE parity)
python -B -m policy.train_verifier_sft --model-path /path/to/Qwen3-VL-8B-Instruct \
  --v-train /path/to/v_train.jsonl --v-val /path/to/v_val.jsonl --output-dir runs/verifier_v5

# 3. Single-base Dual-LoRA router & closed-loop Dev evaluation (do NOT re-run Heldout for tuning/demo selection)
python -B -m policy.dual_lora_router --model-path /path/to/Qwen3-VL-8B-Instruct \
  --prompt-config /path/to/prompt_cfg.json --policy-adapter runs/m3_policy/final_adapter \
  --verifier-adapter runs/verifier_v5/best_adapter --port 39850
python -B -m policy.eval_openapps --run-tag dev_demo --split dev --mode policy_only \
  --api-url <SERVER_URL> --task-freeze-json /path/to/openapps_task_freeze.json \
  --tasks-yaml /path/to/longer_horizon.yaml --output-dir runs/eval_demo

# 4. Assemble a verified, sanitized rollout GIF from any successful Dev episode
python -B tools/record_openapps_gif.py \
  --episode runs/eval_demo/dev_demo/episodes/<TASK_ID>.json \
  --frames-dir runs/eval_demo/dev_demo/frames/<TASK_ID> \
  --out assets/openapps_dev_rollout.gif
```
