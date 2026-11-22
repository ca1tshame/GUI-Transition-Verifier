"""Four-class OpenApps Semantic Verifier LoRA trainer with V-Val threshold selection.

Training uses ``micro_batch_size = 1`` by default so multi-image M-RoPE rotary
positions during training exactly match single-sample online verification.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import shutil
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eval.v5_recompute import metrics_from_confusion_matrix
from fastpe import apply_fastpe
from policy.collator import VERIFIER_LABELS, build_verifier_messages, collate_verifier_batch
from policy.train_policy_sft import find_lm_lora_targets, sha256_dir


@dataclass(frozen=True)
class VerifierSFTConfig:
    """Hyperparameters for 4-class single-token ``ABCD`` semantic verifier LoRA."""

    model_path: str
    v_train_jsonl: str
    v_val_jsonl: str
    output_dir: str
    epochs: int = 2
    micro_batch_size: int = 1
    grad_accum: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    weight_cap: float = 3.5
    max_false_a_rate: float = 0.10
    pixels: int = 131072
    seed: int = 20261122
    fast_patch_embed: bool = True


def compute_class_weights(labels: Sequence[str], cap: float = 3.5) -> list[float]:
    """Compute smoothed inverse-sqrt class weights across ``A/B/C/D``."""
    counts = collections.Counter(labels)
    total = len(labels)
    return [
        round(min(cap, math.sqrt(total / (4.0 * max(1, counts.get(lbl, 1))))), 6)
        for lbl in VERIFIER_LABELS
    ]


def tune_thresholds_on_vval(
    predictions: Sequence[Mapping[str, Any]],
    max_false_a_rate: float = 0.10,
) -> dict[str, float]:
    """Select ``(tau_c, tau_b)`` strictly on V-Val subject to false-alarm cap on true A."""
    true_a = [r for r in predictions if r["gt"] == "A"]
    true_c = [r for r in predictions if r["gt"] == "C"]
    true_b = [r for r in predictions if r["gt"] == "B"]
    best = {"tau_c": 0.55, "tau_b": 0.65, "score": -1.0, "false_a_rate": 0.0, "c_rec": 0.0, "b_rec": 0.0}
    for tc in (0.35, 0.45, 0.55, 0.65, 0.75):
        for tb in (0.45, 0.55, 0.65, 0.75):
            fa = sum(1 for r in true_a if (r["probs"]["C"] >= tc or r["probs"]["B"] >= tb)) / max(1, len(true_a))
            if fa > max_false_a_rate:
                continue
            cr = sum(1 for r in true_c if r["probs"]["C"] >= tc) / max(1, len(true_c))
            br = sum(1 for r in true_b if (r["probs"]["B"] >= tb or r["probs"]["C"] >= tc)) / max(1, len(true_b))
            score = 0.65 * cr + 0.35 * br - 0.5 * fa
            if score > best["score"]:
                best = {
                    "tau_c": tc,
                    "tau_b": tb,
                    "score": round(score, 4),
                    "false_a_rate": round(fa, 4),
                    "c_rec": round(cr, 4),
                    "b_rec": round(br, 4),
                }
    return best


def _predictions_to_confusion(preds: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    conf = {g: {p: 0 for p in VERIFIER_LABELS} for g in VERIFIER_LABELS}
    for r in preds:
        conf[r["gt"]][r["pred"]] += 1
    return conf


def run_verifier_training(cfg: VerifierSFTConfig) -> dict[str, Any]:
    """Train 4-class Semantic Verifier LoRA on V-Train and lock thresholds on V-Val."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

    v_train = [
        json.loads(line)
        for line in pathlib.Path(cfg.v_train_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    v_val = [
        json.loads(line)
        for line in pathlib.Path(cfg.v_val_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    run_dir = pathlib.Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    processor = AutoProcessor.from_pretrained(
        cfg.model_path, local_files_only=True, min_pixels=cfg.pixels, max_pixels=cfg.pixels
    )
    abcd_ids = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in VERIFIER_LABELS]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if cfg.fast_patch_embed:
        apply_fastpe(model, mode="gemm")

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=find_lm_lora_targets(model),
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    cls_weights = torch.tensor(
        compute_class_weights([r["label"] for r in v_train], cap=cfg.weight_cap),
        device="cuda",
        dtype=torch.float32,
    )

    micros_per_epoch = math.ceil(len(v_train) / cfg.micro_batch_size)
    steps_per_epoch = math.ceil(micros_per_epoch / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(15, max(1, total_steps // 10)), num_training_steps=total_steps
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best_epoch = 1
    best_f1 = -1.0
    best_thresh: dict[str, float] | None = None
    epoch_evals = []

    for epoch in range(cfg.epochs):
        order = list(range(len(v_train)))
        random.Random(cfg.seed + epoch).shuffle(order)
        micros = [order[i : i + cfg.micro_batch_size] for i in range(0, len(order), cfg.micro_batch_size)]
        for g_start in range(0, len(micros), cfg.grad_accum):
            group = micros[g_start : g_start + cfg.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            for mb in group:
                batch = collate_verifier_batch(processor, [v_train[idx] for idx in mb])
                tgt = batch.pop("target_idx").to("cuda")
                logits = model(**{k: v.to("cuda") for k, v in batch.items()}, use_cache=False).logits[
                    :, -1, abcd_ids
                ].float()
                # Note: with reduction="mean" (PyTorch default), F.cross_entropy normalizes by
                # cls_weights[tgt].sum(), so at micro_batch_size == 1 (single target token) the
                # class weight cancels out (w_y * nll / w_y == nll) even with grad accumulation.
                # Kept unchanged here to preserve fidelity with the historical V5 Verifier run;
                # future retraining should apply (F.cross_entropy(..., reduction="none") * cls_weights[tgt]).mean()
                # as in training/losses.py.
                loss = F.cross_entropy(logits, tgt, weight=cls_weights) / len(group)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite verifier training loss")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds = []
        with torch.inference_mode():
            for row in v_val:
                msg = build_verifier_messages(
                    row["before_image"], row["after_image"], row["instruction"], row["action"], row.get("visible_history") or []
                )
                txt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                with Image.open(row["before_image"]) as ib, Image.open(row["after_image"]) as ia:
                    inp = processor(text=[txt], images=[ib.convert("RGB"), ia.convert("RGB")], padding=False, return_tensors="pt")
                logits = model(**{k: v.to("cuda") for k, v in inp.items()}, use_cache=False).logits[0, -1, abcd_ids].float()
                probs = logits.softmax(dim=-1).cpu().tolist()
                pred_lbl = VERIFIER_LABELS[max(range(4), key=probs.__getitem__)]
                val_preds.append(
                    {"sample_id": row["sample_id"], "gt": row["label"], "pred": pred_lbl, "probs": dict(zip(VERIFIER_LABELS, probs))}
                )
        model.train()

        metrics = metrics_from_confusion_matrix(_predictions_to_confusion(val_preds))
        thresh = tune_thresholds_on_vval(val_preds, max_false_a_rate=cfg.max_false_a_rate)
        ep_dir = run_dir / f"adapter_epoch_{epoch + 1}"
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
        model.save_pretrained(ep_dir, safe_serialization=True)
        epoch_evals.append(
            {
                "epoch": epoch + 1,
                "adapter_path": str(ep_dir),
                "adapter_sha256": sha256_dir(ep_dir),
                "v_val_metrics": metrics,
                "tuned_thresholds_vval": thresh,
            }
        )
        if metrics["macro_f1_4cls"] > best_f1:
            best_f1 = metrics["macro_f1_4cls"]
            best_epoch = epoch + 1
            best_thresh = thresh

    best_dst = run_dir / "best_adapter"
    if best_dst.exists():
        shutil.rmtree(best_dst)
    shutil.copytree(epoch_evals[best_epoch - 1]["adapter_path"], best_dst)
    manifest = {
        "status": "COMPLETED",
        "v_train_count": len(v_train),
        "v_val_count": len(v_val),
        "best_epoch": best_epoch,
        "best_adapter_path": str(best_dst),
        "best_adapter_sha256": sha256_dir(best_dst),
        "selected_thresholds_vval": best_thresh,
        "epoch_evals": epoch_evals,
        "gpu_seconds": round(time.time() - t0, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
    }
    (run_dir / "verifier_train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train 4-class OpenApps Semantic Verifier LoRA.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--v-train", required=True)
    ap.add_argument("--v-val", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--no-fast-patch-embed", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = VerifierSFTConfig(
        model_path=args.model_path,
        v_train_jsonl=args.v_train,
        v_val_jsonl=args.v_val,
        output_dir=args.output_dir,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        fast_patch_embed=not args.no_fast_patch_embed,
    )
    print(json.dumps(run_verifier_training(cfg), indent=2))


if __name__ == "__main__":
    main()
