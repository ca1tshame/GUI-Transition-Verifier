"""Language-tower LoRA SFT trainer for Qwen3-VL GUI policies (M1/M2/M3 pipeline)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import shutil
import time
from dataclasses import dataclass
from typing import Any, Sequence

from fastpe import apply_fastpe
from policy.collator import Qwen3VLPolicyCollator

LM_PROJ_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class PolicySFTConfig:
    """Hyperparameter and safety configuration for Qwen3-VL LM-only LoRA SFT."""

    model_path: str
    split_jsonl: str
    manifest_json: str
    prompt_config_json: str
    output_dir: str
    dev_jsonl: str | None = None
    epochs: int = 2
    micro_batch_size: int = 8
    grad_accum: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    pixels: int = 131072
    seed: int = 20261122
    fast_patch_embed: bool = True


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(directory: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in directory.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(directory)).encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


def find_lm_lora_targets(model: Any) -> list[str]:
    """Select exclusively language-model linear projections (freezing vision tower)."""
    import torch

    targets = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and name.startswith("model.language_model.")
        and any(name.endswith("." + suffix) for suffix in LM_PROJ_SUFFIXES)
    ]
    if not targets:
        raise RuntimeError("No language_model LoRA target linear modules matched")
    return targets


def _save_adapter_clean(dst: pathlib.Path, model: Any) -> str:
    tmp = dst.parent / f"{dst.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tmp.rename(dst)
    return sha256_dir(dst)


def run_policy_training(cfg: PolicySFTConfig) -> dict[str, Any]:
    """Execute fail-closed Policy LoRA SFT with per-epoch checkpointing and optional fastPE."""
    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

    split_path = pathlib.Path(cfg.split_jsonl)
    manifest_path = pathlib.Path(cfg.manifest_json)
    split_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "train_jsonl_sha256" in split_meta and sha256_file(split_path) != split_meta["train_jsonl_sha256"]:
        raise ValueError("Train split SHA-256 mismatch against manifest")
    for check_key in ("dev_overlap_checks", "heldout_overlap_checks"):
        for metric, val in (split_meta.get(check_key) or {}).items():
            if int(val) != 0:
                raise ValueError(f"Non-zero leakage check {check_key}.{metric}={val}")

    train_rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_rows: list[dict[str, Any]] = []
    if cfg.dev_jsonl:
        dev_rows = [
            json.loads(line)
            for line in pathlib.Path(cfg.dev_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    run_dir = pathlib.Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    prompt_cfg = json.loads(pathlib.Path(cfg.prompt_config_json).read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        cfg.model_path, local_files_only=True, min_pixels=cfg.pixels, max_pixels=cfg.pixels
    )
    collate = Qwen3VLPolicyCollator(processor, prompt_cfg["prompt"])

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if cfg.fast_patch_embed:
        apply_fastpe(model, mode="gemm")

    target_modules = find_lm_lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad for n, p in model.named_parameters() if "visual" in n):
        raise RuntimeError("Vision tower parameters must stay frozen")

    micros_per_epoch = math.ceil(len(train_rows) / cfg.micro_batch_size)
    steps_per_epoch = math.ceil(micros_per_epoch / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    global_step = 0
    seen_samples = 0
    last_loss = None
    epoch_adapters: dict[str, Any] = {}
    dev_evals: list[dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        order = list(range(len(train_rows)))
        random.Random(cfg.seed + epoch).shuffle(order)
        micro_batches: list[Sequence[int]] = [
            order[i : i + cfg.micro_batch_size] for i in range(0, len(order), cfg.micro_batch_size)
        ]
        for g_start in range(0, len(micro_batches), cfg.grad_accum):
            group = micro_batches[g_start : g_start + cfg.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for mb in group:
                batch = collate([train_rows[idx] for idx in mb])
                batch_gpu = {k: v.to("cuda") for k, v in batch.items()}
                loss = model(**batch_gpu, use_cache=False).loss / len(group)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step={global_step}")
                loss.backward()
                group_loss += float(loss.detach().item())
                seen_samples += len(mb)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item())
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at step={global_step}")
            optimizer.step()
            scheduler.step()
            global_step += 1
            last_loss = group_loss

        if dev_rows:
            model.eval()
            tot_loss_tok, tot_tok = 0.0, 0
            with torch.inference_mode():
                for s in range(0, len(dev_rows), cfg.micro_batch_size):
                    b = collate(dev_rows[s : s + cfg.micro_batch_size])
                    n_tok = int((b["labels"] != -100).sum().item())
                    out = model(**{k: v.to("cuda") for k, v in b.items()}, use_cache=False)
                    tot_loss_tok += float(out.loss.item()) * n_tok
                    tot_tok += n_tok
            model.train()
            dev_evals.append(
                {"epoch": epoch + 1, "step": global_step, "dev_nll_loss": round(tot_loss_tok / max(1, tot_tok), 5)}
            )

        ep_dir = run_dir / f"adapter_epoch_{epoch + 1}"
        epoch_adapters[f"epoch_{epoch + 1}"] = {
            "path": str(ep_dir),
            "sha256": _save_adapter_clean(ep_dir, model),
        }

    final_dir = run_dir / "final_adapter"
    final_sha = _save_adapter_clean(final_dir, model)
    manifest = {
        "status": "COMPLETED",
        "optimizer_steps": global_step,
        "cumulative_samples_seen": seen_samples,
        "final_train_loss": round(last_loss, 5) if last_loss is not None else None,
        "dev_evals": dev_evals,
        "epoch_adapters": epoch_adapters,
        "final_adapter_path": str(final_dir),
        "final_adapter_sha256": final_sha,
        "gpu_seconds": round(time.time() - t0, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
    }
    (run_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train Qwen3-VL LM-only LoRA GUI policy.")
    ap.add_argument("--model-path", required=True, help="Local Qwen3-VL-8B-Instruct checkpoint directory.")
    ap.add_argument("--split", required=True, help="Path to train JSONL split.")
    ap.add_argument("--manifest", required=True, help="Path to split manifest with SHA256 and overlap assertions.")
    ap.add_argument("--prompt-config", required=True, help="JSON file containing 'prompt' template.")
    ap.add_argument("--output-dir", required=True, help="Directory for epoch checkpoints and train_manifest.json.")
    ap.add_argument("--dev-jsonl", default=None, help="Optional held-out dev JSONL for NLL monitoring only.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--no-fast-patch-embed", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = PolicySFTConfig(
        model_path=args.model_path,
        split_jsonl=args.split,
        manifest_json=args.manifest,
        prompt_config_json=args.prompt_config,
        output_dir=args.output_dir,
        dev_jsonl=args.dev_jsonl,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        fast_patch_embed=not args.no_fast_patch_embed,
    )
    print(json.dumps(run_policy_training(cfg), indent=2))


if __name__ == "__main__":
    main()
