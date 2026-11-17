"""Latency and numerical-parity harness for :class:`FastPatchEmbed`.

Typical use::

    python -m fastpe.benchmark_parity --patches 4096 --dim 1152 --dtype float32
    python -m fastpe.benchmark_parity --device cuda --dtype bfloat16 --json out.json

The harness reports three things that must be read together:

1. **Latency** - p50 / p90 / p99 over steady-state iterations after warmup.
   The speedup number is only meaningful relative to the same device, dtype,
   patch geometry and batch size.
2. **Numerical drift** - max absolute, mean absolute and relative L2
   difference of the pooled embedding and of the downstream logits.
3. **Decision parity** - argmax flip rate, plus flip rate inside the high-risk
   band. A fast path that is fast but drops high-risk alerts is a regression,
   not an optimization.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .fast_patch_embed import (
    FastPatchEmbed,
    ParityBudget,
    audit_parity,
    percentile,
)

__all__ = ["build_pair", "measure", "run", "main"]


class _ReferencePatchEmbed(nn.Module):
    """Minimal stand-in for the upstream Qwen-VL patch-embed module."""

    def __init__(
        self,
        in_channels: int = 3,
        temporal_patch_size: int = 2,
        patch_size: int = 14,
        embed_dim: int = 1152,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel, stride=kernel, bias=bias).to(dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        patches = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(patches).view(-1, self.embed_dim)


def build_pair(
    *,
    patches: int,
    dim: int,
    temporal: int,
    patch: int,
    dtype: torch.dtype,
    device: torch.device,
    embed_dim: int = 1152,
    seed: int = 20261116,
    bias: bool = True,
) -> tuple[FastPatchEmbed, torch.Tensor, nn.Linear]:
    """Create a reference module, its fast wrapper, an input batch and a head."""
    torch.manual_seed(seed)
    reference = _ReferencePatchEmbed(
        in_channels=dim,
        temporal_patch_size=temporal,
        patch_size=patch,
        embed_dim=embed_dim,
        bias=bias,
        dtype=dtype,
    ).to(device)
    wrapper = FastPatchEmbed(reference, mode="gemm").to(device)

    flat = dim * temporal * patch * patch
    hidden = torch.randn(patches, flat, device=device, dtype=dtype)

    head = nn.Linear(embed_dim, 5, bias=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        head.weight.normal_(0.0, 0.02)
    return wrapper, hidden, head


def measure(
    wrapper: FastPatchEmbed,
    hidden: torch.Tensor,
    head: nn.Linear,
    *,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, object]:
    """Time both modes and compare their logits.

    Each mode runs the full patch-embed plus a fixed linear head so that the
    measured span covers the operation a verifier actually performs.
    """

    def step(mode: str) -> torch.Tensor:
        wrapper.set_mode(mode)
        embedding = wrapper(hidden)
        pooled = embedding.mean(dim=0, keepdim=True)
        return head(pooled)

    timings: dict[str, list[float]] = {"original": [], "gemm": []}
    logits: dict[str, torch.Tensor] = {}

    for mode in ("original", "gemm"):
        for _ in range(warmup):
            step(mode)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(iterations):
            start = time.perf_counter()
            out = step(mode)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[mode].append((time.perf_counter() - start) * 1000.0)
        logits[mode] = out.detach().float().cpu()

    embedding_parity = None
    wrapper.set_mode("gemm")
    fast_embedding = wrapper(hidden).detach().float().cpu()
    wrapper.set_mode("original")
    ref_embedding = wrapper(hidden).detach().float().cpu()
    diff = (fast_embedding - ref_embedding).abs()
    embedding_parity = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l2": float(
            torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(ref_embedding) + 1e-12)
        ),
    }

    summary = {}
    for mode in ("original", "gemm"):
        values = timings[mode]
        summary[mode] = {
            "p50_ms": percentile(values, 0.50),
            "p90_ms": percentile(values, 0.90),
            "p99_ms": percentile(values, 0.99),
            "mean_ms": statistics.fmean(values),
            "min_ms": min(values),
        }
    summary["speedup_p50"] = (
        summary["original"]["p50_ms"] / summary["gemm"]["p50_ms"]
        if summary["gemm"]["p50_ms"]
        else float("inf")
    )

    return {
        "latency": summary,
        "embedding_parity": embedding_parity,
        "logits": {"original": logits["original"], "gemm": logits["gemm"]},
    }


@dataclass(frozen=True)
class BenchmarkConfig:
    patches: int = 4096
    dim: int = 3
    temporal: int = 2
    patch: int = 14
    embed_dim: int = 1152
    warmup: int = 10
    iterations: int = 50
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 20261116


def run(config: BenchmarkConfig) -> dict[str, object]:
    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    wrapper, hidden, head = build_pair(
        patches=config.patches,
        dim=config.dim,
        temporal=config.temporal,
        patch=config.patch,
        dtype=dtype,
        device=device,
        embed_dim=config.embed_dim,
        seed=config.seed,
    )

    measured = measure(
        wrapper,
        hidden,
        head,
        device=device,
        warmup=config.warmup,
        iterations=config.iterations,
    )

    # A deterministic pseudo-risk score keeps the audit path exercised without
    # pretending to be a real classifier. Real deployments pass the verifier's
    # hijack probability here.
    ref_logits = measured["logits"]["original"]
    cand_logits = measured["logits"]["gemm"]
    risk = torch.softmax(ref_logits, dim=1)[:, 0].tolist()
    report = audit_parity(
        ref_logits,
        cand_logits,
        reference_risk=risk,
        risk_threshold=0.5,
        budget=ParityBudget(),
    )

    result = {
        "config": {
            "patches": config.patches,
            "in_channels": config.dim,
            "temporal_patch_size": config.temporal,
            "patch_size": config.patch,
            "warmup": config.warmup,
            "iterations": config.iterations,
            "dtype": config.dtype,
            "device": config.device,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "latency": measured["latency"],
        "embedding_parity": measured["embedding_parity"],
        "parity_audit": report.as_dict(),
    }
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=int, default=4096, help="number of patches per batch")
    parser.add_argument("--in-channels", type=int, default=3, dest="dim")
    parser.add_argument("--temporal", type=int, default=2)
    parser.add_argument("--patch", type=int, default=14)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "bfloat16", "float16"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", dest="json_path", default=None, help="write the report as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = BenchmarkConfig(
        patches=args.patches,
        dim=args.dim,
        temporal=args.temporal,
        patch=args.patch,
        warmup=args.warmup,
        iterations=args.iterations,
        dtype=args.dtype,
        device=args.device,
    )
    result = run(config)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0 if result["parity_audit"]["verdict"] == "accept" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
