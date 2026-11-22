"""Single-base dual-LoRA HTTP server routing ``/predict`` (Policy) and ``/verify`` (Verifier)."""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence

from fastpe import apply_fastpe
from policy.collator import VERIFIER_LABELS, Qwen3VLPolicyCollator, build_verifier_messages
from protocols.qwen_computer_use_v2 import TOOL_SCHEMA


@dataclass(frozen=True)
class DualLoraRouterConfig:
    """Configuration for serving co-resident Policy and Verifier LoRA adapters."""

    model_path: str
    prompt_config_json: str
    policy_adapter: str = ""
    verifier_adapter: str = ""
    host: str = "127.0.0.1"
    port: int = 39850
    max_new_tokens: int = 128
    pixels: int = 131072
    fast_patch_embed: bool = True


def serve_dual_lora(cfg: DualLoraRouterConfig) -> None:
    """Start thread-safe HTTP server sharing one Qwen3-VL base model across Policy and Verifier LoRAs."""
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    prompt_cfg = json.loads(pathlib.Path(cfg.prompt_config_json).read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        cfg.model_path, local_files_only=True, min_pixels=cfg.pixels, max_pixels=cfg.pixels
    )
    collate = Qwen3VLPolicyCollator(processor, prompt_cfg["prompt"])
    abcd_ids = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in VERIFIER_LABELS]

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if cfg.fast_patch_embed:
        apply_fastpe(base, mode="gemm")

    pol_path = cfg.policy_adapter.strip() or None
    ver_path = cfg.verifier_adapter.strip() or None
    has_policy_adapter = pol_path is not None
    has_verifier_adapter = ver_path is not None

    if pol_path and ver_path:
        model: Any = PeftModel.from_pretrained(base, pol_path, adapter_name="policy", is_trainable=False)
        model.load_adapter(ver_path, adapter_name="verifier", is_trainable=False)
        model.set_adapter("policy")
    elif pol_path:
        model = PeftModel.from_pretrained(base, pol_path, adapter_name="policy", is_trainable=False)
        model.set_adapter("policy")
    elif ver_path:
        model = PeftModel.from_pretrained(base, ver_path, adapter_name="verifier", is_trainable=False)
        model.set_adapter("verifier")
    else:
        model = base
    model.eval()
    lock = threading.Lock()

    def predict_one(req: dict[str, Any]) -> dict[str, Any]:
        if has_policy_adapter:
            model.set_adapter("policy")
        sample = {
            "before_image": str(req["image_path"]),
            "instruction": req["instruction"],
            "visible_history": list(req.get("visible_history") or [])[-4:],
            "visible_feedback": str(req.get("visible_feedback") or ""),
        }
        im, msg = collate.build_messages(sample)
        pre = processor.apply_chat_template(msg, tools=[TOOL_SCHEMA], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[pre], images=[im], padding=False, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        temp = float(req.get("temperature", 0.0) or 0.0)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(req.get("max_new_tokens", cfg.max_new_tokens)),
            "use_cache": True,
            "do_sample": temp > 0.0,
        }
        if temp > 0.0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9

        t0 = time.perf_counter()
        with torch.inference_mode():
            if not has_policy_adapter and has_verifier_adapter:
                with model.disable_adapter():
                    out_ids = model.generate(**inputs, **gen_kwargs)
            else:
                out_ids = model.generate(**inputs, **gen_kwargs)
        lat = time.perf_counter() - t0
        gen_ids = out_ids[0, prompt_len:]
        raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
        if raw.endswith("<|endoftext|>"):
            raw = raw[: -len("<|endoftext|>")]
        return {"raw": raw, "latency_s": round(lat, 4), "prompt_tokens": prompt_len, "gen_tokens": int(gen_ids.shape[0])}

    def verify_one(req: dict[str, Any]) -> dict[str, Any]:
        if not has_verifier_adapter:
            raise RuntimeError("No verifier adapter loaded")
        model.set_adapter("verifier")
        msg = build_verifier_messages(
            req["before_image"], req["after_image"], req["instruction"], req["action"], req.get("visible_history") or []
        )
        txt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        with Image.open(req["before_image"]) as ib, Image.open(req["after_image"]) as ia:
            inputs = processor(text=[txt], images=[ib.convert("RGB"), ia.convert("RGB")], padding=False, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1, abcd_ids].float()
            probs = logits.softmax(dim=-1)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        p_list = [round(float(x), 5) for x in probs.cpu().tolist()]
        best_i = int(torch.argmax(probs).item())
        return {
            "pred_label": VERIFIER_LABELS[best_i],
            "probs": dict(zip(VERIFIER_LABELS, p_list)),
            "latency_ms": round(lat_ms, 2),
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(
                {
                    "ok": True,
                    "has_policy_adapter": has_policy_adapter,
                    "has_verifier_adapter": has_verifier_adapter,
                    "fast_patch_embed": cfg.fast_patch_embed,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path not in ("/predict", "/verify"):
                self.send_response(404)
                self.end_headers()
                return
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            with lock:
                res = predict_one(req) if self.path == "/predict" else verify_one(req)
            body = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(json.dumps({"status": "READY", "host": cfg.host, "port": cfg.port}), flush=True)
    ThreadingHTTPServer((cfg.host, cfg.port), Handler).serve_forever()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Serve dual-LoRA Policy and Verifier adapters on one base model.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt-config", required=True)
    ap.add_argument("--policy-adapter", default="")
    ap.add_argument("--verifier-adapter", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39850)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--no-fast-patch-embed", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    serve_dual_lora(
        DualLoraRouterConfig(
            model_path=args.model_path,
            prompt_config_json=args.prompt_config,
            policy_adapter=args.policy_adapter,
            verifier_adapter=args.verifier_adapter,
            host=args.host,
            port=args.port,
            max_new_tokens=args.max_new_tokens,
            fast_patch_embed=not args.no_fast_patch_embed,
        )
    )


if __name__ == "__main__":
    main()
