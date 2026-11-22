"""Chat-template collators and prompt builders for Qwen3-VL Policy SFT and 4-class Semantic Verifier."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from protocols.qwen_computer_use_v2 import TOOL_SCHEMA

VERIFIER_LABELS: tuple[str, str, str, str] = ("A", "B", "C", "D")

VERIFIER_SYSTEM_PROMPT: str = "\n".join(
    (
        "Classify the GUI transition progress given the user goal, recent actions, "
        "before screenshot, executed action, and after screenshot.",
        "Options:",
        "A: effective_progress (action moved the task forward toward the stated goal)",
        "B: no_progress_or_stagnation (no meaningful state progress or repeated ineffective action)",
        "C: semantic_error_or_regression (wrong target/field, premature app switch abandoning an "
        "unsubmitted form, or redundant action after completion)",
        "D: insufficient_evidence_or_observe (ambiguous change or loading state requiring further observation)",
        "Answer with a single letter A, B, C, or D.",
    )
)


class Qwen3VLPolicyCollator:
    """Single-screenshot SFT collator with fail-closed prefix/label alignment."""

    def __init__(self, processor: Any, prompt_template: str) -> None:
        self.processor = processor
        self.prompt_template = prompt_template

    def build_messages(
        self, example: Mapping[str, Any]
    ) -> tuple[Any, list[dict[str, Any]]]:
        from PIL import Image

        with Image.open(example["before_image"]) as raw_im:
            image = raw_im.convert("RGB")
        width, height = image.size
        history = list(example.get("visible_history") or [])[-4:]
        text = (
            self.prompt_template.replace("{width}", str(width)).replace("{height}", str(height))
            + "\nTask: "
            + str(example["instruction"])
            + "\nRecent actions: "
            + json.dumps(history, ensure_ascii=False)
        )
        feedback = str(example.get("visible_feedback") or "").strip()
        if feedback:
            text += f"\nNote: {feedback}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(example["before_image"])},
                    {"type": "text", "text": text},
                ],
            }
        ]
        return image, messages

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        images = []
        full_texts = []
        prefixes = []
        for ex in examples:
            im, msg = self.build_messages(ex)
            images.append(im)
            pre = self.processor.apply_chat_template(
                msg, tools=[TOOL_SCHEMA], tokenize=False, add_generation_prompt=True
            )
            assistant = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "computer_use",
                            "arguments": ex["tool_call_v2"],
                        },
                    }
                ],
            }
            full = self.processor.apply_chat_template(
                msg + [assistant], tools=[TOOL_SCHEMA], tokenize=False, add_generation_prompt=False
            )
            if not full.startswith(pre):
                raise ValueError("SFT chat template is not inference-prefix compatible")
            prefixes.append(pre)
            full_texts.append(full)

        batch = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = torch.full_like(batch["input_ids"], -100)
        for i, (pre, im) in enumerate(zip(prefixes, images)):
            pre_ids = self.processor(
                text=[pre], images=[im], padding=False, return_tensors="pt"
            )["input_ids"][0]
            mask = batch["attention_mask"][i].bool()
            idx = torch.where(mask)[0]
            ids = batch["input_ids"][i, idx]
            if len(pre_ids) >= len(ids) or not torch.equal(ids[: len(pre_ids)], pre_ids):
                raise ValueError("token-level inference prefix mismatch")
            labels[i, idx[len(pre_ids) :]] = ids[len(pre_ids) :]
            if not (labels[i] != -100).any():
                raise ValueError("no supervised target tokens remaining after prompt masking")
        batch["labels"] = labels
        return batch


def build_verifier_messages(
    before_image: str,
    after_image: str,
    goal: str,
    action: Mapping[str, Any] | None,
    visible_history: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build dual-screenshot transition classification prompt for single-token A/B/C/D scoring."""
    hist_txt = json.dumps(list(visible_history or [])[-4:], ensure_ascii=False)
    act_txt = json.dumps(dict(action or {}), ensure_ascii=False, sort_keys=True)
    header = (
        f"{VERIFIER_SYSTEM_PROMPT}\n"
        f"Goal: {goal}\n"
        f"Recent actions: {hist_txt}\n"
        f"Executed action: {act_txt}\n"
        f"Before image:"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": header},
                {"type": "image", "image": str(before_image)},
                {"type": "text", "text": "After image:"},
                {"type": "image", "image": str(after_image)},
                {"type": "text", "text": "Label:"},
            ],
        }
    ]


def collate_verifier_batch(
    processor: Any, chunk: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Collate dual-image verifier supervision with target class indices in ``{0,1,2,3}``."""
    import torch
    from PIL import Image

    texts, images, targets = [], [], []
    for row in chunk:
        msg = build_verifier_messages(
            row["before_image"],
            row["after_image"],
            row["instruction"],
            row["action"],
            row.get("visible_history") or [],
        )
        texts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        with Image.open(row["before_image"]) as ib, Image.open(row["after_image"]) as ia:
            images.append(ib.convert("RGB"))
            images.append(ia.convert("RGB"))
        targets.append(VERIFIER_LABELS.index(row["label"]))
    processor.tokenizer.padding_side = "left"
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    batch["target_idx"] = torch.tensor(targets, dtype=torch.long)
    return batch
