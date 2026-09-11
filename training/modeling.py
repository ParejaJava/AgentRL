"""Shared Qwen chat serialization for training, evaluation and local serving."""

from __future__ import annotations

import copy
import json
import re
from typing import Any
from uuid import uuid4


def normalize_messages(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for message in result:
        message["content"] = message.get("content") or ""
        for call in message.get("tool_calls", []):
            args = call["function"]["arguments"]
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
    return result


def render_prompt(tokenizer: Any, messages: list[dict], tools: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        normalize_messages(messages),
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_decision(tokenizer: Any, row: dict, max_length: int) -> dict:
    """Mask the entire prompt and reject truncation or unstable template prefixes."""
    prompt = render_prompt(tokenizer, row["messages"], row["tools"])
    full = tokenizer.apply_chat_template(
        normalize_messages([*row["messages"], row["target"]]),
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full.startswith(prompt):
        raise ValueError(f"Chat template prefix mismatch: {row['case_id']}")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Token boundary merges across supervised target")
    if len(ids) > max_length:
        raise ValueError(
            f"Sample {row['case_id']} has {len(ids)} tokens > {max_length}; no silent truncation"
        )
    if len(ids) <= len(prompt_ids):
        raise ValueError("No supervised tokens")
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
    return {
        "input_ids": ids,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(ids) - len(prompt_ids),
    }


def parse_output(text: str) -> dict:
    """Parse Qwen's explicit tool-call tags; malformed calls are errors, not text."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if text.count("<tool_call>") != text.count("</tool_call>"):
        raise ValueError("Unclosed tool call")
    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        value = json.loads(body.strip())
        json.dumps(value, allow_nan=False)
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise TypeError("Invalid tool name")
        if not isinstance(value.get("arguments"), dict):
            raise TypeError("Invalid tool argument object")
        calls.append(
            {
                "id": f"call_{uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": value["name"],
                    "arguments": json.dumps(value["arguments"], ensure_ascii=False),
                },
            }
        )
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    result = {"role": "assistant", "content": content}
    if calls:
        result["tool_calls"] = calls
    return result


def load_model(model_path: str, adapter: str | None = None, *, quantized: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    options = {}
    if quantized:
        from transformers import BitsAndBytesConfig

        options = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": {"": 0},
        }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **options,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval() if quantized else model.to(device).eval()
    return model, tokenizer
