"""Loopback-only OpenAI-compatible executor for local Agent integration.

Supports non-streaming text/tool chat completions with Qwen templates. This is a
single-GPU development server, not a production deployment or paid API proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from training.modeling import load_model, parse_output, render_prompt


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[dict]
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    parallel_tool_calls: bool | None = None
    n: int = 1
    # LangChain can supply stream usage preferences even for nonstream requests.
    stream_options: dict | None = None


def create_app(
    model_path: str, adapter: str | None, model_name: str, context_window: int = 4096
) -> FastAPI:
    import torch

    state = {}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        torch.set_num_threads(4)
        model, tokenizer = await asyncio.to_thread(load_model, model_path, adapter)
        state.update(model=model, tokenizer=tokenizer)
        yield
        state.clear()

    app = FastAPI(title="Local executor LoRA development server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "ready": bool(state),
            "model": model_name,
            "adapter_loaded": bool(adapter),
        }

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "local"}],
        }

    def generate(request: ChatRequest) -> dict:
        model, tokenizer = state["model"], state["tokenizer"]
        tools = [] if request.tool_choice == "none" else request.tools
        prompt = render_prompt(tokenizer, request.messages, tools)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        length = inputs["input_ids"].shape[1]
        max_new = request.max_completion_tokens or request.max_tokens or 256
        if length + max_new > context_window:
            raise HTTPException(400, "context_length_exceeded")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        tokens = outputs[0, length:].tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        try:
            message = parse_output(raw)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                502, f"Invalid executor tool output: {type(exc).__name__}"
            ) from exc
        truncated = bool(
            tokens and tokens[-1] != tokenizer.eos_token_id and len(tokens) == max_new
        )
        reason = (
            "length"
            if truncated
            else ("tool_calls" if message.get("tool_calls") else "stop")
        )
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": {
                "prompt_tokens": length,
                "completion_tokens": len(tokens),
                "total_tokens": length + len(tokens),
            },
        }

    @app.post("/v1/chat/completions")
    async def completions(request: ChatRequest) -> dict:
        if request.model != model_name:
            raise HTTPException(404, "Unknown executor model version")
        if request.stream or request.n != 1:
            raise HTTPException(400, "Only non-streaming n=1 completions are supported")
        if request.tool_choice not in (None, "auto", "none"):
            raise HTTPException(400, "Only auto/none tool choice is supported")
        if not request.messages or any(
            m.get("role") not in {"system", "user", "assistant", "tool"}
            or (m.get("content") is not None and not isinstance(m["content"], str))
            for m in request.messages
        ):
            raise HTTPException(400, "Only nonempty text chat messages are supported")
        # This comparison endpoint is intentionally greedy, regardless of caller
        # temperature; expose that explicitly in the server documentation.
        async with lock:
            result = await asyncio.to_thread(generate, request)
            if request.tool_choice == "none" and result["choices"][0]["message"].get(
                "tool_calls"
            ):
                raise HTTPException(502, "Executor violated tool_choice=none")
            return result

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3-0.6B")
    parser.add_argument("--adapter")
    parser.add_argument("--name", default="globex-executor-base")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--context-window", type=int, default=4096)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "name": args.name,
                "decoding": "greedy",
                "host": "127.0.0.1",
            }
        ),
        flush=True,
    )
    uvicorn.run(
        create_app(args.model, args.adapter, args.name, args.context_window),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
