"""Local LoRA SFT with explicit target-only loss and reproducible evidence.

This uses PyTorch + Transformers + PEFT directly: no TRL version dependency,
external API calls, hidden chain-of-thought targets or automatic Hub uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from training.data import digest, read_jsonl, validate_splits
from training.modeling import encode_decision, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3-0.6B")
    parser.add_argument("--train", default="eval/executor/train.jsonl")
    parser.add_argument("--dev", default="eval/executor/dev.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Train LoRA over an NF4 double-quantized base with BF16 compute",
    )
    parser.add_argument(
        "--trim-cuda-cache",
        action="store_true",
        help="Release unused CUDA allocator blocks after optimizer steps on memory-constrained Windows GPUs",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; choose a new run directory")
    if min(args.epochs, args.rank, args.gradient_accumulation, args.max_length) < 1:
        raise ValueError("Training integer parameters must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This training configuration requires CUDA")
    rows, dev_rows = read_jsonl(args.train), read_jsonl(args.dev)
    if any(r["split"] != "train" for r in rows) or any(
        r["split"] != "dev" for r in dev_rows
    ):
        raise ValueError("Training may only read train/dev, never held-out test")
    validate_splits(rows + dev_rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows or not dev_rows:
        raise ValueError("Empty train/dev split")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    started = time.perf_counter()
    model, tokenizer = load_model(args.model, quantized=args.qlora)
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    encoded = [encode_decision(tokenizer, row, args.max_length) for row in rows]
    dev_encoded = [encode_decision(tokenizer, row, args.max_length) for row in dev_rows]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    args.output.mkdir(parents=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update(
        {
            "train_sha256": digest(rows),
            "dev_sha256": digest(dev_rows),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": model.get_nb_trainable_parameters()[1],
            "stored_parameter_elements": sum(p.numel() for p in model.parameters()),
            "sequence_lengths": [len(e["input_ids"]) for e in encoded],
            "target_tokens": [e["target_tokens"] for e in encoded],
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True)
            ),
            "loss_mask": "all prompt tokens -100; only final assistant target supervised",
            "source_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path("training").glob("*.py"))
            },
            "model_config_sha256": hashlib.sha256(
                (Path(args.model) / "config.json").read_bytes()
            ).hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(
                (Path(args.model) / "tokenizer_config.json").read_bytes()
            ).hexdigest(),
        }
    )
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "ready",
                "samples": len(encoded),
                "max_tokens": max(len(e["input_ids"]) for e in encoded),
                "trainable_parameters": config["trainable_parameters"],
            }
        ),
        flush=True,
    )

    def forward(sample):
        # Qwen's logits_to_keep avoids materializing vocabulary logits for the
        # masked prompt. Include the preceding position to predict target[0].
        target_count = sample["target_tokens"]
        output = model(
            input_ids=torch.tensor([sample["input_ids"]], device="cuda"),
            attention_mask=torch.ones(
                (1, len(sample["input_ids"])), device="cuda", dtype=torch.long
            ),
            logits_to_keep=target_count + 1,
        )
        logits = output.logits[:, :-1, :].float()
        targets = torch.tensor([sample["labels"][-target_count:]], device="cuda")
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

    def evaluate_loss():
        model.eval()
        numerator = denominator = 0
        with torch.no_grad():
            for sample in dev_encoded:
                loss = forward(sample)
                numerator += float(loss) * sample["target_tokens"]
                denominator += sample["target_tokens"]
        return numerator / denominator

    baseline_loss = evaluate_loss()
    best_loss = math.inf
    logs = [{"epoch": 0, "dev_loss": baseline_loss}]
    for epoch in range(args.epochs):
        model.train()
        indices = list(range(len(encoded)))
        random.shuffle(indices)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(indices):
            group_start = (
                position // args.gradient_accumulation * args.gradient_accumulation
            )
            group_size = min(args.gradient_accumulation, len(indices) - group_start)
            loss = forward(encoded[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            running += float(loss.detach())
            (loss / group_size).backward()
            if (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(
                indices
            ):
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if args.trim_cuda_cache:
                    torch.cuda.empty_cache()
            if (position + 1) % 16 == 0 or position + 1 == len(indices):
                progress = {
                    "epoch": epoch + 1,
                    "sample": position + 1,
                    "loss": running / (position + 1),
                    "elapsed_seconds": time.perf_counter() - started,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                }
                with (args.output / "progress.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
        dev_loss = evaluate_loss()
        entry = {
            "epoch": epoch + 1,
            "train_loss": running / len(indices),
            "dev_loss": dev_loss,
        }
        logs.append(entry)
        if dev_loss < best_loss:
            best_loss = dev_loss
            model.save_pretrained(args.output / "adapter")
            tokenizer.save_pretrained(args.output / "adapter")
        (args.output / "metrics.json").write_text(
            json.dumps(
                {
                    "epochs": logs,
                    "selected_dev_loss": best_loss,
                    "duration_seconds": time.perf_counter() - started,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(entry), flush=True)
    print(
        f"Saved best development-loss adapter to {args.output / 'adapter'}", flush=True
    )


if __name__ == "__main__":
    main()
