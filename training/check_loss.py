"""Prove target-sliced loss/gradients match full masked causal loss on tiny Qwen."""

import json

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    torch.manual_seed(11)
    torch.set_num_threads(2)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        attention_dropout=0.0,
    )
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    labels = ids.clone()
    labels[:, :4] = -100
    full_loss = model(input_ids=ids, labels=labels).loss
    full_loss.backward()
    full_grad = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, logits_to_keep=4).logits[:, :-1].float()
    sliced_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 64), ids[:, -3:].reshape(-1)
    )
    sliced_loss.backward()
    torch.testing.assert_close(full_loss, sliced_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        full_grad, model.lm_head.weight.grad, atol=1e-6, rtol=1e-6
    )
    print(
        json.dumps(
            {
                "passed": True,
                "full_loss": float(full_loss.detach()),
                "sliced_loss": float(sliced_loss.detach()),
                "gradient_max_error": float(
                    (full_grad - model.lm_head.weight.grad).abs().max()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
