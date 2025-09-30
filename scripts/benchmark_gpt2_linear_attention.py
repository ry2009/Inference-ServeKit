#!/usr/bin/env python3
"""Benchmark GPT-2 forward latency with softmax vs. linear attention."""

from __future__ import annotations

import argparse
import time
import types
import statistics

import torch
from transformers import GPT2LMHeadModel

from prime_stack.kernels.linear_attention import causal_linear_attention


def linear_attention_forward(self, hidden_states, layer_past=None, attention_mask=None,
                             head_mask=None, use_cache=False, output_attentions=False,
                             past_key_value=None, cache_position=None, **kwargs):
    query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    bsz, seq_len, _ = query.size()
    query = query.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    key = key.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    value = value.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    scale_attn = getattr(self, "scale_attn", False)
    if scale_attn:
        scale_attn_factor = getattr(self, "scale_attn_factor", 1.0)
        inv_norm = getattr(self, "inv_norm_factor", 1.0)
        query = query * scale_attn_factor
        key = key * inv_norm

    attn_output = causal_linear_attention(query, key, value)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
    attn_output = self.c_proj(attn_output)

    outputs = (attn_output, None)
    if output_attentions:
        outputs += (None,)
    return outputs


def restore_attention(model, originals):
    for block, orig in zip(model.transformer.h, originals):
        block.attn.forward = orig


def apply_linear_attention(model):
    originals = []
    for block in model.transformer.h:
        originals.append(block.attn.forward)
        block.attn.forward = types.MethodType(linear_attention_forward, block.attn)
    return originals


def measure(model, input_ids, iters=5):
    timings = []
    with torch.no_grad():
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(input_ids, use_cache=False)
            elapsed = (time.perf_counter() - start) * 1e3
            timings.append(elapsed)
    return statistics.mean(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    model.eval()

    max_len = model.config.n_ctx
    if args.seq > max_len:
        raise ValueError(f"Sequence length {args.seq} exceeds model maximum {max_len}")

    input_ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq), device=device)

    softmax_ms = measure(model, input_ids, args.iters)
    tokens = args.batch * args.seq

    originals = apply_linear_attention(model)
    try:
        linear_ms = measure(model, input_ids, args.iters)
    finally:
        restore_attention(model, originals)

    print(f"Model: {args.model} | device={device} | batch={args.batch} | seq={args.seq}")
    print(f"Softmax attention : {softmax_ms:.2f} ms per forward | tokens/sec={tokens/(softmax_ms/1e3):.1f}")
    print(f"Linear attention  : {linear_ms:.2f} ms per forward | tokens/sec={tokens/(linear_ms/1e3):.1f}")
    print(f"Speedup           : {softmax_ms / linear_ms:.2f}x")


if __name__ == "__main__":
    main()
