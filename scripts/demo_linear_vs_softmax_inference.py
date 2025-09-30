#!/usr/bin/env python3
"""Showcase throughput impact of linear attention in an autoregressive decode loop."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from prime_stack.kernels.linear_attention import linear_attention


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def run_decode(length: int, layers: int, dim: int, heads: int, backend: str, device: str, steps: int) -> float:
    torch.manual_seed(0)
    dtype = torch.float32 if device == "cpu" else torch.float16
    head_dim = dim // heads

    q_proj = [torch.randn(dim, dim * 3, device=device, dtype=dtype) / (dim ** 0.5) for _ in range(layers)]
    out_proj = [torch.randn(dim, dim, device=device, dtype=dtype) / (dim ** 0.5) for _ in range(layers)]

    tokens = torch.randn(1, length, dim, device=device, dtype=dtype)

    times = []
    for step in range(steps):
        x = tokens.clone()
        start = time.perf_counter()
        for layer in range(layers):
            qkv = x @ q_proj[layer]
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q.view(1, length, heads, head_dim).transpose(1, 2)
            k = k.view(1, length, heads, head_dim).transpose(1, 2)
            v = v.view(1, length, heads, head_dim).transpose(1, 2)

            if backend == "linear":
                out = linear_attention(q, k, v)
            else:
                out = flash_attention(q, k, v)

            out = out.transpose(1, 2).contiguous().view(1, length, dim)
            x = out @ out_proj[layer]

        torch.cuda.synchronize() if device.startswith("cuda") else None
        elapsed = (time.perf_counter() - start) * 1e3
        times.append(elapsed)
    return statistics.mean(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    softmax_ms = run_decode(args.length, args.layers, args.dim, args.heads, "softmax", device, args.steps)
    linear_ms = run_decode(args.length, args.layers, args.dim, args.heads, "linear", device, args.steps)

    tokens_processed = args.length * args.layers
    print(f"Decode loop (length={args.length}, layers={args.layers}, dim={args.dim}, device={device})")
    print(f"  Softmax attention  : {softmax_ms:.2f} ms per forward")
    print(f"  Linear attention   : {linear_ms:.2f} ms per forward")
    print(f"  Speedup            : {softmax_ms / linear_ms:.2f}×")
    print(f"  Tokens per second  : softmax={tokens_processed / (softmax_ms/1e3):.1f}, linear={tokens_processed / (linear_ms/1e3):.1f}")


if __name__ == "__main__":
    main()
