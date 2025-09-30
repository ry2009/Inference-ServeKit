#!/usr/bin/env python3
"""Benchmark standard vs. linear attention kernels on the current machine."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from prime_stack.kernels import linear_attention, triton_linear_attention, is_triton_available


def attention_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def benchmark(seq_len: int, dim: int, heads: int, warmup: int = 5, iters: int = 20, device: str = "cuda", backend: str = "pytorch") -> dict[str, float]:
    torch.manual_seed(0)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    dtype = torch.float16 if device == "cuda" else torch.float32
    q = torch.randn(1, heads, seq_len, dim // heads, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def run(fn):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        _ = out[0, 0, 0, 0].item()
        return (time.perf_counter() - start) * 1e3

    # Warmup baseline
    for _ in range(warmup):
        _ = attention_baseline(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()

    baseline = [run(attention_baseline) for _ in range(iters)]

    # Warmup linear
    for _ in range(warmup):
        _ = linear_attention(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()

    if backend == "triton":
        if not is_triton_available():
            raise RuntimeError("Triton backend requested but triton is not installed")

        def triton_fn(q, k, v):
            return triton_linear_attention(q, k, v)

        linear_times = [run(triton_fn) for _ in range(iters)]
    else:
        linear_times = [run(linear_attention) for _ in range(iters)]

    return {
        "seq": seq_len,
        "baseline_ms": statistics.mean(baseline),
        "linear_ms": statistics.mean(linear_times),
        "speedup": statistics.mean(baseline) / statistics.mean(linear_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lengths", nargs="*", type=int, default=[256, 512, 1024])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--backend", choices=["pytorch", "triton"], default="pytorch")
    args = parser.parse_args()

    results = []
    for seq in args.lengths:
        stats = benchmark(seq, args.dim, args.heads, iters=args.iters, device=args.device, backend=args.backend)
        results.append(stats)
        print(
            f"seq={seq:<4} baseline={stats['baseline_ms']:.3f} ms "
            f"linear={stats['linear_ms']:.3f} ms speedup={stats['speedup']:.2f}x"
        )

    avg_speedup = statistics.mean(r["speedup"] for r in results)
    print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
