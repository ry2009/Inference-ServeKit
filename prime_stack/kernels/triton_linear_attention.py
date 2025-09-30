from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def is_triton_available() -> bool:
    return triton is not None


@triton.jit  # type: ignore[misc]
def _kernel(q_ptr, k_ptr, v_ptr, out_ptr, norm_ptr, B, H, T, D, stride_qb, stride_qh, stride_qt, stride_qd,
            stride_kb, stride_kh, stride_kt, stride_kd, stride_vb, stride_vh, stride_vt, stride_vd,
            BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offset = batch_idx * H + head_idx

    q_base = q_ptr + offset * stride_qh
    k_base = k_ptr + offset * stride_kh
    v_base = v_ptr + offset * stride_vh
    out_base = out_ptr + offset * stride_qh
    norm_base = norm_ptr + offset * stride_qh

    kv_acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    norm_acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    t_offsets = tl.arange(0, BLOCK_T)
    d_offsets = tl.arange(0, BLOCK_D)

    for t in range(0, T, BLOCK_T):
        k_ptrs = k_base + (t + t_offsets)[:, None] * stride_kt + d_offsets[None, :] * stride_kd
        v_ptrs = v_base + (t + t_offsets)[:, None] * stride_vt + d_offsets[None, :] * stride_vd

        mask = (t + t_offsets)[:, None] < T
        k_chunk = tl.load(k_ptrs, mask=mask, other=0.0)
        v_chunk = tl.load(v_ptrs, mask=mask, other=0.0)

        k_chunk = tl.elu(k_chunk) + 1.0
        v_chunk = tl.elu(v_chunk) + 1.0

        kv_acc += tl.dot(k_chunk.to(tl.float32), v_chunk.to(tl.float32))
        norm_acc += tl.sum(k_chunk, axis=1)

    for t in range(0, T, BLOCK_T):
        q_ptrs = q_base + (t + t_offsets)[:, None] * stride_qt + d_offsets[None, :] * stride_qd
        mask = (t + t_offsets)[:, None] < T
        q_chunk = tl.load(q_ptrs, mask=mask, other=0.0)
        q_chunk = tl.elu(q_chunk) + 1.0

        out = tl.dot(q_chunk.to(tl.float32), kv_acc)
        norm = tl.dot(q_chunk.to(tl.float32), norm_acc)

        out = out / (norm[:, None] + 1e-6)
        tl.store(out_base + (t + t_offsets)[:, None] * stride_qt + d_offsets[None, :] * stride_qd,
                 out.to(q_chunk.dtype), mask=mask)
        tl.store(norm_base + t + t_offsets, norm, mask=t + t_offsets < T)


def triton_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if not is_triton_available():
        raise RuntimeError("Triton is required for triton_linear_attention")

    assert q.dim() == 4
    assert q.shape == k.shape == v.shape

    B, H, T, D = q.shape
    out = torch.empty_like(q)
    norm = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

    grid = (B, H)
    BLOCK_T = min(128, triton.next_power_of_2(T))
    BLOCK_D = min(64, triton.next_power_of_2(D))

    _kernel[grid](
        q,
        k,
        v,
        out,
        norm,
        B,
        H,
        T,
        D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )
    return out
