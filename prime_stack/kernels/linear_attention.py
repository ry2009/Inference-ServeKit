"""Linear attention kernels and helpers.

These implementations follow the kernel-trick formulation (ELU+1) to achieve
O(T) complexity versus the standard quadratic attention. They operate on
[batch, heads, seq, head_dim] tensors and assume CUDA tensors for peak speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kernel_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Apply the ELU+1 feature map to ensure non-negativity (favoring kernel trick)."""
    return F.elu(x) + 1.0


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute linear attention with O(T) complexity.

    Args:
        q: Query tensor [batch, heads, seq, head_dim]
        k: Key tensor [batch, heads, seq, head_dim]
        v: Value tensor [batch, heads, seq, head_dim]
    Returns:
        Tensor [batch, heads, seq, head_dim]
    """

    q_prime = kernel_feature_map(q)
    k_prime = kernel_feature_map(k)

    # Compute K^T V once: [batch, heads, head_dim, head_dim]
    kv = torch.einsum("bhnd,bhne->bhde", k_prime, v)

    # Apply to queries: [batch, heads, seq, head_dim]
    out = torch.einsum("bhnd,bhde->bhne", q_prime, kv)

    # Normalizer: [batch, heads, seq]
    z = torch.einsum("bhnd,bhd->bhn", q_prime, k_prime.sum(dim=2))
    out = out / (z.unsqueeze(-1) + 1e-6)
    return out


@torch.inference_mode()
def linear_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return linear_attention(q, k, v)


def causal_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal linear attention using cumulative sums (O(T))."""

    q_prime = kernel_feature_map(q)
    k_prime = kernel_feature_map(k)

    # Compute cumulative sums for numerator and denominator
    kv = torch.einsum("bhnd,bhne->bhned", k_prime, v)
    kv_cumsum = kv.cumsum(dim=2)
    denom = k_prime.cumsum(dim=2)

    numer = torch.einsum("bhnd,bhned->bhne", q_prime, kv_cumsum)
    denom_term = torch.einsum("bhnd,bhnd->bhn", q_prime, denom)
    out = numer / (denom_term.unsqueeze(-1) + 1e-6)
    return out
