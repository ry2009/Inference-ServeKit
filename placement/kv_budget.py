def kv_bytes(
    seq_len: int,
    layers: int,
    heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    batch: int = 1,
) -> int:
    """Estimate KV cache bytes for a given transformer configuration."""
    per_pos = heads * head_dim * 2 * dtype_bytes
    return layers * seq_len * per_pos * batch
