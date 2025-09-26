def eviction_cost(hbm_bytes: int, hit_rate: float, age_s: float, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1e-3) -> float:
    """Simple cost heuristic for deciding which prefix entries to evict."""
    return alpha * hbm_bytes + beta / (hit_rate + 1e-3) + gamma * age_s
