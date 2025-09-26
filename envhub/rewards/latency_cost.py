def latency_penalty(ms: int, coef: float = 0.001) -> float:
    return -coef * ms
