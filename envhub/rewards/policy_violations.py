def policy_violation_penalty(event: str) -> float:
    """Map policy violation types to negative rewards."""
    table = {
        "unsafe_content": -1.0,
        "latency_miss": -0.5,
        "tool_error": -0.3,
    }
    return table.get(event, 0.0)
