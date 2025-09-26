import numpy as np


def group_relative(scores):
    """Compute relative advantages within a GRPO sample group."""
    arr = np.array(scores, dtype=np.float32)
    return list(arr - arr.mean())
