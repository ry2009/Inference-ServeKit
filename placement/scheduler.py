from __future__ import annotations

from typing import List, Optional


class Scheduler:
    """Placement scheduler scoring candidates based on free HBM and queue penalty."""

    def score_node(self, node: object, warm: bool, kv_required: int, slo: int) -> Optional[float]:
        free_hbm = getattr(node, "free_hbm", 0)
        if free_hbm <= kv_required * 1.1:
            return None
        link_bw = getattr(node, "link_bw", 0.0)
        queue_penalty = getattr(node, "queue_penalty", 1.0)
        bonus = 0.2 if warm else 0.0
        slo_factor = max(1.0, slo / 250)
        score = (free_hbm / kv_required) + link_bw - queue_penalty - slo_factor + bonus
        return score

    def pick_slice(self, required_kv: int, candidates: List[dict]) -> Optional[str]:
        scored = sorted(
            candidates,
            key=lambda c: (
                c.get("free_hbm", 0) - required_kv,
                -c.get("queue_penalty", 0),
                c.get("link_bw", 0),
            ),
            reverse=True,
        )
        for candidate in scored:
            if candidate.get("free_hbm", 0) > required_kv * 1.1:
                return candidate.get("id")
        return None
