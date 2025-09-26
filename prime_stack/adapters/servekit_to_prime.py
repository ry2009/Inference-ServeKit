from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prime_stack.control_plane.router import RoutingRequest


@dataclass
class EpisodeRequest:
    env_id: str
    model: str
    prompt: str
    prompt_fp: bytes | None
    seq_len: int
    batch: int


def to_routing_request(req: EpisodeRequest, kv_estimator) -> RoutingRequest:
    kv_bytes = kv_estimator(seq_len=req.seq_len, batch=req.batch)
    return RoutingRequest(
        prompt_fp=req.prompt_fp,
        kv_estimate=kv_bytes,
        slo_latency_ms=250,
        model=req.model,
    )
