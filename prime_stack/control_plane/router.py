from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class RoutingRequest:
    prompt_fp: bytes | None
    kv_estimate: int
    slo_latency_ms: int
    model: str


@dataclass
class NodeInfo:
    id: str
    warm_prefixes: set[bytes]
    free_hbm: int
    queue_penalty: float
    link_bw: float


class Router:
    """Simple control-plane router that picks warm, well-provisioned nodes."""

    def __init__(self, scheduler, cache_index, registry):
        self.scheduler = scheduler
        self.cache_index = cache_index
        self.registry = registry

    def route(self, req: RoutingRequest) -> Optional[str]:
        warm_nodes = self.cache_index.lookup(req.prompt_fp) if req.prompt_fp else []
        candidates = self.registry.nodes_for_model(req.model)

        scored = []
        for node in candidates:
            warm = node.id in warm_nodes
            score = self.scheduler.score_node(
                node=node,
                warm=warm,
                kv_required=req.kv_estimate,
                slo=req.slo_latency_ms,
            )
            if score is not None:
                scored.append((score, node.id))

        if not scored:
            fallback = [n.id for n in candidates]
            return random.choice(fallback) if fallback else None

        scored.sort(reverse=True)
        return scored[0][1]


class CacheIndex:
    """Small in-memory cache index used by Router; backed by GlobalPrefixCache."""

    def __init__(self):
        self._index: dict[bytes, set[str]] = {}

    def register(self, prefix: bytes, node_id: str):
        self._index.setdefault(prefix, set()).add(node_id)

    def unregister_node(self, node_id: str):
        for nodes in self._index.values():
            nodes.discard(node_id)

    def lookup(self, prefix: bytes | None) -> Iterable[str]:
        if prefix is None:
            return []
        return list(self._index.get(prefix, []))
