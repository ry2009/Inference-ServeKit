from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelRecord:
    name: str
    version: str
    artifacts: Dict[str, str]
    metrics: Dict[str, float]
    tags: List[str] = field(default_factory=list)


@dataclass
class NodeRecord:
    id: str
    models: List[str]
    free_hbm: int
    link_bw: float
    queue_penalty: float


class Registry:
    """In-memory registry for models, artifacts, and serving nodes."""

    def __init__(self):
        self.models: Dict[str, ModelRecord] = {}
        self.nodes: Dict[str, NodeRecord] = {}

    def register_model(self, record: ModelRecord):
        self.models[record.name] = record

    def register_node(self, record: NodeRecord):
        self.nodes[record.id] = record

    def update_node_capacity(self, node_id: str, free_hbm: int, queue_penalty: float):
        if node_id not in self.nodes:
            return
        record = self.nodes[node_id]
        record.free_hbm = free_hbm
        record.queue_penalty = queue_penalty

    def nodes_for_model(self, model: str) -> List[NodeRecord]:
        return [n for n in self.nodes.values() if model in n.models]

    def artifact_path(self, model: str, artifact: str) -> str | None:
        rec = self.models.get(model)
        if not rec:
            return None
        return rec.artifacts.get(artifact)
