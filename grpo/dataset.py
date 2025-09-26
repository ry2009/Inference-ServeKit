from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class Experience:
    prompt: str
    tokens: list
    reward: float
    advantage: float
    accepted_mask: list | None = None


class ExperienceBuffer:
    """Simple in-memory buffer for GRPO experience tuples."""

    def __init__(self):
        self._items: list[Experience] = []

    def extend(self, items: Iterable[Experience]):
        self._items.extend(items)

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._items)

    def clear(self):
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
