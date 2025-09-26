from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class Job:
    id: str
    payload: dict


class JobQueue:
    """Async queue abstraction (backed by Redis streams or Kafka later)."""

    def __init__(self):
        self._queue: asyncio.Queue[Job] = asyncio.Queue()

    async def put(self, job: Job):
        await self._queue.put(job)

    async def get(self, timeout: Optional[float] = None) -> Optional[Job]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def stream(self) -> AsyncIterator[Job]:
        while True:
            job = await self._queue.get()
            yield job
