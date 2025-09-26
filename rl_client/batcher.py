import asyncio
import collections
import time
from typing import Any, NamedTuple


class Req(NamedTuple):
    future: asyncio.Future
    engine: Any
    args: dict
    key: tuple


class Batcher:
    """Async decode batcher that coalesces compatible requests."""

    def __init__(self, engine, interval_ms: int = 8, max_batch: int = 32, p95_slo_ms: int = 300):
        self.engine = engine
        self.q: asyncio.Queue[Req] = asyncio.Queue()
        self.interval = interval_ms / 1000
        self.max_batch = max_batch
        self.p95_slo_ms = p95_slo_ms

    async def submit(self, **kwargs):
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        key = (kwargs["model"], kwargs.get("grammar"), kwargs.get("speculative"))
        await self.q.put(Req(fut, self.engine, kwargs, key))
        return await fut

    async def run(self):
        while True:
            group: list[Req] = []
            try:
                first = await asyncio.wait_for(self.q.get(), timeout=self.interval)
                group.append(first)
            except asyncio.TimeoutError:
                await asyncio.sleep(self.interval)
                continue

            start_ts = time.time()
            while len(group) < self.max_batch:
                try:
                    item = self.q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item.key == group[0].key:
                    group.append(item)
                else:
                    # Put back unmatched request and stop expanding batch.
                    self.q.put_nowait(item)
                    break

            coros = [self._collect(req.engine.continue_decode(**req.args)) for req in group]
            streams = await asyncio.gather(*coros, return_exceptions=True)
            latency_ms = (time.time() - start_ts) * 1000
            if latency_ms > self.p95_slo_ms:
                # TODO: emit a metric or log for SLO violation once observability is wired.
                pass

            for req, result in zip(group, streams):
                req.future.set_result(result)

    async def _collect(self, agen):
        out = []
        async for token in agen:
            out.append(token)
        return out
