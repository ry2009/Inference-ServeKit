import asyncio
import itertools
import time


class DummyAdapter:
    """In-memory adapter that synthesizes tokens for development."""

    def __init__(self):
        self._counter = itertools.count()

    async def prefill(self, model: str, prompt: str, grammar: str | None):
        await asyncio.sleep(0.01)
        return {"session_id": f"dummy-{next(self._counter)}", "tokens": len(prompt.split())}

    async def continue_decode(
        self,
        session_id: str,
        obs: str,
        max_new: int,
        grammar: str | None,
        speculative: bool,
    ):
        for idx in range(max_new):
            await asyncio.sleep(0.005)
            yield {
                "token": f"tok-{idx}",
                "t_us": int(time.time() * 1e6),
                "kv_bytes": (idx + 1) * 1024,
                "boundary": idx == max_new - 1,
            }

    async def close_session(self, session_id: str):
        await asyncio.sleep(0)
