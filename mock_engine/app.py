from __future__ import annotations

import asyncio
import random
import time

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock LLM Engine")


class PrefillReq(BaseModel):
    model: str
    prompt: str
    grammar: str | None = None


class DecodeReq(BaseModel):
    session_id: str
    obs: str
    max_new_tokens: int
    grammar: str | None = None
    speculative: bool = False


@app.post("/prefill")
async def prefill(req: PrefillReq):
    await asyncio.sleep(0.01)
    tokens = len(req.prompt.split())
    return {"session_id": f"mock-{hash(req.prompt) & 0xFFFF:X}", "tokens": tokens}


@app.post("/decode")
async def decode(req: DecodeReq):
    async def generator():
        for idx in range(req.max_new_tokens):
            await asyncio.sleep(0.005)
            token = f"tok_{idx}" if not req.speculative else f"draft_{idx}"
            yield {
                "token": token,
                "t_us": int(time.time() * 1e6),
                "kv_bytes": (idx + 1) * 2048,
                "boundary": (idx + 1) % 5 == 0,
            }

    return [token async for token in generator()]
