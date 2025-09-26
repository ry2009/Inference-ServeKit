from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import grpc
from fastapi import FastAPI, HTTPException

from api import primerl_pb2, primerl_pb2_grpc

app = FastAPI(title="PrimeRL Async Decode Client")

_CHANNEL = None
_CLIENT = None


@asynccontextmanager
def _ensure_client() -> AsyncIterator[primerl_pb2_grpc.PrimeRLStub]:
    global _CHANNEL, _CLIENT
    if _CLIENT is None:
        target = "localhost:50051"
        _CHANNEL = grpc.aio.insecure_channel(target)
        _CLIENT = primerl_pb2_grpc.PrimeRLStub(_CHANNEL)
    try:
        yield _CLIENT
    finally:
        # keep channel open for reuse
        pass


@app.on_event("startup")
async def on_startup():
    # Warm channel
    async with _ensure_client():
        pass


@app.on_event("shutdown")
async def on_shutdown():
    global _CHANNEL, _CLIENT
    if _CHANNEL is not None:
        await _CHANNEL.close()
    _CHANNEL = None
    _CLIENT = None


@app.post("/start-episode")
async def start_episode(req: dict):
    async with _ensure_client() as client:
        message = primerl_pb2.StartReq(
            env_id=req.get("env_id", "default"),
            model=req.get("model", "llama3-8b"),
            prompt_fp=req.get("prompt_fp", b""),
            prompt=req.get("prompt", ""),
            pin_prefill=req.get("pin_prefill", False),
        )
        resp = await client.StartEpisode(message)
        return {"session_id": resp.session_id, "cache_hit": resp.cache_hit}


@app.post("/step")
async def step(req: dict):
    session_id = req.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    async def request_stream():
        yield primerl_pb2.StepReq(
            session_id=session_id,
            obs=req.get("obs", ""),
            max_new_tokens=req.get("max_new_tokens", 128),
            grammar_id=req.get("grammar_id", ""),
            speculative=req.get("speculative", False),
        )

    async with _ensure_client() as client:
        responses = []
        call = client.Step(request_stream())
        async for token in call:
            responses.append(
                {
                    "token": token.token,
                    "t_us": token.t_us,
                    "kv_bytes": token.kv_bytes,
                    "boundary": token.boundary,
                    "accepted": token.accepted,
                }
            )
        return {"tokens": responses}


@app.post("/end-episode")
async def end_episode(req: dict):
    session_id = req.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    async with _ensure_client() as client:
        message = primerl_pb2.EndReq(session_id=session_id)
        resp = await client.EndEpisode(message)
        return {"evicted": resp.evicted}
