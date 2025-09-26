from __future__ import annotations

import grpc

from api import primerl_pb2, primerl_pb2_grpc


class PrimeRLGrpcClient:
    def __init__(self, target: str = "localhost:50051"):
        self._target = target
        self._channel: grpc.aio.Channel | None = None
        self._stub: primerl_pb2_grpc.PrimeRLStub | None = None

    async def _ensure_stub(self) -> primerl_pb2_grpc.PrimeRLStub:
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(self._target)
            self._stub = primerl_pb2_grpc.PrimeRLStub(self._channel)
        return self._stub

    async def start_episode(self, **kwargs):
        stub = await self._ensure_stub()
        req = primerl_pb2.StartReq(**kwargs)
        return await stub.StartEpisode(req)

    async def step(self, *step_reqs):
        stub = await self._ensure_stub()

        async def iterator():
            for req in step_reqs:
                yield primerl_pb2.StepReq(**req)

        call = stub.Step(iterator())
        tokens = []
        async for resp in call:
            tokens.append(resp)
        return tokens

    async def end_episode(self, session_id: str):
        stub = await self._ensure_stub()
        return await stub.EndEpisode(primerl_pb2.EndReq(session_id=session_id))

    async def close(self):
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
