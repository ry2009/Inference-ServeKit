from __future__ import annotations

import asyncio
import logging
import os

import grpc
from prometheus_client import start_http_server

from api import primerl_pb2_grpc
from cache.global_prefix_cache import GlobalPrefixCache
from engines import DummyAdapter, SGLangAdapter, TRTLLMAdapter, VLLMAdapter
from placement.kv_budget import kv_bytes
from placement.scheduler import Scheduler
from prime_stack.control_plane import CacheIndex, ModelRecord, NodeRecord, Registry, Router
from rl_client.session_manager import SessionManager
from server.service import PrimeRLService

logging.basicConfig(level=logging.INFO)


def build_engine(engine_type: str, base_url: str | None):
    if engine_type == "vllm":
        if not base_url:
            raise ValueError("PRIMERL_ENGINE_BASE_URL must be set for vLLM engine")
        return VLLMAdapter(base_url)
    if engine_type == "sglang":
        if not base_url:
            raise ValueError("PRIMERL_ENGINE_BASE_URL must be set for SGLang engine")
        return SGLangAdapter(base_url)
    if engine_type == "trtllm":
        if not base_url:
            raise ValueError("PRIMERL_ENGINE_BASE_URL must be set for TRT-LLM engine")
        return TRTLLMAdapter(base_url)
    return DummyAdapter()


async def serve():
    engine_type = os.getenv("PRIMERL_ENGINE", "dummy").lower()
    base_url = os.getenv("PRIMERL_ENGINE_BASE_URL")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    listen_port = int(os.getenv("PRIMERL_PORT", "50051"))
    metrics_port = int(os.getenv("PRIMERL_METRICS_PORT", "9300"))
    node_id = os.getenv("PRIMERL_NODE_ID", "node-local")

    engine = build_engine(engine_type, base_url)
    prefix_cache = GlobalPrefixCache(redis_url)
    cache_index = CacheIndex()
    registry = Registry()
    scheduler = Scheduler()
    registry.register_node(
        NodeRecord(
            id=node_id,
            models=["llama3-8b"],
            free_hbm=80 * 1024**3,
            link_bw=900.0,
            queue_penalty=0.1,
        )
    )
    registry.register_model(
        ModelRecord(
            name="llama3-8b",
            version="0.1",
            artifacts={"weights": "s3://placeholder"},
            metrics={"tokens_per_sec": 1000.0},
        )
    )
    router = Router(scheduler=scheduler, cache_index=cache_index, registry=registry)

    def kv_estimator(seq_len: int, batch: int) -> int:
        return kv_bytes(seq_len=seq_len, layers=40, heads=40, head_dim=128, batch=batch)

    session_manager = SessionManager()
    service = PrimeRLService(
        engine,
        prefix_cache,
        session_manager,
        cache_index=cache_index,
        node_id=node_id,
        router=router,
        kv_estimator=kv_estimator,
    )

    server = grpc.aio.server()
    primerl_pb2_grpc.add_PrimeRLServicer_to_server(service, server)
    server.add_insecure_port(f"[::]:{listen_port}")

    logging.info("PrimeRL server starting on port %s (engine=%s)", listen_port, engine_type)
    start_http_server(metrics_port)
    logging.info("Metrics exporter listening on %s", metrics_port)

    await server.start()
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:  # pragma: no cover
        logging.info("PrimeRL server cancelled")
    finally:
        await service.shutdown()
        await server.stop(grace=None)


def main():
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:  # pragma: no cover
        logging.info("PrimeRL server interrupted")


if __name__ == "__main__":
    main()
