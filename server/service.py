from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import AsyncIterator, List

import httpx
from opentelemetry import trace

import grpc

from api import primerl_pb2, primerl_pb2_grpc
from cache.global_prefix_cache import GlobalPrefixCache
from cache.prefix_fingerprint import normalize, rolling_hash
from perf import exporters
from prime_stack.adapters import build_trace
from prime_stack.control_plane.router import RoutingRequest
from rl_client.batcher import Batcher
from rl_client.session_manager import SessionManager
from speculation.tool_boundary_spec import ToolBoundarySpec

logger = logging.getLogger(__name__)


class PrimeRLService(primerl_pb2_grpc.PrimeRLServicer):
    """PrimeRL gRPC service bridging trainers to model engines."""

    def __init__(
        self,
        engine,
        prefix_cache: GlobalPrefixCache,
        session_manager: SessionManager,
        cache_index=None,
        node_id: str | None = None,
        router=None,
        kv_estimator=None,
    ):
        self.engine = engine
        self.prefix_cache = prefix_cache
        self.session_manager = session_manager
        self.cache_index = cache_index
        self.batcher = Batcher(engine)
        self._batcher_task = asyncio.create_task(self.batcher.run())
        self.node_id = node_id or os.getenv("PRIMERL_NODE_ID", "node-local")
        self.speculator = ToolBoundarySpec(engine, engine, boundary_token="[TOOL_END]")
        self.verifier_url = os.getenv("PRIMERL_VERIFIER_URL")
        self.verifier_client = httpx.AsyncClient(timeout=30) if self.verifier_url else None
        self.router = router
        self.kv_estimator = kv_estimator
        self.tracer = trace.get_tracer("primerl.service")

    async def StartEpisode(
        self, request: primerl_pb2.StartReq, context: grpc.aio.ServicerContext
    ) -> primerl_pb2.StartResp:
        model = request.model or ""
        prompt_text = request.prompt
        prompt_fp = request.prompt_fp

        with self.tracer.start_as_current_span(
            "StartEpisode",
            attributes={"env_id": request.env_id, "model": request.model},
        ):
            if not prompt_fp and prompt_text:
                normalized = normalize(prompt_text)
                prompt_fp = rolling_hash(normalized)

            cache_hit = False
            if prompt_fp:
                meta = self.prefix_cache.get(prompt_fp)
                cache_hit = meta is not None
                counter = exporters.cache_hit if cache_hit else exporters.cache_miss
                counter.labels(model=model).inc()

            session_id = self.session_manager.start(request.env_id, model)
            meta_kwargs = {}
            if prompt_fp:
                meta_kwargs["prompt_fp"] = prompt_fp.hex()
            if prompt_text:
                meta_kwargs["prompt"] = prompt_text
            if meta_kwargs:
                self.session_manager.set_meta(session_id, **meta_kwargs)

            if self.router and self.kv_estimator and prompt_text:
                seq_len = len(prompt_text.split())
                kv_est = self.kv_estimator(seq_len=seq_len, batch=1)
                routing_req = self.router.route(
                    RoutingRequest(
                        prompt_fp=prompt_fp,
                        kv_estimate=kv_est,
                        slo_latency_ms=300,
                        model=model,
                    )
                )
                logger.info("Routing session %s to node %s", session_id, routing_req)

            engine_session_id = None
            if request.pin_prefill and prompt_text:
                try:
                    response = await self.engine.prefill(model=model, prompt=prompt_text, grammar=None)
                    engine_session_id = response.get("session_id")
                    exporters.tokens.labels(phase="prefill", model=model).inc(
                        response.get("tokens", 0)
                    )
                    if prompt_fp and self.cache_index and engine_session_id:
                        self.cache_index.register(prompt_fp, self.node_id)
                        self.prefix_cache.put(
                            prompt_fp,
                            meta={"model": model, "node_id": self.node_id, "tier": "hbm"},
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Prefill failed for session %s", session_id)
                    await context.abort(grpc.StatusCode.INTERNAL, str(exc))

            if engine_session_id:
                self.session_manager.bind_engine(session_id, engine_session_id)
                if prompt_fp:
                    meta = {"engine_session_id": engine_session_id, "model": model}
                    self.prefix_cache.put(prompt_fp, meta)

            return primerl_pb2.StartResp(session_id=session_id, cache_hit=cache_hit)

    async def Step(
        self, request_iterator: AsyncIterator[primerl_pb2.StepReq], context: grpc.aio.ServicerContext
    ) -> AsyncIterator[primerl_pb2.StepResp]:
        async for request in request_iterator:
            with self.tracer.start_as_current_span(
                "Step",
                attributes={
                    "session_id": request.session_id,
                    "speculative": request.speculative,
                    "grammar_id": request.grammar_id,
                },
            ):
                session = self.session_manager.get(request.session_id)
                if session is None:
                    await context.abort(grpc.StatusCode.NOT_FOUND, "unknown session")
                    return

                engine_session_id = session.get("engine_session_id") or request.session_id
                model = session.get("model", "unknown")
                prompt_text = session.get("meta", {}).get("prompt", "") + request.obs
                exporters.queue_depth.labels(model=model).inc()
                try:
                    if request.speculative and request.grammar_id:
                        try:
                            tokens, accepted_mask = await self.speculator.generate(
                                session_id=engine_session_id,
                                obs=request.obs,
                                max_new=request.max_new_tokens,
                                grammar=request.grammar_id,
                                prompt=prompt_text,
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("Speculation failed; falling back to normal decode")
                            tokens = await self.batcher.submit(
                                session_id=engine_session_id,
                                model=model,
                                obs=request.obs,
                                max_new=request.max_new_tokens,
                                grammar=request.grammar_id or None,
                                speculative=False,
                                prompt=prompt_text,
                            )
                            accepted_mask = [True] * len(tokens)
                    else:
                        tokens = await self.batcher.submit(
                            session_id=engine_session_id,
                            model=model,
                            obs=request.obs,
                            max_new=request.max_new_tokens,
                            grammar=request.grammar_id or None,
                            speculative=request.speculative,
                            prompt=prompt_text,
                        )
                        accepted_mask = [True] * len(tokens)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Decode failure for session %s: %s", request.session_id, exc)
                    tokens, accepted_mask = await self._failover_replay(
                        session, request, model
                    )
                finally:
                    exporters.queue_depth.labels(model=model).dec()

                token_texts: List[str] = []
                for idx, token in enumerate(tokens):
                    accepted = accepted_mask[idx] if idx < len(accepted_mask) else True
                    kv_bytes = token.get("kv_bytes", 0)
                    self.session_manager.touch(request.session_id, kv_bytes=kv_bytes)
                    exporters.tokens.labels(phase="decode", model=model).inc()
                    latency = token.get("t_us", 0) / 1_000_000
                    exporters.latency.labels(route="Step", model=model).observe(latency)
                    exporters.kv_bytes.labels(model=model).set(kv_bytes)
                    token_texts.append(token.get("token", ""))
                    yield primerl_pb2.StepResp(
                        token=token.get("token", ""),
                        t_us=token.get("t_us", 0),
                        kv_bytes=kv_bytes,
                        boundary=token.get("boundary", False),
                        accepted=accepted,
                    )
                self.session_manager.record_tokens(request.session_id, token_texts, accepted_mask)

    async def EndEpisode(
        self, request: primerl_pb2.EndReq, context: grpc.aio.ServicerContext
    ) -> primerl_pb2.EndResp:
        with self.tracer.start_as_current_span(
            "EndEpisode", attributes={"session_id": request.session_id}
        ):
            session = self.session_manager.get(request.session_id)
            if session is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "unknown session")
                return primerl_pb2.EndResp(evicted=False)

            engine_session_id = session.get("engine_session_id")
            if engine_session_id and hasattr(self.engine, "close_session"):
                try:
                    await self.engine.close_session(engine_session_id)  # type: ignore[misc]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to close engine session %s: %s", engine_session_id, exc)

            if self.verifier_client:
                trace = self.session_manager.trace(request.session_id)
                episode = {
                    "episode_id": request.session_id,
                    "model": trace["model"],
                    "prompt_fp": trace.get("meta", {}).get("prompt_fp"),
                    "tokens": " ".join(trace["tokens"]),
                    "accepted_mask": trace["accepted_mask"],
                    "tools": trace.get("tools", []),
                    "meta": trace.get("meta", {}),
                }
                metrics = {"kv_bytes": trace["kv_bytes"]}
                policy_meta = {"sandbox_profile": "default", "egress_blocked": True}
                verifier_payload = build_trace(episode, metrics, policy_meta)
                try:
                    response = await self.verifier_client.post(
                        f"{self.verifier_url}/verify", json=verifier_payload
                    )
                    response.raise_for_status()
                    session["verifier_result"] = response.json()
                except httpx.HTTPError as exc:  # noqa: BLE001
                    logger.warning("Verifier call failed: %s", exc)

            self.session_manager.end(request.session_id)
            exporters.latency.labels(route="EndEpisode", model=session.get("model", "unknown")).observe(0)
            return primerl_pb2.EndResp(evicted=True)

    async def shutdown(self):
        self._batcher_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._batcher_task
        if self.verifier_client:
            await self.verifier_client.aclose()

    async def _failover_replay(self, session: dict, request: primerl_pb2.StepReq, model: str):
        prompt = session.get("meta", {}).get("prompt")
        engine_session_id = session.get("engine_session_id")
        if not prompt:
            raise RuntimeError("Missing prompt for failover replay")
        try:
            response = await self.engine.prefill(model=model, prompt=prompt, grammar=request.grammar_id or None)
            engine_session_id = response.get("session_id")
            if engine_session_id:
                self.session_manager.bind_engine(request.session_id, engine_session_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failover prefill failed: %s", exc)
            raise
        engine_sid = engine_session_id or request.session_id
        prompt_text = session.get("meta", {}).get("prompt", "") + request.obs
        tokens = await self.batcher.submit(
            session_id=engine_sid,
            model=model,
            obs=request.obs,
            max_new=request.max_new_tokens,
            grammar=request.grammar_id or None,
            speculative=False,
            prompt=prompt_text,
        )
        return tokens, [True] * len(tokens)
