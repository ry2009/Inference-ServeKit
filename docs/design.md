# PrimeRL-ServeKit++ Design

## Overview
PrimeRL-ServeKit++ bridges reinforcement learning trainers (PPO, GRPO) with large language model inference frameworks such as vLLM, SGLang, and TensorRT-LLM. The service exposes a gRPC API that presents stateful decode semantics, prefix caching, and grammar-aware speculative execution.

## Major Components
- **gRPC API**: Defines `StartEpisode`, `Step`, and `EndEpisode` streaming RPCs for managing decode sessions.
- **Engine Adapters**: Async HTTP clients that talk to model backends and provide uniform prefill/continue interfaces.
- **RL Client Layer**: Session manager, batcher, and grammar loader to orchestrate trainer traffic.
- **Prefix Cache**: Redis-backed cache keyed by prompt fingerprints to warm subsequent episodes.
- **Speculation**: Draft/verify speculation across engines with grammar-aware boundary handling.
- **Placement**: MIG inventory and scheduling utilities to route requests to GPU slices based on KV budgets.
- **GRPO Stack**: Sampler, rater, advantage computation, and learner hooks for group-relative optimization.
- **Env Hub**: Tool connectors, reward shapers, and JSON grammars for tool-safe interactions.
- **Observability**: Prometheus metrics, OpenTelemetry traces, and Grafana dashboards for performance visibility.
- **Prime Stack Integrations**: Control plane router/registry, Prime Verifier, sandboxed tool runners, eval registry, and adapters that tie ServeKit outputs into signed reward pipelines.

## Data Flow
1. Trainer calls `StartEpisode`, optionally pinning prefix prefill.
2. Batcher coalesces `Step` requests and forwards to engine adapter.
3. Prefix cache resolves fingerprint hits, reducing cold-start latency.
4. Speculation module drafts responses and verifies accepted tokens.
5. Metrics exporters record latency, tokens, queue depth, and cache health.
6. Placement module provides MIG-aware routing guidance, consumed by deployment/controller logic.

## Future Work
- Implement rollback for speculative mismatch and integrate accepted token masks into GRPO learner.
- Complete trainer adapters (TRL, CleanRL, RLlib) with reference examples.
- Extend Env Hub connectors with authenticated browser automation and sandboxed code execution policies.
- Replace mock engine with production vLLM/SGLang/TRT-LLM deployments and export real perf snapshots.
- Integrate Prime control-plane queues (Kafka/PubSub) for multi-region fanout.

## Integration Checklist
1. Configure `PRIMERL_ENGINE` + `PRIMERL_ENGINE_BASE_URL` to your serving pools (vLLM/SGLang/TRT-LLM).
2. Register fleet state in `prime_stack/control_plane/registry.py` (models, node capacities, tags).
3. Point `PRIMERL_VERIFIER_URL` to the existing Prime Verifier (or keep the bundled one).
4. Populate `prime_stack/eval_registry/tasks` with house evals and nightly sweeps.
5. Wire Prometheus scrape + Grafana dashboards using `k8s/HelmChart/dashboards/grafana.json`.
6. Run `scripts/prime_demo.sh` to verify PPO, GRPO, perf, and eval harness in one pass.
7. For GPU validation, point `PRIMERL_ENGINE_BASE_URL` at a Shadeform (or comparable) vLLM endpoint; see `artifacts/shadeform/` for an end-to-end H200 example.
