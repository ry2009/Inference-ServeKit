# PrimeRL-ServeKit++

PrimeRL-ServeKit++ provides a production-style reinforcement learning serving bridge that allows PPO/GRPO trainers to call an inference service (vLLM, SGLang, TRT-LLM) with stateful decode, global prefix caching, grammar-aware speculative tool use, MIG-aware placement, and first-class observability. The repository is intentionally modular so each subsystem (engines, RL client, cache, placement, speculation, GRPO, Env Hub, performance gates) can be iterated independently or shipped incrementally.

## Highlights
- gRPC bridge (`PrimeRL`) exposing `StartEpisode`, `Step`, and `EndEpisode` streaming APIs.
- Engine adapters for vLLM, SGLang, and TRT-LLM with stateful decode and speculative execution surfaces.
- RL client with batching, session tracking, and grammar integration for tool-aware interactions.
- Global prefix cache backed by Redis with fingerprint normalization and cost-aware eviction heuristics.
- Grammar-aware draft/verify speculation to accelerate tool-call workflows while preserving correctness.
- MIG-aware placement and KV budgeting helpers to map workloads to heterogeneous GPU fleets.
- GRPO utilities (sampler, rater, advantage) with hooks for tool-trace scoring and accepted-token masks.
- Env Hub connectors (SQL, browser, code sandbox, HTTP) with reward shapers and schema grammars.
- Observability stack exporting Prometheus metrics, OpenTelemetry traces, and Grafana dashboards.
- Performance benchmark matrix with CI perf gates to block regressions automatically.
- Kubernetes Helm chart for deploying the bridge with autoscaling hooks and dashboards.
- Prime Stack modules: control plane router + registry, verifier service with signed logs, sandbox policies, and eval registry runners.

## Quickstart
1. Create a Python virtual environment and install dependencies:
   ```bash
   make dev
   ```
2. Generate the gRPC bindings:
   ```bash
   make gen-proto
   ```
3. Add the repo to your Python path when running modules (e.g., `export PYTHONPATH=$PYTHONPATH:$(pwd)`).
4. Launch the gRPC service locally:
   ```bash
   PYTHONPATH=. python -m server.main
   ```
   or use Docker Compose to bring up Redis alongside the bridge.
5. For an end-to-end smoke, run `scripts/dev_loop.sh` to exercise PPO, GRPO, and perf gates.

### Real engine vs. mock engine
- Set `PRIMERL_ENGINE_BASE_URL` to your vLLM/SGLang/TRT-LLM endpoint and `PRIMERL_ENGINE` accordingly (defaults to `dummy`).
- `docker-compose.yml` includes a lightweight mock engine (`mock_engine/app.py`) so you can run the full stack (`docker compose up redis engine verifier primerl`). Swap it out by editing the environment variables or removing the `engine` service when targeting real backends.

### Artifacts & Observability
- Sample verifier reward output lives in `artifacts/sample_verifier_reward.json`; perf snapshots in `artifacts/sample_perf_results.csv`.
- Prometheus metrics default to `:9300`; import `k8s/HelmChart/dashboards/grafana.json` into Grafana to visualize tokens/s, p95, cache hit-rate, and KV bytes.
- Shadeform H200 run (TinyLlama 1.1B via vLLM) captured in `artifacts/shadeform/`:
  - `primerl.log`, `vllm.log`, `verifier.log` – live logs from GPU-backed inference.
  - `metrics.txt` – Prometheus scrape showing actual token counters.
  - `run_ppo.txt`, `run_grpo.txt` – PPO & GRPO outputs against the remote engine.
  - `results_remote.csv` – perf sweep executed on the GPU host.
- `scripts/analyze_metrics.py` parses any Prometheus dump (e.g., `metrics.txt`) and prints per-model token totals / latency buckets.
4. Run the PPO demo, then graduate to GRPO with speculative decoding enabled.

See `docs/design.md` for the architectural overview and `docs/runbook.md` for operational guidance.
