# PrimeRL-ServeKit++ Runbook

## Incident Playbooks
- **Tail Latency Spikes**
  - Check batcher queue depth and latency metrics.
  - Reduce batch interval or disable speculation for tool-heavy prompts.
  - Validate GPU utilization and MIG placement decisions.
- **Cache Thrash**
  - Inspect Redis prefix cache hit rate.
  - Adjust fingerprint normalization and eviction cost weights.
  - Pre-warm hot prefixes on job enqueue.
- **MIG Fragmentation**
  - Run `placement.mig_inventory.list_gpus()` to inspect free slices.
  - Rebalance workloads so long-context jobs occupy larger slices.
- **Verifier Outage**
  - Requests fall back to local scoring; monitor `primerl_policy_penalty` metric.
  - Restart the `verifier` deployment (Docker Compose or Helm) and re-run smoke evals.
- **Cache Miss Spike**
  - Inspect `primerl_prefix_cache_misses_total` vs. hits to confirm drift.
  - Warm prefixes via `prime_stack/control_plane/router.CacheIndex` or prefill jobs.
- **Reward Regression after Speculation**
  - Compare accepted token masks vs. total tokens per sample.
  - Gate speculation on reward delta epsilon and automatically fallback.

## Operational Tasks
- Rotate Redis credentials and flush cache on incompatible schema changes.
- Regenerate protobuf stubs with `make gen-proto` when the API evolves.
- Schedule nightly `perf/bench_matrix.py` sweeps and compare against baselines.
- Validate Grafana dashboards after chart changes to ensure Prometheus metrics match queries.
- Run `scripts/dev_loop.sh` before merges for an end-to-end PPO/GRPO + perf sanity.
- Use `scripts/prime_demo.sh` for customer-facing demos; it boots Redis/mock-engine/verifier/servekit, runs PPO, GRPO, eval harness, and perf gates, then tears everything down.
- Replicating the Shadeform H200 validation: follow the Shadeform API flow (create → poll → SSH), start vLLM (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`), then run PrimeRL server + verifier. Example logs and metrics live in `artifacts/shadeform/`.
