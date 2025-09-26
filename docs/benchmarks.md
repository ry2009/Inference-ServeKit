# Benchmarking Plan

1. Run `perf/bench_matrix.py` across context lengths {4k, 16k, 32k}, batch sizes {1, 8, 32}, precisions {bf16, fp8, int4}, speculation {off, on}, cache {off, on}. Use `--mini` in CI smoke tests.
2. Collect metrics: tokens/sec for prefill and decode, time-to-first-byte, p95 latency, GPU utilization, cache hit rate.
3. Persist results to `perf/results.csv`; compare with golden `perf/baseline.csv` via `perf/perf_gate.py`.
4. Track deltas on PRs and fail CI when regressions exceed 5%.
5. For MIG-aware tests, record slice allocation and HBM headroom to validate placement heuristics.
