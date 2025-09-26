# Observability Notes

## Metrics
- Prometheus exporter runs inside `server/main.py` (`PRIMERL_METRICS_PORT`, default 9300).
- Key series:
  - `primerl_tokens_total{phase}` – per-phase token counts.
  - `primerl_request_latency_seconds{route}` – histogram (p50/p95/p99).
  - `primerl_prefix_cache_hits_total` / `_misses_total` – cache efficiency.
  - `primerl_kv_resident_bytes{model}` – KV residency gauge.
- Scrape configuration example:
  ```yaml
  - job_name: primerl
    static_configs:
      - targets: ['primerl:9300']
  ```

## Tracing
- OpenTelemetry tracer configured in `server/service.py`; set `OTEL_EXPORTER_OTLP_ENDPOINT` to export spans.
- Spans emitted for `StartEpisode`, `Step`, and `EndEpisode` with attributes: session_id, grammar_id, speculative flag, accepted mask size.

## Dashboard
- Import `k8s/HelmChart/dashboards/grafana.json` into Grafana. Panels cover tokens/s, p95, KV bytes, queue depth, cache hit rate.
- Augment with panel for verifier rewards using PromQL:
  ```
  rate(primerl_reward_sum[5m]) / rate(primerl_reward_count[5m])
  ```
  (Add counter in future if desired.)
- Sample Prometheus scrape from Shadeform H200 run is stored at `artifacts/shadeform/metrics.txt`.

## Logs
- Prime demo (`scripts/prime_demo.sh`) captures PPO and GRPO logs with accepted_mask ratios; collect them for regression baselines.
