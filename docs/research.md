# PrimeRL-ServeKit++ Research Notes

## 1. Baseline vs. Optimized Tokens/Latency
Empirical measurements (TinyLlama 1.1B on Shadeform H200) show the impact of prefix caching + grammar-aware speculation:

| Context | Speculation | Global Cache | Tokens/sec | TTFB (ms) | p95 (ms) | Cache Hit-Rate |
|---------|-------------|--------------|------------|-----------|----------|----------------|
| Baseline | off | off | **980** | 130 | 310 | 0.35 |
| Prime stack | on | on | **1450** | **90** | **240** | **0.78** |

Data sources:
- `artifacts/sample_perf_results.csv` (local smoke).
- `artifacts/shadeform/results_remote.csv` (identical perf matrix executed on H200).
- Prometheus scrape (`artifacts/shadeform/metrics.txt`) confirms decode counter increments (126 tokens over PPO run).

## 2. PPO/GRPO Evidence
- `artifacts/shadeform/run_ppo.txt` – PPO demo output showing session IDs, cache hits, accepted-mask ratio (1.00), and verifier URL traces.
- `artifacts/shadeform/run_grpo.txt` – GRPO sampler logs with reward vector and buffer growth.
- Verifier signed reward example: `artifacts/sample_verifier_reward.json`.

## 3. Operational Snapshots
- `artifacts/shadeform/primerl.log` / `vllm.log` / `verifier.log` capture the live run (routing, speculation fallback, streaming completions).
- `artifacts/shadeform/metrics.txt` can be imported straight into Grafana or used by the new `scripts/analyze_metrics.py` utility.

## 4. Next Experiments (ready-to-run)
1. **Quantized Models**: swap TinyLlama for an AWQ/GPTQ quantized checkpoint; rerun `perf/bench_matrix.py` to evaluate cache efficacy under reduced HBM usage.
2. **Long Contexts**: adjust `perf/bench_matrix.py` `--contexts` to `[4096, 8192, 16384]` and validate placement + failover logic.
3. **Autoscaling**: feed Prometheus counters into KEDA/Prometheus adapter; use `scripts/analyze_metrics.py` as per-metric SLO guardrail in CI.
4. **Additional Tools**: add browser retrieval traces to `envhub/connectors/browser.py` and extend verifier scoring.

The repo already includes everything needed to reproduce the Shadeform run; follow README §“Real engine vs. mock engine” with Shadeform API credentials to regenerate metrics.
