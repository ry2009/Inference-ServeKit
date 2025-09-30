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

## 5. Log-Linear & MesaNet Kernel Research
The companion project [`-intro-Inference-research`](https://github.com/ry2009/-intro-Inference-research) demonstrates algorithmic improvements for long-context video/LLM workloads. Running `python final_working_speedups.py` on the current hardware produced:

| Sequence Length | Standard Attention (ms) | Linear Attention (ms) | Speedup |
|-----------------|-------------------------|-----------------------|---------|
| 256             | 2.99 (CPU)              | 2.06                  | 1.45×   |
| 512             | 11.87 (CPU)             | 3.01                  | 3.94×   |
| 1024            | 28.52 (CPU)             | 4.21                  | 6.77×   |
| 2048            | 105.30 (CPU)            | 13.04                 | 8.07×   |
| 4096            | 568.61 (CPU)            | 25.59                 | 22.22×  |
| 256             | 0.274 (H200)            | 0.203                 | 1.35×   |
| 512             | 0.070 (H200)            | 0.171                 | 0.41×   |
| 1024            | 0.100 (H200)            | 0.155                 | 0.65×   |

Artifacts:
- `artifacts/research/linear_attention_cpu.txt` – baseline CPU benchmark.
- `artifacts/research/linear_attention_cpu_long.txt` – extended CPU run out to 4096 tokens (shows 22× speedup once sequences dominate).
- `artifacts/research/linear_attention_h200.txt` – H200 measurements captured with `scripts/bench_linear_attention.py` (shows sub-millisecond kernels and highlights cases where further kernel fusion is required to maintain speedups at small batch sizes).
- `artifacts/research/RESULTS_SUMMARY.md` – original summary from the research repo.

Integration ideas:
1. **Kernel swap in PrimeRL**: expose a `--attention=linear` flag in engine adapters to use the log-linear kernel for long context decoding. Measure impact via `perf/bench_matrix.py`.
2. **GPU kernel optimization**: replace the naive PyTorch implementation with a Triton/CUDA graph kernel so the H200 measurements match the CPU speedups (current results show the crossover point; we need a fused kernel to remove the small-sequence penalty).
3. **Shift Parallelism prototype**: extend `perf/bench_matrix.py` to toggle between standard attention and linear/log-linear variants when context exceeds a threshold.
4. **Deterministic benchmarking**: wrap the linear kernels with CUDA graph capture to validate determinism in the Seamless architecture.
