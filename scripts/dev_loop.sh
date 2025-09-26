#!/usr/bin/env bash
set -euo pipefail

function header() {
  echo "\n=== $1 ==="
}

header "Lint & unit tests"
pytest -q

header "Build images"
docker compose build

header "Bring up stack"
docker compose up -d redis verifier primerl

header "gRPC codegen"
make gen-proto

header "Smoke PPO over service"
python demos/ppo_summarize_tooluse.py --engine vllm --model llama3-8b-instruct \
  --context 4096 --prefix-cache on --speculative off --grammar sql_v1 --steps 5 || true

header "Perf matrix (mini)"
python perf/bench_matrix.py --mini --out perf/results.csv
python perf/perf_gate.py --baseline perf/baseline.csv --current perf/results.csv || true

header "Mini GRPO"
python grpo/learner.py --k 2 --env sql_qa --steps 10 --verifier http://localhost:8080 || true
