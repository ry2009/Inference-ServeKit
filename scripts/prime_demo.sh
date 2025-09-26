#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null; then
  echo "Docker required for prime_demo" >&2
  exit 1
fi

STACK=(redis engine verifier primerl)

echo "[prime_demo] building containers"
docker compose build "${STACK[@]}"

echo "[prime_demo] starting stack"
docker compose up -d "${STACK[@]}"
trap 'docker compose down' EXIT

sleep 5

function run_step() {
  echo -e "\n[prime_demo] $1"
  shift
  "$@"
}

run_step "PPO demo" \
  python demos/ppo_summarize_tooluse.py --prompt "Plan a SQL query" \
    --grammar sql_v1 --speculative --steps 8 --verifier http://localhost:8080

run_step "GRPO mini" \
  python grpo/learner.py --steps 3 --k 3 --speculative \
    --verifier http://localhost:8080 --target localhost:50051

run_step "Eval harness" \
  python demos/eval_harness.py --verifier http://localhost:8080

run_step "Perf mini" \
  python perf/bench_matrix.py --mini --out perf/results.csv

run_step "Perf gate" \
  python perf/perf_gate.py --baseline perf/baseline.csv --current perf/results.csv || true

echo "\n[prime_demo] metrics available on http://localhost:9300" 

echo "[prime_demo] completed"
