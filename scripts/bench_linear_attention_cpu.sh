#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=. scripts/bench_linear_attention.py --device cpu --iters 5 --lengths 128 256 512
