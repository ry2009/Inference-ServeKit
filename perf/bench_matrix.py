"""Benchmark matrix runner for PrimeRL-ServeKit++."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class BenchCase:
    context_tokens: int
    batch_size: int
    precision: str
    speculation: bool
    cache: bool


def sweep_cases(mini: bool = False) -> Iterable[BenchCase]:
    contexts = [4096, 16384, 32768] if not mini else [4096]
    batches = [1, 8, 32] if not mini else [1]
    precisions = ["bf16", "fp8", "int4"] if not mini else ["bf16"]
    for ctx in contexts:
        for batch in batches:
            for precision in precisions:
                for spec in [False, True] if not mini else [False, True]:
                    for cache in [False, True]:
                        yield BenchCase(ctx, batch, precision, spec, cache)


def run_case(case: BenchCase) -> dict:
    # TODO: integrate with actual load generator against deployed engine.
    return {
        "context_tokens": case.context_tokens,
        "batch_size": case.batch_size,
        "precision": case.precision,
        "speculation": case.speculation,
        "cache": case.cache,
        "tokens_per_sec": 1000.0,
        "ttfb_ms": 120.0,
        "p95_ms": 280.0,
        "gpu_util": 0.6,
        "cache_hit_rate": 0.7,
    }


def main(out_path: str, mini: bool = False):
    rows = [run_case(case) for case in sweep_cases(mini=mini)]
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="perf/results.csv")
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()
    main(args.out, mini=args.mini)
