#!/usr/bin/env python3
"""Parse Prometheus text metrics emitted by PrimeRL and derive headline stats."""

from __future__ import annotations

import argparse
from pathlib import Path

import re

TOKEN_RE = re.compile(r"primerl_tokens_total\{([^}]*)\}\s+([0-9.]+)")
LATENCY_RE = re.compile(r"primerl_request_latency_seconds_bucket\{([^}]*)\}\s+([0-9.]+)")


def parse_labels(label_str: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in label_str.split(","):
        if not item:
            continue
        key, value = item.split("=")
        labels[key] = value.strip('"')
    return labels


def summarize_tokens(metrics: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for match in TOKEN_RE.finditer(metrics):
        labels = parse_labels(match.group(1))
        model = labels.get("model", "unknown")
        phase = labels.get("phase", "decode")
        key = f"{model}:{phase}"
        totals[key] = float(match.group(2))
    return totals


def summarize_latency(metrics: str) -> dict[str, float]:
    # naive p95 estimator from histogram buckets (if present)
    latencies: dict[str, float] = {}
    for match in LATENCY_RE.finditer(metrics):
        labels = parse_labels(match.group(1))
        model = labels.get("model", "unknown")
        route = labels.get("route", "Step")
        le = labels.get("le")
        if le == "0.95":
            latencies[f"{model}:{route}"] = float(match.group(2))
    return latencies


def main(path: Path):
    metrics = path.read_text()
    tokens = summarize_tokens(metrics)
    latency = summarize_latency(metrics)

    print("Tokens generated per model/phase:")
    for key, value in tokens.items():
        print(f"  {key}: {value}")

    if latency:
        print("\nLatency buckets (<=0.95):")
        for key, value in latency.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PrimeRL Prometheus metrics")
    parser.add_argument("path", default="artifacts/shadeform/metrics.txt")
    args = parser.parse_args()
    main(Path(args.path))
