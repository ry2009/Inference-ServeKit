"""Run tasks from the eval registry and summarize verifier rewards."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from prime_stack.eval_registry.runners.run_task import run_task


def main():
    parser = argparse.ArgumentParser(description="PrimeRL Eval Harness")
    parser.add_argument("--tasks-dir", default="prime_stack/eval_registry/tasks")
    parser.add_argument("--verifier", default="http://localhost:8080")
    parser.add_argument("--model", default="latest")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    results = []
    for task_path in tasks_dir.glob("*.yaml"):
        result = run_task(str(task_path), verifier_url=args.verifier, model=args.model)
        results.append((task_path.stem, result))

    for name, result in results:
        reward = result.get("reward")
        signature = result.get("signature", "")[:8]
        print(f"task={name} reward={reward:.3f} sig={signature}…")

    mean_reward = sum(r.get("reward", 0.0) for _, r in results) / max(len(results), 1)
    print(f"mean_reward={mean_reward:.3f}")


if __name__ == "__main__":
    main()
