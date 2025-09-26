from __future__ import annotations

import json
from pathlib import Path

import httpx


def load_task(path: str) -> dict:
    return json.loads(json.dumps(_yaml_to_dict(Path(path).read_text())))


def _yaml_to_dict(text: str) -> dict:
    import yaml

    return yaml.safe_load(text)


def run_task(task_path: str, verifier_url: str, model: str):
    task = load_task(task_path)
    trace = {
        "episode_id": task["task"],
        "model": model,
        "prompt_fp": "",
        "tokens": "",
        "tools": [],
        "meta": task,
    }
    response = httpx.post(f"{verifier_url}/verify", json=trace, timeout=30)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--verifier", default="http://localhost:8080")
    parser.add_argument("--model", default="latest")
    args = parser.parse_args()
    print(run_task(args.task, args.verifier, args.model))
