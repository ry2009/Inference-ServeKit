from __future__ import annotations

from fastapi import FastAPI

from .policy_engine import run_policies
from .score_fns import score_code_task, score_generic, score_sql_task
from .signer import merkle_and_sign

app = FastAPI(title="Prime Verifier")


@app.post("/verify")
def verify(trace: dict):
    tools = trace.get("tools", [])
    task_scores = score_generic(trace)
    if tools:
        tool_name = tools[0].get("name", "")
        if tool_name.startswith("sql"):
            task_scores = score_sql_task(trace)
        elif tool_name.startswith("code"):
            task_scores = score_code_task(trace)

    policy_scores = run_policies(trace)
    reward = task_scores["task_score"] + task_scores.get("latency_penalty", 0.0) + policy_scores["policy_penalty"]
    root, signature = merkle_and_sign(trace, reward)

    return {
        "episode_id": trace.get("episode_id"),
        "scores": task_scores | policy_scores,
        "reward": reward,
        "checks": [
            {"check": "task_score", "pass": task_scores["task_score"] >= 0.5},
        ],
        "merkle_root": root,
        "signature": signature,
    }
