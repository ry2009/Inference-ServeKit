from __future__ import annotations

from typing import Dict


def score_sql_task(trace: Dict) -> Dict[str, float]:
    meta = trace.get("meta", {})
    expected = meta.get("expected_rows")
    result_rows = trace.get("tools", [{}])[0].get("result", {}).get("rows")
    hit = expected is not None and result_rows == expected
    latency_ms = trace.get("tools", [{}])[0].get("result", {}).get("ms", 0)
    return {
        "task_score": 1.0 if hit else 0.0,
        "latency_penalty": -0.001 * latency_ms,
    }


def score_code_task(trace: Dict) -> Dict[str, float]:
    tool = trace.get("tools", [{}])[0]
    result = tool.get("result", {})
    rc = result.get("rc", 1)
    latency_ms = result.get("ms", 0)
    return {
        "task_score": 1.0 if rc == 0 else 0.0,
        "latency_penalty": -0.001 * latency_ms,
    }


def score_generic(trace: Dict) -> Dict[str, float]:
    return {"task_score": 0.0, "latency_penalty": 0.0}
