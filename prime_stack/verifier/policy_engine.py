from __future__ import annotations

from typing import Dict


POLICY_DEFAULTS = {
    "egress_blocked": True,
    "sandbox_profile": "default",
    "max_tool_walltime_ms": 30_000,
}


def run_policies(trace: Dict) -> Dict[str, float]:
    """Apply lightweight policy checks to an episode trace."""

    meta = trace.get("policy_meta", {})
    penalties = 0.0

    if not meta.get("egress_blocked", POLICY_DEFAULTS["egress_blocked"]):
        penalties -= 0.5

    sandbox = meta.get("sandbox_profile", POLICY_DEFAULTS["sandbox_profile"])
    if sandbox not in {"seccomp_sql_v1", "seccomp_code_v1", "default"}:
        penalties -= 0.2

    for tool in trace.get("tools", []):
        elapsed = tool.get("result", {}).get("ms", 0)
        if elapsed and elapsed > POLICY_DEFAULTS["max_tool_walltime_ms"]:
            penalties -= 0.1

    return {"policy_penalty": penalties}
