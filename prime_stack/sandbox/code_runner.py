from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict

SANDBOX_PROFILES = {
    "seccomp_code_v1": {
        "cmd": ["python"],
    },
    "seccomp_sql_v1": {
        "cmd": ["python"],
    },
}


def run_snippet(language: str, source: str, profile: str = "seccomp_code_v1", timeout: int = 5) -> Dict:
    if language.lower() != "python":
        return {"ok": False, "rc": 1, "stdout": "", "stderr": "unsupported language", "ms": 0}

    profile_cfg = SANDBOX_PROFILES.get(profile, SANDBOX_PROFILES["seccomp_code_v1"])
    interpreter = profile_cfg["cmd"]

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "snippet.py"
        script_path.write_text(source)
        start = time.time()
        proc = subprocess.run(
            interpreter + [str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = int((time.time() - start) * 1000)

    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ms": elapsed,
    }
