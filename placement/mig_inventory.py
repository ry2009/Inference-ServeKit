import json
import subprocess
from typing import List


def list_gpus() -> List[dict]:
    """Return GPU inventory using nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=json",
        ]
    )
    data = json.loads(out)
    return data.get("gpu", [])
