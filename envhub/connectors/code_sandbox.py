import subprocess
import tempfile
from pathlib import Path


class CodeSandbox:
    """Execute code snippets in a disposable sandbox."""

    def run(self, language: str, source: str) -> dict:
        if language.lower() != "python":
            return {"ok": False, "error": f"unsupported language: {language}"}

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(source)
            proc = subprocess.run(["python", str(script)], capture_output=True, text=True)
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "rc": proc.returncode,
        }
