from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

_GRAMMAR_DIR = Path(__file__).resolve().parent.parent / "envhub" / "grammars"


def load(grammar_id: str) -> Any:
    """Load a grammar schema by id from the shared JSON bundle."""
    gram_path = _GRAMMAR_DIR / "tool_schemas.json"
    data = orjson.loads(gram_path.read_bytes())
    try:
        return data[grammar_id]
    except KeyError as exc:
        raise ValueError(f"Unknown grammar id: {grammar_id}") from exc


def list_grammars() -> list[str]:
    gram_path = _GRAMMAR_DIR / "tool_schemas.json"
    data = orjson.loads(gram_path.read_bytes())
    return sorted(data.keys())
