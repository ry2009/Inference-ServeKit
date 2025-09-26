SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"const": "sql"},
        "query": {"type": "string"},
        "result": {"type": "object"},
    },
}

CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"const": "code"},
        "stdout": {"type": "string"},
    },
}
