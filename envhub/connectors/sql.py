import sqlite3
import time


class SQLTool:
    """Execute SQL queries against a local SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def run(self, query: str):
        start = time.time()
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(query)
            rows = cur.fetchall()
        return {
            "ok": True,
            "rows": rows,
            "ms": int((time.time() - start) * 1000),
        }
