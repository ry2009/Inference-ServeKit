import time
from typing import Optional

import orjson
import redis
from redis.exceptions import RedisError


class GlobalPrefixCache:
    """Redis-backed prefix cache storing prompt metadata."""

    def __init__(self, url: str = "redis://localhost:6379/0"):
        self.redis = redis.Redis.from_url(url)

    def put(self, fingerprint: bytes, meta: dict, node_id: str | None = None, tier: str = "hbm") -> None:
        key = f"pf:{fingerprint.hex()}"
        payload = {
            "meta": orjson.dumps(meta),
            "ts": time.time(),
            "tier": tier,
        }
        if node_id:
            payload["nodes"] = orjson.dumps([node_id])
        try:
            self.redis.hset(key, mapping=payload)
        except RedisError:
            return

    def get(self, fingerprint: bytes) -> Optional[dict]:
        key = f"pf:{fingerprint.hex()}"
        try:
            result = self.redis.hgetall(key)
        except RedisError:
            return None
        if not result:
            return None
        meta = orjson.loads(result[b"meta"])
        meta["tier"] = result.get(b"tier", b"").decode() if result.get(b"tier") else None
        if b"nodes" in result:
            meta["nodes"] = orjson.loads(result[b"nodes"])
        self.redis.hincrby(key, "hits", 1)
        return meta

    def register_node(self, fingerprint: bytes, node_id: str):
        key = f"pf:{fingerprint.hex()}"
        try:
            self.redis.hset(key, "nodes", orjson.dumps([node_id]))
        except RedisError:
            return
