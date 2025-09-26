from cache.global_prefix_cache import GlobalPrefixCache


class DummyRedis:
    def __init__(self):
        self.store = {}

    def hset(self, key, mapping=None, **kwargs):
        data = mapping or kwargs
        bucket = self.store.setdefault(key, {})
        for field, value in data.items():
            key_bytes = field.encode() if isinstance(field, str) else field
            if isinstance(value, str):
                bucket[key_bytes] = value.encode()
            else:
                bucket[key_bytes] = value

    def hgetall(self, key):
        raw = self.store.get(key, {})
        return {k if isinstance(k, bytes) else k.encode(): v for k, v in raw.items()}

    def hincrby(self, key, field, amount):
        self.store.setdefault(key, {})[field.encode() if isinstance(field, str) else field] = amount


def test_cache_put_get(monkeypatch):
    cache = GlobalPrefixCache()
    dummy = DummyRedis()
    monkeypatch.setattr(cache, "redis", dummy)
    fp = b"abc"
    cache.put(fp, {"model": "test"}, node_id="node-a", tier="hbm")
    meta = cache.get(fp)
    assert meta["model"] == "test"
    assert meta["tier"] == "hbm"
