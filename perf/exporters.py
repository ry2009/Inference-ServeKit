from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "tokens",
    "latency",
    "queue_depth",
    "cache_hit",
    "cache_miss",
    "kv_bytes",
]

tokens = Counter("primerl_tokens_total", "Tokens generated", ["phase", "model"])
latency = Histogram(
    "primerl_request_latency_seconds",
    "Request latency by route/model",
    ["route", "model"],
)
queue_depth = Gauge("primerl_queue_depth", "Requests queued", ["model"])
cache_hit = Counter("primerl_prefix_cache_hits_total", "Prefix cache hits", ["model"])
cache_miss = Counter("primerl_prefix_cache_misses_total", "Prefix cache misses", ["model"])
kv_bytes = Gauge("primerl_kv_resident_bytes", "Resident KV bytes", ["model"])
