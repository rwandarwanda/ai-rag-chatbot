import time
import hashlib
from utils.config import CACHE_TTL, CACHE_MAX_SIZE

_cache = {}


def _make_key(query, model):
    raw = f"{query}::{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached(query, model="default"):
    key = _make_key(query, model)
    entry = _cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["timestamp"] > CACHE_TTL:
        del _cache[key]
        return None
    return entry["value"]


def set_cache(query, value, model="default"):
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_cache, key=lambda k: _cache[k]["timestamp"])
        del _cache[oldest_key]
    key = _make_key(query, model)
    _cache[key] = {
        "value": value,
        "timestamp": time.time()
    }


def clear_cache():
    _cache = {}


def cache_stats():
    return {
        "size": len(_cache),
        "max_size": CACHE_MAX_SIZE,
        "ttl": CACHE_TTL
    }
