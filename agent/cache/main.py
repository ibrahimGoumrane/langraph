import json
from functools import wraps
from typing import Any, Dict
import redis

# In-memory cache using Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_tool_result(tool_name: str, args: Dict) -> tuple[bool, Any]:
    """
    Check if a tool result is cached.
    Returns: (is_cached, result)
    """
    cache_key = _make_cache_key(tool_name, args)
    cached_result = redis_client.get(cache_key)
    if cached_result is not None:
        return True, json.loads(cached_result)
    return False, None

def store_tool_result(tool_name: str, args: Dict, result: Any) -> None:
    """Store a tool result in cache"""
    cache_key = _make_cache_key(tool_name, args)
    redis_client.set(cache_key, json.dumps(result))

def _make_cache_key(tool_name: str, args: Dict) -> str:
    """Create a cache key from tool name and arguments"""
    args_str = json.dumps(args, sort_keys=True)
    return f"{tool_name}:{args_str}"

def clear_cache() -> None:
    """Clear all cached tool results"""
    redis_client.flushdb()

def get_cache_stats() -> Dict:
    """Get cache statistics"""
    return {
        "cached_results": redis_client.dbsize(),
        "cache_keys": list(redis_client.keys())
    }
