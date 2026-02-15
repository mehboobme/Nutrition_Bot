"""Caching utilities for the RAG application.

Provides LRU caching for LLM responses and retrieval results to reduce
API costs and improve response times.
"""
import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar
from collections import OrderedDict
from threading import Lock
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry with TTL support."""
    value: Any
    created_at: float
    ttl: float
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl <= 0:
            return False  # No expiration
        return time.time() - self.created_at > self.ttl


class LRUCache:
    """Thread-safe LRU cache with TTL support and statistics.
    
    Features:
    - Configurable max size and TTL
    - Thread-safe operations
    - Hit/miss statistics
    - Automatic eviction of expired entries
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store.
            default_ttl: Default time-to-live in seconds (0 = no expiration).
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a hash key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value or None if not found/expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds (None = use default).
        """
        with self._lock:
            if ttl is None:
                ttl = self._default_ttl
            
            # Remove oldest entries if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl
            )
    
    def invalidate(self, key: str) -> bool:
        """
        Remove a specific entry from the cache.
        
        Args:
            key: The cache key to remove.
            
        Returns:
            True if the key was found and removed, False otherwise.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed.
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "default_ttl": self._default_ttl
            }


# Global cache instances
_llm_cache = LRUCache(max_size=500, default_ttl=3600)  # 1 hour TTL
_retrieval_cache = LRUCache(max_size=200, default_ttl=1800)  # 30 min TTL


def get_llm_cache() -> LRUCache:
    """Get the global LLM response cache."""
    return _llm_cache


def get_retrieval_cache() -> LRUCache:
    """Get the global retrieval cache."""
    return _retrieval_cache


def cached_llm_call(ttl: Optional[float] = None):
    """
    Decorator to cache LLM call results.
    
    Args:
        ttl: Time-to-live in seconds for cached results.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache = get_llm_cache()
            key = cache._make_key(func.__name__, *args, **kwargs)
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Call function
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator


def cached_retrieval(ttl: Optional[float] = None):
    """
    Decorator to cache retrieval results.
    
    Args:
        ttl: Time-to-live in seconds for cached results.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache = get_retrieval_cache()
            key = cache._make_key(func.__name__, *args, **kwargs)
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Retrieval cache hit for {func.__name__}")
                return cached_result
            
            # Call function
            logger.debug(f"Retrieval cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator
