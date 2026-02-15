"""Rate limiting utilities for API calls.

Provides token bucket rate limiting to prevent API abuse and manage costs.
"""
import logging
import time
from threading import Lock
from dataclasses import dataclass
from typing import Optional
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 90000  # OpenAI default for gpt-4o-mini
    burst_multiplier: float = 1.5


class TokenBucket:
    """Token bucket rate limiter with burst support.
    
    This implementation allows for burst traffic while maintaining
    average rate limits over time.
    """
    
    def __init__(
        self,
        rate: float,
        capacity: Optional[float] = None,
        initial_tokens: Optional[float] = None
    ):
        """
        Initialize the token bucket.
        
        Args:
            rate: Token refill rate per second.
            capacity: Maximum bucket capacity (default: rate * 1.5).
            initial_tokens: Initial tokens (default: capacity).
        """
        self._rate = rate
        self._capacity = capacity or rate * 1.5
        self._tokens = initial_tokens if initial_tokens is not None else self._capacity
        self._last_update = time.monotonic()
        self._lock = Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_update = now
    
    def acquire(self, tokens: float = 1.0, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire.
            blocking: Whether to wait if tokens aren't available.
            timeout: Maximum time to wait (None = wait forever).
            
        Returns:
            True if tokens were acquired, False otherwise.
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._refill()
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                
                if not blocking:
                    return False
                
                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._rate
            
            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            time.sleep(min(wait_time, 0.1))  # Sleep in small increments
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class RateLimiter:
    """Multi-tier rate limiter for API calls.
    
    Enforces limits on:
    - Requests per minute
    - Requests per hour
    - Tokens per minute (for LLM APIs)
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.
        
        Args:
            config: Rate limit configuration.
        """
        self._config = config or RateLimitConfig()
        
        # Request rate limiters
        self._rpm_bucket = TokenBucket(
            rate=self._config.requests_per_minute / 60,
            capacity=self._config.requests_per_minute * self._config.burst_multiplier / 60
        )
        self._rph_bucket = TokenBucket(
            rate=self._config.requests_per_hour / 3600,
            capacity=self._config.requests_per_hour * self._config.burst_multiplier / 3600
        )
        
        # Token rate limiter
        self._tpm_bucket = TokenBucket(
            rate=self._config.tokens_per_minute / 60,
            capacity=self._config.tokens_per_minute * self._config.burst_multiplier / 60
        )
        
        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._blocked_requests = 0
        self._lock = Lock()
    
    def acquire(self, estimated_tokens: int = 1000, timeout: Optional[float] = 30.0) -> bool:
        """
        Acquire permission to make an API call.
        
        Args:
            estimated_tokens: Estimated tokens for this request.
            timeout: Maximum time to wait.
            
        Returns:
            True if request is allowed, False if rate limited.
        """
        # Check all rate limits
        if not self._rpm_bucket.acquire(1.0, blocking=True, timeout=timeout):
            logger.warning("Rate limited: requests per minute exceeded")
            with self._lock:
                self._blocked_requests += 1
            return False
        
        if not self._rph_bucket.acquire(1.0, blocking=True, timeout=timeout):
            logger.warning("Rate limited: requests per hour exceeded")
            with self._lock:
                self._blocked_requests += 1
            return False
        
        if not self._tpm_bucket.acquire(float(estimated_tokens), blocking=True, timeout=timeout):
            logger.warning("Rate limited: tokens per minute exceeded")
            with self._lock:
                self._blocked_requests += 1
            return False
        
        with self._lock:
            self._total_requests += 1
            self._total_tokens += estimated_tokens
        
        return True
    
    def report_actual_tokens(self, actual_tokens: int) -> None:
        """
        Report actual tokens used (for more accurate rate limiting).
        
        Args:
            actual_tokens: Actual tokens consumed by the request.
        """
        with self._lock:
            self._total_tokens = self._total_tokens - 1000 + actual_tokens  # Adjust estimate
    
    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "blocked_requests": self._blocked_requests,
                "available_rpm": self._rpm_bucket.available_tokens * 60,
                "available_tpm": self._tpm_bucket.available_tokens * 60,
            }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limited(estimated_tokens: int = 1000):
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        estimated_tokens: Estimated tokens for this call.
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            if not limiter.acquire(estimated_tokens):
                raise RateLimitExceededError(
                    f"Rate limit exceeded for {func.__name__}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""
    pass
