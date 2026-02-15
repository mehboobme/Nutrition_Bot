"""Metrics and monitoring utilities for the RAG application.

Provides comprehensive metrics collection for:
- Request latency
- Token usage
- Cache performance
- Error rates
- Workflow execution
"""
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional
from collections import defaultdict
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A single metric measurement."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    count: int = 0
    total: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    
    @property
    def avg(self) -> float:
        """Calculate average latency."""
        return self.total / self.count if self.count > 0 else 0.0
    
    def record(self, value: float) -> None:
        """Record a latency measurement."""
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)


class MetricsCollector:
    """Centralized metrics collection and aggregation.
    
    Collects metrics for:
    - API latencies
    - Token usage
    - Cache hit rates
    - Error counts
    - Workflow step durations
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize the metrics collector.
        
        Args:
            retention_hours: Hours to retain detailed metrics.
        """
        self._retention = timedelta(hours=retention_hours)
        self._lock = Lock()
        
        # Latency metrics
        self._latencies: Dict[str, LatencyStats] = defaultdict(LatencyStats)
        
        # Counter metrics
        self._counters: Dict[str, int] = defaultdict(int)
        
        # Gauge metrics (current values)
        self._gauges: Dict[str, float] = {}
        
        # Recent measurements for time-series
        self._recent_measurements: List[MetricValue] = []
        
        # Error tracking
        self._errors: Dict[str, int] = defaultdict(int)
        self._last_error: Optional[Dict[str, Any]] = None
        
        # Start time for uptime calculation
        self._start_time = datetime.now()
    
    def record_latency(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a latency measurement.
        
        Args:
            name: Metric name (e.g., 'llm_response', 'retrieval').
            duration: Duration in seconds.
            labels: Optional labels for this measurement.
        """
        with self._lock:
            self._latencies[name].record(duration)
            self._recent_measurements.append(MetricValue(
                value=duration,
                timestamp=datetime.now(),
                labels={"metric": name, **(labels or {})}
            ))
            self._cleanup_old_measurements()
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name (e.g., 'requests_total', 'cache_hits').
            value: Amount to increment by.
        """
        with self._lock:
            self._counters[name] += value
    
    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Gauge name (e.g., 'active_sessions', 'cache_size').
            value: Current value.
        """
        with self._lock:
            self._gauges[name] = value
    
    def record_error(self, error_type: str, details: Optional[str] = None) -> None:
        """
        Record an error occurrence.
        
        Args:
            error_type: Type/category of error.
            details: Optional error details.
        """
        with self._lock:
            self._errors[error_type] += 1
            self._last_error = {
                "type": error_type,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"Error recorded: {error_type} - {details}")
    
    def _cleanup_old_measurements(self) -> None:
        """Remove measurements older than retention period."""
        cutoff = datetime.now() - self._retention
        self._recent_measurements = [
            m for m in self._recent_measurements
            if m.timestamp > cutoff
        ]
    
    @contextmanager
    def measure_latency(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager to measure operation latency.
        
        Args:
            name: Metric name.
            labels: Optional labels.
            
        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record_latency(name, duration, labels)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self._lock:
            uptime = datetime.now() - self._start_time
            
            return {
                "uptime_seconds": uptime.total_seconds(),
                "latencies": {
                    name: {
                        "count": stats.count,
                        "avg_ms": stats.avg * 1000,
                        "min_ms": stats.min * 1000 if stats.min != float('inf') else 0,
                        "max_ms": stats.max * 1000
                    }
                    for name, stats in self._latencies.items()
                },
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "errors": dict(self._errors),
                "last_error": self._last_error,
                "total_errors": sum(self._errors.values())
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring endpoints."""
        summary = self.get_summary()
        total_errors = summary["total_errors"]
        total_requests = self._counters.get("requests_total", 0)
        
        # Calculate error rate
        error_rate = total_errors / total_requests if total_requests > 0 else 0.0
        
        # Determine health status
        if error_rate > 0.1:  # >10% error rate
            status = "unhealthy"
        elif error_rate > 0.05:  # >5% error rate
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "uptime_seconds": summary["uptime_seconds"],
            "error_rate": error_rate,
            "total_requests": total_requests,
            "avg_latency_ms": summary["latencies"].get("request", {}).get("avg_ms", 0)
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._counters.clear()
            self._gauges.clear()
            self._recent_measurements.clear()
            self._errors.clear()
            self._last_error = None
            self._start_time = datetime.now()


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def timed(metric_name: str):
    """
    Decorator to automatically time a function.
    
    Args:
        metric_name: Name for the latency metric.
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.measure_latency(metric_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def counted(counter_name: str):
    """
    Decorator to automatically count function calls.
    
    Args:
        counter_name: Name for the counter metric.
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            metrics.increment_counter(counter_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                metrics.record_error(f"{counter_name}_error", str(e))
                raise
        return wrapper
    return decorator
