"""
PROVIDER API Metrics
=====================
Tracks API request statistics for observability.
"""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    """Accumulated API request metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_latency: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.total_requests if self.total_requests else 0.0

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retried_requests": self.retried_requests,
            "avg_latency_ms": round(self.avg_latency * 1000, 1),
            "p95_latency_ms": round(self.p95_latency * 1000, 1),
            "min_latency_ms": round(self.min_latency * 1000, 1) if self.min_latency != float("inf") else 0,
            "max_latency_ms": round(self.max_latency * 1000, 1),
        }


class MetricsCollector:
    """Singleton metrics collector for API observability."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = APIMetrics()
        return cls._instance

    @property
    def metrics(self) -> APIMetrics:
        return self._metrics

    def track_request(self, latency: float, success: bool = True, retried: bool = False):
        """Record a single API request."""
        m = self._metrics
        m.total_requests += 1
        m.total_latency += latency
        m.min_latency = min(m.min_latency, latency)
        m.max_latency = max(m.max_latency, latency)

        # Keep last 1000 latencies for percentile calculation
        m.latencies.append(latency)
        if len(m.latencies) > 1000:
            m.latencies = m.latencies[-1000:]

        if success:
            m.successful_requests += 1
        else:
            m.failed_requests += 1
        if retried:
            m.retried_requests += 1

    def log_health_check(self):
        """Log current metrics as a health check."""
        m = self._metrics
        logger.info(
            f"API Health: {m.total_requests} requests "
            f"({m.successful_requests} ok, {m.failed_requests} failed, {m.retried_requests} retried) | "
            f"Latency: avg={m.avg_latency*1000:.0f}ms p95={m.p95_latency*1000:.0f}ms"
        )

    def reset(self):
        """Reset all metrics."""
        self._metrics = APIMetrics()
