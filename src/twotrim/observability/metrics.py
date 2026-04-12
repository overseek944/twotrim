"""Metrics collection and Prometheus exposition.

Tracks per-request and aggregate metrics for token compression.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from twotrim.types import AggregateMetrics, RequestMetrics, estimate_cost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (lazy-loaded)
# ---------------------------------------------------------------------------
_prom_initialized = False
_prom_counters: dict[str, Any] = {}
_prom_histograms: dict[str, Any] = {}
_prom_gauges: dict[str, Any] = {}


def _init_prometheus() -> None:
    """Initialize Prometheus metrics."""
    global _prom_initialized
    if _prom_initialized:
        return
    try:
        from prometheus_client import Counter, Gauge, Histogram

        _prom_counters["requests_total"] = Counter(
            "twotrim_requests_total", "Total requests processed",
            ["model", "mode", "cache_hit"],
        )
        _prom_counters["tokens_original"] = Counter(
            "twotrim_tokens_original_total", "Total original tokens",
        )
        _prom_counters["tokens_compressed"] = Counter(
            "twotrim_tokens_compressed_total", "Total compressed tokens",
        )
        _prom_counters["tokens_saved"] = Counter(
            "twotrim_tokens_saved_total", "Total tokens saved",
        )
        _prom_counters["cost_saved"] = Counter(
            "twotrim_cost_saved_usd_total", "Total estimated cost saved (USD)",
        )
        _prom_histograms["compression_ratio"] = Histogram(
            "twotrim_compression_ratio", "Compression ratio distribution",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        _prom_histograms["compression_time_ms"] = Histogram(
            "twotrim_compression_time_ms", "Compression latency in ms",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
        )
        _prom_histograms["quality_score"] = Histogram(
            "twotrim_quality_score", "Quality score distribution",
            buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
        )
        _prom_gauges["cache_hit_rate"] = Gauge(
            "twotrim_cache_hit_rate", "Rolling cache hit rate",
        )
        _prom_gauges["active_sessions"] = Gauge(
            "twotrim_active_sessions", "Active memory sessions",
        )

        _prom_initialized = True
        logger.info("Prometheus metrics initialized")
    except Exception as e:
        logger.debug("Prometheus metrics not available: %s", e)
        _prom_initialized = True  # Don't retry


class MetricsCollector:
    """Collect and aggregate compression metrics."""

    def __init__(self, window_size: int = 1000) -> None:
        self._history: deque[RequestMetrics] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._total = AggregateMetrics()

    def record(self, metrics: RequestMetrics) -> None:
        """Record metrics for a single request."""
        _init_prometheus()

        with self._lock:
            self._history.append(metrics)

            # Update totals
            self._total.total_requests += 1
            self._total.total_tokens_original += metrics.original_tokens
            self._total.total_tokens_compressed += metrics.compressed_tokens
            self._total.total_tokens_saved += metrics.tokens_saved
            self._total.total_cost_saved_usd += metrics.estimated_cost_saved_usd

            for strategy in metrics.strategies_applied:
                self._total.strategy_usage[strategy] = (
                    self._total.strategy_usage.get(strategy, 0) + 1
                )

        # Update Prometheus
        if _prom_counters:
            try:
                _prom_counters["requests_total"].labels(
                    model=metrics.model,
                    mode="unknown",
                    cache_hit=str(metrics.cache_hit),
                ).inc()
                _prom_counters["tokens_original"].inc(metrics.original_tokens)
                _prom_counters["tokens_compressed"].inc(metrics.compressed_tokens)
                _prom_counters["tokens_saved"].inc(metrics.tokens_saved)
                _prom_counters["cost_saved"].inc(metrics.estimated_cost_saved_usd)
                _prom_histograms["compression_ratio"].observe(metrics.compression_ratio)
                _prom_histograms["compression_time_ms"].observe(metrics.compression_time_ms)
                if metrics.quality_score is not None:
                    _prom_histograms["quality_score"].observe(metrics.quality_score)
            except Exception:
                pass

    def get_aggregate(self) -> AggregateMetrics:
        """Get aggregate metrics."""
        with self._lock:
            total = self._total.model_copy()
            if total.total_requests > 0:
                total.avg_compression_ratio = (
                    1 - total.total_tokens_compressed / max(total.total_tokens_original, 1)
                )
                cache_hits = sum(1 for m in self._history if m.cache_hit)
                total.cache_hit_rate = cache_hits / len(self._history) if self._history else 0

                quality_scores = [m.quality_score for m in self._history if m.quality_score is not None]
                if quality_scores:
                    total.avg_quality_score = sum(quality_scores) / len(quality_scores)

                compression_times = [m.compression_time_ms for m in self._history]
                if compression_times:
                    total.avg_compression_time_ms = sum(compression_times) / len(compression_times)

            return total

    def get_recent(self, n: int = 100) -> list[RequestMetrics]:
        """Get the N most recent request metrics."""
        with self._lock:
            items = list(self._history)
            return items[-n:]

    def create_request_metrics(
        self,
        request_id: str,
        request_type: str,
        model: str,
        original_tokens: int,
        compressed_tokens: int,
        compression_time_ms: float,
        total_time_ms: float,
        cache_hit: bool,
        strategies: list[str],
        quality_score: float | None = None,
    ) -> RequestMetrics:
        """Create a RequestMetrics object with cost estimation."""
        tokens_saved = original_tokens - compressed_tokens
        cost_saved = estimate_cost(model, tokens_saved)

        from twotrim.types import RequestType
        try:
            req_type = RequestType(request_type)
        except ValueError:
            req_type = RequestType.CHAT_COMPLETION

        return RequestMetrics(
            request_id=request_id,
            request_type=req_type,
            model=model,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1 - (compressed_tokens / max(original_tokens, 1)),
            tokens_saved=tokens_saved,
            estimated_cost_saved_usd=cost_saved,
            compression_time_ms=compression_time_ms,
            total_time_ms=total_time_ms,
            cache_hit=cache_hit,
            strategies_applied=strategies,
            quality_score=quality_score,
        )


# Module singleton
_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
