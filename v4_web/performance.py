"""Small helpers for recording request timing metrics."""

from __future__ import annotations


def record_request_timing(metrics: dict, elapsed_seconds: float) -> None:
    """Update aggregate request timing metrics in place."""
    elapsed_ms = max(0.0, elapsed_seconds * 1000)
    metrics["duration_ms_total"] += elapsed_ms
    metrics["duration_ms_max"] = max(metrics["duration_ms_max"], elapsed_ms)
    if elapsed_ms >= 1000:
        metrics["slow_requests"] += 1

