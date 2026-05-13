"""MetricsCollector: lightweight in-memory metrics bus.

Decoupled from any specific monitoring system. The collector exposes:
    - `incr(name, by=1)`        — counter
    - `gauge(name, value)`      — last-write-wins gauge
    - `observe(name, value)`    — histogram-ish (stores raw values)
    - `snapshot()`              — current view as a dict

Plug a function into `on_emit` to push to Prometheus, Datadog, stdout, etc.
We don't add a dependency.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MetricsCollector:
    on_emit: Callable[[str, str, float], None] | None = None
    _counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _gauges: dict[str, float] = field(default_factory=dict)
    _observations: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def incr(self, name: str, by: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += by
        self._maybe_emit("counter", name, by)

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value
        self._maybe_emit("gauge", name, value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._observations[name].append(value)
        self._maybe_emit("observation", name, value)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            histograms = {
                k: _summarize(values) for k, values in self._observations.items()
            }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histograms,
            }

    def _maybe_emit(self, kind: str, name: str, value: float) -> None:
        if self.on_emit is None:
            return
        try:
            self.on_emit(kind, name, value)
        except Exception:  # noqa: BLE001 — emitters should never break the producer
            pass


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    sorted_values = sorted(values)
    n = len(sorted_values)
    mean = sum(sorted_values) / n
    return {
        "n": float(n),
        "mean": float(mean),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "p50": float(sorted_values[n // 2]),
        "p95": float(sorted_values[min(n - 1, math.floor(0.95 * n))]),
    }
