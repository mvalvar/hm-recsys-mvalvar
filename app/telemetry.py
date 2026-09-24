"""Módulo de Telemetría, Métricas de Rendimiento y Observabilidad.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Provee:
- Colector thread-safe de latencias (p50, p95, p99).
- Métricas de proporción de cold-start vs recomendaciones personalizadas.
- Exportador en formato estándar Prometheus para integración con Grafana / Kubernetes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np


class RecsysMetricsCollector:
    """Colector thread-safe de telemetría y métricas operacionales del recomendador."""

    _instance: RecsysMetricsCollector | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, max_samples: int = 10_000) -> None:
        self._lock = threading.Lock()
        self.request_count: int = 0
        self.cold_start_count: int = 0
        self.personalized_count: int = 0
        self.error_count: int = 0
        self.latencies: deque[float] = deque(maxlen=max_samples)
        self.start_time: float = time.time()

    @classmethod
    def get_instance(cls) -> RecsysMetricsCollector:
        """Retorna la instancia Singleton del colector de telemetría."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_request(
        self,
        latency_ms: float,
        is_cold_start: bool,
        is_error: bool = False,
    ) -> None:
        """Registra una petición completada de recomendación."""
        with self._lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            else:
                self.latencies.append(latency_ms)
                if is_cold_start:
                    self.cold_start_count += 1
                else:
                    self.personalized_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Calcula el resumen de telemetría con percentiles de latencia."""
        with self._lock:
            uptime_s = time.time() - self.start_time
            lat_array = np.array(self.latencies) if self.latencies else np.array([0.0])

            p50 = float(np.percentile(lat_array, 50)) if len(self.latencies) > 0 else 0.0
            p95 = float(np.percentile(lat_array, 95)) if len(self.latencies) > 0 else 0.0
            p99 = float(np.percentile(lat_array, 99)) if len(self.latencies) > 0 else 0.0

            total_valid = self.cold_start_count + self.personalized_count
            cold_start_ratio = (self.cold_start_count / total_valid) if total_valid > 0 else 0.0

            return {
                "uptime_seconds": round(uptime_s, 1),
                "total_requests": self.request_count,
                "personalized_requests": self.personalized_count,
                "cold_start_requests": self.cold_start_count,
                "cold_start_ratio": round(cold_start_ratio, 4),
                "errors_total": self.error_count,
                "latency_p50_ms": round(p50, 3),
                "latency_p95_ms": round(p95, 3),
                "latency_p99_ms": round(p99, 3),
            }

    def export_prometheus_format(self) -> str:
        """Exporta métricas en formato compatible con Prometheus."""
        summary = self.get_summary()
        lines = [
            "# HELP recsys_requests_total Total de solicitudes de recomendación procesadas",
            "# TYPE recsys_requests_total counter",
            f"recsys_requests_total {summary['total_requests']}",
            "# HELP recsys_cold_start_total Total de recomendaciones que utilizaron fallback por cold start",
            "# TYPE recsys_cold_start_total counter",
            f"recsys_cold_start_total {summary['cold_start_requests']}",
            "# HELP recsys_personalized_total Total de recomendaciones personalizadas servidas por LightGBM",
            "# TYPE recsys_personalized_total counter",
            f"recsys_personalized_total {summary['personalized_requests']}",
            "# HELP recsys_latency_p95_ms Latencia en percentil 95 en milisegundos",
            "# TYPE recsys_latency_p95_ms gauge",
            f"recsys_latency_p95_ms {summary['latency_p95_ms']}",
            "# HELP recsys_latency_p50_ms Latencia mediana (p50) en milisegundos",
            "# TYPE recsys_latency_p50_ms gauge",
            f"recsys_latency_p50_ms {summary['latency_p50_ms']}",
            "# HELP recsys_uptime_seconds Tiempo total de actividad del microservicio",
            "# TYPE recsys_uptime_seconds gauge",
            f"recsys_uptime_seconds {summary['uptime_seconds']}",
        ]
        return "\n".join(lines) + "\n"
