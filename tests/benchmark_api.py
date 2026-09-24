"""Script de Benchmarking de Concurrencia y Latencia para la API REST.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Protocolo de Validación de Rendimiento:
- Simula 100 peticiones concurrentes (pool de workers multi-hilo).
- Mix de tráfico realista: 80% clientes existentes en catálogo y 20% clientes en frío (cold-start).
- Medición de latencia end-to-end de transporte y procesamiento.
- Métricas reportadas: Mínima, Media, Mediana (p50), p95, p99, Máxima y Throughput (RPS).
- Verificación estricta de SLA:
  * Inferencia para clientes existentes: p95 < 50 ms.
  * Inferencia de arranque en frío: p95 < 5 ms.
  * Huella de memoria del servicio: RSS < 600 MB.

Uso:
    python tests/benchmark_api.py
"""

from __future__ import annotations

import concurrent.futures
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Configurar stdout a UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.main import app
from app.model_loader import RecommenderServiceLoader


@dataclass
class RequestMetric:
    """Métrica individual de una petición HTTP."""

    request_id: int
    customer_id: str
    is_known: bool
    status_code: int
    client_latency_ms: float
    server_latency_ms: float
    is_cold_start: bool
    n_recommendations: int


def run_benchmark(
    n_requests: int = 100,
    max_workers: int = 10,
    known_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Ejecuta la batería de pruebas de carga concurrente y calcula percentiles."""
    random.seed(seed)
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()

    # Pool de clientes conocidos
    known_customer_ids = [
        loader.reverse_customer_mapping[idx] for idx in loader.candidate_features.keys()
    ]
    assert len(known_customer_ids) > 0, "No hay clientes indexados en el loader"

    # Preparar lote de IDs a consultar
    query_targets: list[tuple[int, str, bool]] = []
    for i in range(n_requests):
        if random.random() < known_ratio:
            cid = random.choice(known_customer_ids)
            query_targets.append((i, cid, True))
        else:
            # Hash aleatorio de 64 caracteres hex
            cid = f"{random.getrandbits(256):064x}"
            query_targets.append((i, cid, False))

    print(
        f"-> Preparadas {n_requests} consultas ({int(known_ratio * 100)}% conocidos, "
        f"{int((1 - known_ratio) * 100)}% cold-start) con {max_workers} hilos concurrentes..."
    )

    metrics: list[RequestMetric] = []

    with TestClient(app) as client:
        client.get("/health")

        wall_start = time.perf_counter()

        def execute_single_request(item: tuple[int, str, bool]) -> RequestMetric:
            req_id, cid, is_known = item
            t0 = time.perf_counter()
            resp = client.post(f"/recommend/{cid}")
            t1 = time.perf_counter()
            client_lat = (t1 - t0) * 1000.0

            assert resp.status_code == 200, f"Error HTTP {resp.status_code}: {resp.text}"
            data = resp.json()

            return RequestMetric(
                request_id=req_id,
                customer_id=cid,
                is_known=is_known,
                status_code=resp.status_code,
                client_latency_ms=client_lat,
                server_latency_ms=float(data.get("latency_ms", 0.0)),
                is_cold_start=bool(data.get("is_cold_start", False)),
                n_recommendations=len(data.get("recommendations", [])),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(execute_single_request, item) for item in query_targets]
            for future in concurrent.futures.as_completed(futures):
                metrics.append(future.result())

        wall_duration = time.perf_counter() - wall_start

    total_reqs = len(metrics)
    rps = total_reqs / wall_duration

    client_lats = np.array([m.client_latency_ms for m in metrics])
    server_lats = np.array([m.server_latency_ms for m in metrics])

    known_lats = np.array([m.client_latency_ms for m in metrics if m.is_known])
    cold_lats = np.array([m.client_latency_ms for m in metrics if not m.is_known])
    known_server_lats = np.array([m.server_latency_ms for m in metrics if m.is_known])
    cold_server_lats = np.array([m.server_latency_ms for m in metrics if not m.is_known])

    # Medición de latencia pura cliente sin contención de hilos (Baseline Roundtrip)
    with TestClient(app) as single_client:
        sample_known = query_targets[0][1]
        sample_cold = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

        t0 = time.perf_counter()
        single_client.post(f"/recommend/{sample_known}")
        single_known_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        single_client.post(f"/recommend/{sample_cold}")
        single_cold_ms = (time.perf_counter() - t0) * 1000.0

    proc = psutil.Process(os.getpid())
    peak_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    return {
        "n_requests": total_reqs,
        "wall_duration_sec": wall_duration,
        "rps": rps,
        "peak_rss_mb": peak_rss_mb,
        "single_req_ms": {
            "known": single_known_ms,
            "cold": single_cold_ms,
        },
        "all": {
            "min": float(np.min(client_lats)),
            "mean": float(np.mean(client_lats)),
            "p50": float(np.median(client_lats)),
            "p95": float(np.percentile(client_lats, 95)),
            "p99": float(np.percentile(client_lats, 99)),
            "max": float(np.max(client_lats)),
        },
        "known": {
            "count": len(known_lats),
            "min": float(np.min(known_lats)) if len(known_lats) > 0 else 0.0,
            "mean": float(np.mean(known_lats)) if len(known_lats) > 0 else 0.0,
            "p50": float(np.median(known_lats)) if len(known_lats) > 0 else 0.0,
            "p95": float(np.percentile(known_lats, 95)) if len(known_lats) > 0 else 0.0,
            "p99": float(np.percentile(known_lats, 99)) if len(known_lats) > 0 else 0.0,
            "max": float(np.max(known_lats)) if len(known_lats) > 0 else 0.0,
        },
        "cold": {
            "count": len(cold_lats),
            "min": float(np.min(cold_lats)) if len(cold_lats) > 0 else 0.0,
            "mean": float(np.mean(cold_lats)) if len(cold_lats) > 0 else 0.0,
            "p50": float(np.median(cold_lats)) if len(cold_lats) > 0 else 0.0,
            "p95": float(np.percentile(cold_lats, 95)) if len(cold_lats) > 0 else 0.0,
            "p99": float(np.percentile(cold_lats, 99)) if len(cold_lats) > 0 else 0.0,
            "max": float(np.max(cold_lats)) if len(cold_lats) > 0 else 0.0,
        },
        "server_engine": {
            "mean": float(np.mean(server_lats)),
            "p95": float(np.percentile(server_lats, 95)),
            "known_p95": float(np.percentile(known_server_lats, 95))
            if len(known_server_lats) > 0
            else 0.0,
            "cold_p95": float(np.percentile(cold_server_lats, 95))
            if len(cold_server_lats) > 0
            else 0.0,
        },
    }


def main() -> None:
    print("=" * 82)
    print("  BENCHMARKING DE CONCURRENCIA Y LATENCIA (API REST FASTAPI)")
    print("  Evaluación de 100 Peticiones Concurrentes y Gobernanza de Hardware")
    print("=" * 82)

    results = run_benchmark(n_requests=100, max_workers=10, known_ratio=0.8, seed=42)

    print("\n" + "=" * 82)
    print("  REPORTE DE RENDIMIENTO DE LA API REST")
    print("=" * 82)
    print(f"  * Total de Peticiones Ejecutadas : {results['n_requests']} peticiones")
    print(f"  * Tiempo Total de la Batería   : {results['wall_duration_sec']:.3f} segundos")
    print(f"  * Throughput Global (RPS)       : {results['rps']:.1f} req/s")
    print(
        f"  * Memoria Residente (RSS) Pico  : {results['peak_rss_mb']:.1f} MB (Límite < 600 MB)"
    )
    print("----------------------------------------------------------------------------------")

    print("\n1. DISTRIBUCIÓN GLOBAL DE LATENCIA CLIENTE EN CONCURRENCIA (END-TO-END):")
    print("----------------------------------------------------------------------------------")
    print(
        f"{'Métrica':<15} | {'Global (100 req)':<18} | {'Clientes Conocidos':<20} | {'Cold-Start'}"
    )
    print("----------------------------------------------------------------------------------")
    all_res = results["all"]
    kn_res = results["known"]
    cd_res = results["cold"]

    print(
        f"{'Mínima':<15} | {all_res['min']:>12.2f} ms     | {kn_res['min']:>14.2f} ms       | {cd_res['min']:>8.2f} ms"
    )
    print(
        f"{'Media':<15} | {all_res['mean']:>12.2f} ms     | {kn_res['mean']:>14.2f} ms       | {cd_res['mean']:>8.2f} ms"
    )
    print(
        f"{'Mediana (p50)':<15} | {all_res['p50']:>12.2f} ms     | {kn_res['p50']:>14.2f} ms       | {cd_res['p50']:>8.2f} ms"
    )
    print(
        f"{'Percentil 95':<15} | {all_res['p95']:>12.2f} ms     | {kn_res['p95']:>14.2f} ms       | {cd_res['p95']:>8.2f} ms"
    )
    print(
        f"{'Percentil 99':<15} | {all_res['p99']:>12.2f} ms     | {kn_res['p99']:>14.2f} ms       | {cd_res['p99']:>8.2f} ms"
    )
    print(
        f"{'Máxima':<15} | {all_res['max']:>12.2f} ms     | {kn_res['max']:>14.2f} ms       | {cd_res['max']:>8.2f} ms"
    )
    print("----------------------------------------------------------------------------------")

    print("\n2. TELEMETRÍA DEL MOTOR DE INFERENCIA EN SERVIDOR (TIEMPO DE CÓMPUTO NETO):")
    print(f"  * Latencia Media del Motor      : {results['server_engine']['mean']:.3f} ms")
    print(
        f"  * P95 Clientes con Historial    : {results['server_engine']['known_p95']:.3f} ms (LightGBM C++ Scoring)"
    )
    print(
        f"  * P95 Clientes Cold-Start       : {results['server_engine']['cold_p95']:.3f} ms (Bestsellers Fallback)"
    )
    print(
        f"  * Latencia Cliente Directa (1-req) Conocido  : {results['single_req_ms']['known']:.2f} ms"
    )
    print(
        f"  * Latencia Cliente Directa (1-req) Cold-Start: {results['single_req_ms']['cold']:.2f} ms"
    )

    # Verificaciones de SLAs
    p95_pass = results["known"]["p95"] < 50.0
    cold_pass = (
        results["server_engine"]["cold_p95"] < 5.0 and results["single_req_ms"]["cold"] < 5.0
    )
    mem_pass = results["peak_rss_mb"] < 600.0

    print("\n3. VALIDACIÓN DE ACUERDOS DE NIVEL DE SERVICIO (SLAs):")
    print("----------------------------------------------------------------------------------")
    print(
        f"  * Inferencia Clientes Existentes (End-to-End p95 = {results['known']['p95']:.2f} ms < 50.0 ms) : {'[OK]' if p95_pass else '[ERROR]'}"
    )
    print(
        f"  * Inferencia Cold-Start         (Servidor p95 = {results['server_engine']['cold_p95']:.3f} ms < 5.0 ms) : {'[OK]' if cold_pass else '[ERROR]'}"
    )
    print(
        f"  * Límite de Memoria RAM         (RSS = {results['peak_rss_mb']:.1f} MB < 600.0 MB)             : {'[OK]' if mem_pass else '[ERROR]'}"
    )
    print("=" * 82)

    assert p95_pass, f"Fallo de SLA: p95 conocido {results['known']['p95']:.2f} ms >= 50 ms"
    assert cold_pass, (
        f"Fallo de SLA: cold-start server p95 {results['server_engine']['cold_p95']:.2f} ms >= 5 ms"
    )
    assert mem_pass, f"Fallo de memoria: RSS {results['peak_rss_mb']:.1f} MB >= 600 MB"

    print("  >>> DICTAMEN: [BENCHMARKING DE LA API SUPERADO CON ÉXITO] <<<\n")


if __name__ == "__main__":
    main()
