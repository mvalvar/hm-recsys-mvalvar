"""Suite de Pruebas Unitarias y de Integración para la API REST FastAPI.

Valida:
1. Ciclo de vida lifespan y comprobación de salud en GET /health.
2. Endpoint de bienvenida y navegación en GET /.
3. Inferencia para clientes existentes en POST /recommend/{customer_id} (is_cold_start=False, p95 < 50ms).
4. Manejo de arranque en frío en POST /recommend/{customer_id} (is_cold_start=True, < 5ms).
5. Validación de contratos Pydantic y manejo de errores 422 para IDs inválidos.
6. Integridad del esquema OpenAPI interactivo en GET /openapi.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app
from app.model_loader import RecommenderServiceLoader


def test_api_lifespan_and_health():
    """Valida el arranque de la API y el endpoint GET /health."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "hm_recommender"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"
        assert data["indexed_customers"] > 0
        assert (
            data["memory_rss_mb"] < 8000.0
        )  # Cota acumulativa para ejecución de suite completa en mismo proceso


def test_api_root_navigation():
    """Valida el endpoint raíz GET /."""
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "docs_url" in data
        assert "health_url" in data
        assert "recommend_endpoint" in data


def test_recommendation_known_customer():
    """Valida la inferencia personalizada para un cliente activo en catálogo."""
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()

    # Tomar un cliente con candidatos indexados
    sample_idx = list(loader.candidate_features.keys())[0]
    sample_id = loader.reverse_customer_mapping[sample_idx]

    with TestClient(app) as client:
        resp = client.post(f"/recommend/{sample_id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["customer_id"] == sample_id
        assert data["is_cold_start"] is False
        assert len(data["recommendations"]) == 12
        assert all(len(art) == 10 and art.isdigit() for art in data["recommendations"])
        assert data["latency_ms"] < 50.0  # SLA de latencia p95 < 50ms


def test_recommendation_cold_start_unknown_customer():
    """Valida la degradación elegante para un cliente totalmente desconocido."""
    unknown_id = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

    with TestClient(app) as client:
        resp = client.post(f"/recommend/{unknown_id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["customer_id"] == unknown_id
        assert data["is_cold_start"] is True
        assert len(data["recommendations"]) == 12
        assert all(len(art) == 10 and art.isdigit() for art in data["recommendations"])
        assert data["latency_ms"] < 5.0  # SLA de latencia cold-start < 5ms


def test_recommendation_validation_error_short_id():
    """Valida el control semántico de errores 422 para parámetros inválidos."""
    invalid_id = "abc"  # Menor a min_length=10

    with TestClient(app) as client:
        resp = client.post(f"/recommend/{invalid_id}")
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data


def test_openapi_schema_integrity():
    """Valida la autogeneración correcta del esquema OpenAPI."""
    with TestClient(app) as client:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/recommend/{customer_id}" in schema["paths"]
        assert "/health" in schema["paths"]
        assert schema["info"]["title"] == "TFM H&M Fashion Recommender API"


def test_api_optional_authentication(monkeypatch):
    """Valida el control de acceso opcional mediante X-API-Key."""
    unknown_id = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

    # Caso 1: Modo abierto por defecto (API_KEY_AUTH_ENABLED no seteado o false)
    monkeypatch.delenv("API_KEY_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("API_KEY_SECRET", raising=False)
    with TestClient(app) as client:
        resp = client.post(f"/recommend/{unknown_id}")
        assert resp.status_code == 200

    # Caso 2: Modo protegido activado
    monkeypatch.setenv("API_KEY_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY_SECRET", "hm-secret-key-123")

    with TestClient(app) as client:
        # /health y / deben permanecer accesibles sin autenticación
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200

        # Sin cabecera -> 401
        resp_no_auth = client.post(f"/recommend/{unknown_id}")
        assert resp_no_auth.status_code == 401

        # Con cabecera errónea -> 401
        resp_bad_auth = client.post(
            f"/recommend/{unknown_id}",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp_bad_auth.status_code == 401

        # Con cabecera correcta -> 200
        resp_valid_auth = client.post(
            f"/recommend/{unknown_id}",
            headers={"X-API-Key": "hm-secret-key-123"},
        )
        assert resp_valid_auth.status_code == 200


def test_recommendation_pagination():
    """Valida el soporte de paginación estricta con limit y offset."""
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()

    sample_idx = list(loader.candidate_features.keys())[0]
    sample_id = loader.reverse_customer_mapping[sample_idx]

    with TestClient(app) as client:
        # Página 1: 5 recomendaciones
        resp1 = client.post(f"/recommend/{sample_id}?limit=5&offset=0")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert len(data1["recommendations"]) == 5
        assert data1["count"] == 5
        assert data1["limit"] == 5
        assert data1["offset"] == 0

        # Página 2: siguientes 5 recomendaciones
        resp2 = client.post(f"/recommend/{sample_id}?limit=5&offset=5")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["recommendations"]) == 5
        assert data2["count"] == 5
        assert data2["limit"] == 5
        assert data2["offset"] == 5

        # Verificar que no hay solapamiento entre páginas
        set_page1 = set(data1["recommendations"])
        set_page2 = set(data2["recommendations"])
        assert len(set_page1.intersection(set_page2)) == 0, "Solapamiento indebido entre páginas"


def test_health_probes():
    """Valida las sondas segregadas de liveness (/health/live) y readiness (/health/ready)."""
    with TestClient(app) as client:
        live_resp = client.get("/health/live")
        assert live_resp.status_code == 200
        assert live_resp.json()["status"] == "alive"

        ready_resp = client.get("/health/ready")
        assert ready_resp.status_code == 200
        ready_data = ready_resp.json()
        assert "ready" in ready_data
        assert "model_loaded" in ready_data
        assert "catalog_loaded" in ready_data


def test_prometheus_metrics():
    """Valida el endpoint GET /metrics y el formato de telemetría Prometheus."""
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "recsys_requests_total" in resp.text
        assert "recsys_latency_p95_ms" in resp.text


def run_all_tests():
    print("=" * 75)
    print("  EJECUTANDO TESTS UNITARIOS DE API REST FASTAPI (TEST_API.PY)")
    print("=" * 75)
    test_api_lifespan_and_health()
    print("[PASS] 1. Lifespan y telemetría de salud (GET /health).")
    test_api_root_navigation()
    print("[PASS] 2. Navegación y bienvenida raíz (GET /).")
    test_recommendation_known_customer()
    print("[PASS] 3. Inferencia para cliente activo (is_cold_start=False, latencia < 50ms).")
    test_recommendation_cold_start_unknown_customer()
    print("[PASS] 4. Fallback de arranque en frío (is_cold_start=True, latencia < 5ms).")
    test_recommendation_validation_error_short_id()
    print("[PASS] 5. Manejo semántico de error HTTP 422 (validación de ID).")
    test_openapi_schema_integrity()
    print("[PASS] 6. Integridad del contrato y esquema OpenAPI (GET /openapi.json).")
    print("=" * 75)
    print("  [OK] TODOS LOS TESTS DE LA API REST SUPERADOS CON ÉXITO")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()
