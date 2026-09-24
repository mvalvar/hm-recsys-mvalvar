"""Suite de Pruebas Unitarias para el Pipeline Canónico RecSys en 6 Etapas.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.model_loader import RecommenderServiceLoader
from src.recsys_pipeline.pipeline import CanonicalRecsysPipeline, RecsysRequest


def test_pipeline_execution_known_customer():
    """Valida la ejecución de las 6 etapas para un cliente activo."""
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()

    pipeline = CanonicalRecsysPipeline(loader)
    sample_idx = list(loader.candidate_features.keys())[0]
    sample_id = loader.reverse_customer_mapping[sample_idx]

    req = RecsysRequest(customer_id=sample_id, limit=6, offset=0)
    resp = pipeline.execute(req)

    assert resp.customer_id == sample_id
    assert resp.is_cold_start is False
    assert len(resp.recommendations) == 6
    assert resp.count == 6
    assert resp.latency_ms > 0.0


def test_pipeline_cold_start_fallback():
    """Valida la degradación elegante a popularidad para un cliente frío."""
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()

    pipeline = CanonicalRecsysPipeline(loader)
    unknown_id = "0000000000000000000000000000000000000000000000000000000000000000"

    req = RecsysRequest(customer_id=unknown_id, limit=12)
    resp = pipeline.execute(req)

    assert resp.customer_id == unknown_id
    assert resp.is_cold_start is True
    assert len(resp.recommendations) == 12
    assert resp.recommendations == loader.popular_fallback[:12]


def test_pipeline_filter_exclusions():
    """Valida que la etapa de Filter excluya exitosamente los artículos vetados."""
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()

    pipeline = CanonicalRecsysPipeline(loader)
    sample_idx = list(loader.candidate_features.keys())[0]
    sample_id = loader.reverse_customer_mapping[sample_idx]

    # Tomar la recomendación número 1 habitual
    base_resp = pipeline.execute(RecsysRequest(customer_id=sample_id, limit=6))
    top_item = base_resp.recommendations[0]

    # Ejecutar con ese artículo excluido
    filtered_resp = pipeline.execute(
        RecsysRequest(customer_id=sample_id, limit=6, excluded_article_ids={top_item})
    )

    assert top_item not in filtered_resp.recommendations
    assert len(filtered_resp.recommendations) == 6


def test_pipeline_pagination_non_overlapping():
    """Valida la paginación determinista de candidatos."""
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()

    pipeline = CanonicalRecsysPipeline(loader)
    sample_idx = list(loader.candidate_features.keys())[0]
    sample_id = loader.reverse_customer_mapping[sample_idx]

    p1 = pipeline.execute(RecsysRequest(customer_id=sample_id, limit=4, offset=0))
    p2 = pipeline.execute(RecsysRequest(customer_id=sample_id, limit=4, offset=4))

    assert len(p1.recommendations) == 4
    assert len(p2.recommendations) == 4
    assert set(p1.recommendations).isdisjoint(set(p2.recommendations))
