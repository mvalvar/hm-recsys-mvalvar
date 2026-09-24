"""Suite de Pruebas Unitarias para Métricas de Ranking (MAP@K, Recall@K, HitRate@K).

Valida:
1. Precisión matemática de AP@K y MAP@K con predicciones duplicadas (sin compactación indebida).
2. Caso de frontera: actual=[1, 2], predicted=[1, 1, 2] da 5/6 (~0.8333), no 1.0.
3. Ausencia de doble conteo para ítems repetidos tras el primer acierto.
4. Preservación estricta de la ventana Top-K (los ítems más allá de K no entran al corte tras duplicados).
5. Casos de listas vacías (actual vacío, predicción vacía, k <= 0).
6. Coherencia de Recall@K, HitRate@K, Catalog Coverage y evaluate_ranking_df con Polars.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation.metrics import (  # noqa: E402
    ap_at_k,
    catalog_coverage,
    evaluate_ranking_df,
    hit_rate_at_k,
    map_at_k,
    recall_at_k,
)


def test_ap_at_k_with_duplicates_does_not_compact():
    """Valida que los duplicados no compacten la lista inflando artificialmente el score."""
    actual = [1, 2]
    predicted = [1, 1, 2]
    # Pos 1: item 1 (hit 1, P=1/1, rel=1) -> 1.0
    # Pos 2: item 1 (duplicado, ya visto, rel=0) -> 0
    # Pos 3: item 2 (hit 2, P=2/3, rel=1) -> 2/3
    # Denominador: min(len(actual), 3) = 2
    # Score: (1.0 + 2/3) / 2 = (5/3) / 2 = 5/6 = 0.8333333333333334
    score = ap_at_k(actual, predicted, k=3)
    assert math.isclose(score, 5.0 / 6.0, rel_tol=1e-6), f"Esperado 5/6 (~0.8333), obtenido {score}"

    # Bajo k=12 el resultado para esta predicción de 3 ítems debe ser idéntico
    score_k12 = ap_at_k(actual, predicted, k=12)
    assert math.isclose(score_k12, 5.0 / 6.0, rel_tol=1e-6)


def test_ap_at_k_duplicate_after_first_hit_does_not_score_again():
    """Valida que una predicción repetida tras el primer acierto no sume nuevamente."""
    actual = [10]
    predicted = [10, 10, 10, 10]
    # Pos 1: hit 1, P=1/1=1.0. Las demás son duplicadas y no suman.
    # Score = 1.0 / min(1, 4) = 1.0
    score = ap_at_k(actual, predicted, k=4)
    assert math.isclose(score, 1.0, rel_tol=1e-6)

    # Si el acierto está en la posición 2 y se repite en 3 y 4
    predicted_later = [99, 10, 10, 10]
    # Pos 1: miss (rel=0)
    # Pos 2: hit 1, P=1/2=0.5, rel=1
    # Pos 3, 4: duplicados de 10, rel=0
    # Score = 0.5 / min(1, 4) = 0.5
    score_later = ap_at_k(actual, predicted_later, k=4)
    assert math.isclose(score_later, 0.5, rel_tol=1e-6)


def test_ap_at_k_empty_cases():
    """Valida que entradas vacías o k<=0 retornen 0.0 de forma segura."""
    assert ap_at_k([], [1, 2, 3], k=12) == 0.0
    assert ap_at_k([1, 2], [], k=12) == 0.0
    assert ap_at_k([], [], k=12) == 0.0
    assert ap_at_k([1, 2], [1, 2], k=0) == 0.0
    assert ap_at_k([1, 2], [1, 2], k=-5) == 0.0


def test_ap_at_k_beyond_k_not_promoted():
    """Valida que artículos fuera del corte K no se promuevan al haber duplicados en el top K."""
    actual = [1, 2]
    # Top 3 son [1, 1, 99]. El artículo 2 está en la posición 4 (fuera de top 3).
    predicted = [1, 1, 99, 2]
    score = ap_at_k(actual, predicted, k=3)
    # Para k=3: solo se evalúan [1, 1, 99]. El ítem 2 en pos 4 NO entra.
    # Pos 1: item 1 (hit 1, P=1/1=1.0)
    # Pos 2: item 1 (duplicado)
    # Pos 3: item 99 (fallo)
    # Total = 1.0 / min(len(actual), 3) = 1.0 / 2 = 0.5
    assert math.isclose(score, 0.5, rel_tol=1e-6), (
        f"El ítem fuera de k=3 no debe promoverse, obtenido {score}"
    )


def test_map_at_k_multiple_users():
    """Valida la media agregada MAP@K sobre múltiples usuarios."""
    actuals = {
        1: [10, 20],
        2: [30],
        3: [40, 50, 60],
        4: [],  # usuario sin compras reales
    }
    predictions = {
        1: [10, 20],  # AP=1.0
        2: [99, 30],  # AP=0.5
        3: [
            40,
            99,
            50,
        ],  # Pos 1: 1/1=1.0, Pos 2: miss, Pos 3: 2/3=0.6667 -> sum=1.6667 / 3 = 0.555555...
        4: [10, 20],  # AP=0.0
    }
    # MAP: promedio sobre 4 usuarios
    ap1 = ap_at_k([10, 20], [10, 20], k=3)
    ap2 = ap_at_k([30], [99, 30], k=3)
    ap3 = ap_at_k([40, 50, 60], [40, 99, 50], k=3)
    ap4 = ap_at_k([], [10, 20], k=3)
    expected_map = (ap1 + ap2 + ap3 + ap4) / 4.0

    calculated_map = map_at_k(actuals, predictions, k=3)
    assert math.isclose(calculated_map, expected_map, rel_tol=1e-6)
    assert map_at_k({}, {}) == 0.0


def test_recall_and_hit_rate_at_k():
    """Valida que Recall@K y HitRate@K calculen proporciones correctas."""
    actuals = {
        1: [10, 20],
        2: [30],
    }
    predictions = {
        1: [10, 99],  # 1 de 2 recuperados -> recall = 0.5, hit = 1
        2: [88, 77],  # 0 de 1 recuperados -> recall = 0.0, hit = 0
    }
    assert math.isclose(recall_at_k(actuals, predictions, k=2), 0.25, rel_tol=1e-6)
    assert math.isclose(hit_rate_at_k(actuals, predictions, k=2), 0.5, rel_tol=1e-6)


def test_catalog_coverage():
    """Valida el cálculo de cobertura de catálogo."""
    predictions = {
        1: [101, 102],
        2: [102, 103],
    }
    # Artículos únicos recomendados en top 2: {101, 102, 103} = 3 artículos
    cov = catalog_coverage(predictions, total_catalog_size=10, k=2)
    assert math.isclose(cov, 0.3, rel_tol=1e-6)


def test_evaluate_ranking_df_polars():
    """Valida la interfaz sobre Polars DataFrame."""
    df = pl.DataFrame(
        {
            "customer_idx": [1, 2],
            "actual_articles": [[1, 2], [3]],
            "predicted_articles": [[1, 1, 2], [9, 8, 7]],
        }
    )
    metrics = evaluate_ranking_df(df, k=3, total_catalog_size=10)
    assert "map@3" in metrics
    assert "recall@3" in metrics
    assert "hit_rate@3" in metrics
    assert "coverage@3" in metrics
    # Usuario 1 da 5/6, Usuario 2 da 0.0 -> MAP = (5/6 + 0) / 2 = 5/12
    assert math.isclose(metrics["map@3"], (5.0 / 6.0) / 2.0, rel_tol=1e-6)
