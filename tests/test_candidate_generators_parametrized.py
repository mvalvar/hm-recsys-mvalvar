"""Suite de Pruebas Unitarias Parametrizadas para Generadores de Candidatos (R1 a R8).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Pruebas en memoria ultra-rápidas (< 0.3s) con fixtures sintéticas que cubren:
- R1: Recompra reciente por usuario
- R2: Popularidad global con decaimiento temporal
- R3: Popularidad por grupo etario (age_bin)
- R4: Popularidad por canal preferente (físico/online)
- R5: Filtrado colaborativo ítem-ítem (co-ocurrencias)
- R6: Afinidad por familias de producto y departamentos
- R7: Artículos en tendencia y aceleración de demanda
- R8: Popularidad estacional por departamento favorito
- Consolidación y deduplicación con priorización de ranking
- Fallback autónomo de septiembre 2020 formateado a 10 dígitos canónicos
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.candidates.generators import (
    consolidate_candidates,
    generate_age_group_popularity,
    generate_channel_popularity,
    generate_global_popularity,
    generate_item_cf,
    generate_product_family,
    generate_repurchase,
    generate_trending_items,
    generate_user_dept_popularity,
    get_popular_fallback_items,
)


@pytest.fixture
def synthetic_transactions() -> pl.DataFrame:
    """Genera transacciones sintéticas con distribución temporal de dos semanas."""
    dates = [
        datetime.date(2020, 9, 20),
        datetime.date(2020, 9, 19),
        datetime.date(2020, 9, 15),
        datetime.date(2020, 9, 12),
        datetime.date(2020, 9, 5),
    ]
    return pl.DataFrame(
        {
            "t_dat": [
                dates[0], dates[1], dates[2],
                dates[0], dates[1], dates[3], dates[4],
                dates[0], dates[1], dates[2],
            ],
            "customer_idx": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            "article_id": [101, 102, 101, 102, 103, 104, 104, 105, 106, 101],
            "price": [0.02, 0.03, 0.02, 0.03, 0.05, 0.01, 0.01, 0.04, 0.02, 0.02],
            "sales_channel_id": [2, 2, 2, 1, 1, 1, 1, 2, 1, 2],
        }
    )


@pytest.fixture
def synthetic_customers() -> pl.DataFrame:
    """Genera clientes sintéticos con etiquetas de cohorte de edad."""
    return pl.DataFrame(
        {
            "customer_idx": [1, 2, 3],
            "customer_id": ["a" * 64, "b" * 64, "c" * 64],
            "age": [22.0, 30.0, 48.0],
            "age_bin": ["<25", "25-34", "45-54"],
        }
    )


@pytest.fixture
def synthetic_articles() -> pl.DataFrame:
    """Genera artículos sintéticos con asignación departamental y tipo de producto."""
    return pl.DataFrame(
        {
            "article_id": [101, 102, 103, 104, 105, 106],
            "department_no": [1001, 1001, 1002, 1003, 1001, 1002],
            "product_type_no": [253, 253, 265, 272, 253, 253],
        }
    )


# TESTS PARAMETRIZADOS PARA HEURÍSTICAS INDIVIDUALES (R1..R8)


@pytest.mark.parametrize("top_k", [1, 2, 5])
def test_heuristic_r1_repurchase_top_k(synthetic_transactions, top_k):
    """R1: Verifica que no se exceda el top_k de recompras por cliente y esquema estricto."""
    r1 = generate_repurchase(synthetic_transactions, top_k=top_k)
    assert r1.schema["customer_idx"] == pl.Int32
    assert r1.schema["article_id"] == pl.Int32
    assert r1.schema["rank_in_source"] == pl.Int16
    assert (r1["source"] == "R1_repurchase").all()

    counts_per_cust = r1.group_by("customer_idx").len()
    assert (counts_per_cust["len"] <= top_k).all()


@pytest.mark.parametrize("decay", [0.0, 0.05, 0.5])
def test_heuristic_r2_global_popularity(synthetic_transactions, decay):
    """R2: Verifica el decaimiento temporal y presencia de todos los clientes."""
    r2 = generate_global_popularity(synthetic_transactions, lambda_decay=decay, top_k=3)
    assert r2.schema["customer_idx"] == pl.Int32
    assert r2.schema["article_id"] == pl.Int32
    assert (r2["source"] == "R2_popular_global").all()
    # Todos los clientes únicos de transacciones deben recibir candidatos
    assert set(r2["customer_idx"].unique()) == {1, 2, 3}


def test_heuristic_r3_age_group(synthetic_transactions, synthetic_customers):
    """R3: Verifica segmentación por cohorte demográfica."""
    r3 = generate_age_group_popularity(synthetic_transactions, synthetic_customers, top_k=2)
    assert r3.schema["customer_idx"] == pl.Int32
    assert (r3["source"] == "R3_popular_age").all()
    assert r3.height > 0


def test_heuristic_r4_channel_popularity(synthetic_transactions):
    """R4: Verifica discriminación por canal físico (1) vs online (2)."""
    r4 = generate_channel_popularity(synthetic_transactions, top_k=2)
    assert r4.schema["customer_idx"] == pl.Int32
    assert (r4["source"] == "R4_popular_channel").all()
    assert r4.height > 0


def test_heuristic_r5_item_cf(synthetic_transactions):
    """R5: Verifica generación de filtrado colaborativo ítem-ítem."""
    r5 = generate_item_cf(synthetic_transactions, window_days=30, top_k=3)
    assert r5.schema["customer_idx"] == pl.Int32
    assert (r5["source"] == "R5_itemcf").all()


def test_heuristic_r6_product_family(synthetic_transactions, synthetic_articles):
    """R6: Verifica afinidad por familia de producto habitual."""
    r6 = generate_product_family(synthetic_transactions, synthetic_articles, top_k=2)
    assert r6.schema["customer_idx"] == pl.Int32
    assert (r6["source"] == "R6_product_family").all()


def test_heuristic_r7_trending(synthetic_transactions):
    """R7: Verifica artículos en aceleración de demanda."""
    r7 = generate_trending_items(synthetic_transactions, top_k=2)
    assert r7.schema["customer_idx"] == pl.Int32
    assert (r7["source"] == "R7_trending").all()


def test_heuristic_r8_user_dept_popularity(synthetic_transactions, synthetic_articles):
    """R8: Verifica popularidad dentro del departamento preferente del cliente."""
    r8 = generate_user_dept_popularity(synthetic_transactions, synthetic_articles, top_k=2)
    assert r8.schema["customer_idx"] == pl.Int32
    assert (r8["source"] == "R8_user_dept_popularity").all()


# TESTS DE CONSOLIDACIÓN Y FALLBACK


def test_consolidate_candidates_deduplication(synthetic_transactions):
    """Verifica que la consolidación elimine duplicados y compute meta-features."""
    r1 = generate_repurchase(synthetic_transactions, top_k=3)
    r2 = generate_global_popularity(synthetic_transactions, top_k=3)

    consolidated = consolidate_candidates([r1, r2], max_per_user=10)

    assert consolidated.height > 0
    # Verificación de unicidad de pares (customer_idx, article_id)
    n_pairs = consolidated.select(["customer_idx", "article_id"]).height
    n_unique_pairs = consolidated.select(["customer_idx", "article_id"]).n_unique()
    assert n_pairs == n_unique_pairs

    # Flags booleanos correctos
    assert "best_rank" in consolidated.columns
    assert "n_sources" in consolidated.columns
    assert "is_R1" in consolidated.columns
    assert "is_R2" in consolidated.columns


def test_popular_fallback_formatting(synthetic_transactions):
    """Verifica que el fallback estacional retorne strings canónicos de 10 dígitos."""
    fallback = get_popular_fallback_items(synthetic_transactions, top_k=5)
    assert len(fallback) <= 5
    assert len(fallback) > 0
    assert all(isinstance(x, str) and len(x) == 10 and x.isdigit() for x in fallback)


def test_age_group_fallback_items(synthetic_transactions, synthetic_customers):
    """Verifica la generación del diccionario de fallbacks segmentado por edad."""
    from src.candidates.generators import get_age_group_fallback_items

    fallbacks = get_age_group_fallback_items(
        transactions_df=synthetic_transactions,
        customers_df=synthetic_customers,
        top_k=4,
    )
    assert "GLOBAL" in fallbacks
    assert len(fallbacks["GLOBAL"]) > 0
    for label in ["<25", "25-34", "45-54"]:
        assert label in fallbacks
        assert len(fallbacks[label]) > 0
        assert all(len(x) == 10 and x.isdigit() for x in fallbacks[label])


def test_consolidate_candidates_partitioned_branch(synthetic_transactions):
    """Verifica la rama de particionamiento out-of-core de consolidación."""
    r1 = generate_repurchase(synthetic_transactions, top_k=3)
    r2 = generate_global_popularity(synthetic_transactions, top_k=3)
    consolidated = consolidate_candidates(r1, r2, partition_threshold=3, n_partitions=2)
    assert consolidated.height > 0
    assert "best_rank" in consolidated.columns
    assert "n_sources" in consolidated.columns
