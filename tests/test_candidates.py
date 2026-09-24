"""Suite de Pruebas Unitarias para el Módulo de Generación de Candidatos (R1 a R8).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Valida:
1. Heurística R1 (Recompra personal con recencia y frecuencia).
2. Heurística R2 (Popularidad global con decaimiento temporal exponencial).
3. Heurística R3 (Popularidad segmentada por cohorte demográfica de edad).
4. Heurística R4 (Popularidad por canal preferente físico vs online).
5. Heurística R5 (Filtrado colaborativo ítem-ítem por co-ocurrencia).
6. Heurística R6 (Afinidad a familias de producto y departamentos).
7. Heurística R7 (Artículos en tendencia y aceleración de demanda).
8. Heurística R8 (Popularidad estacional en el departamento favorito del cliente).
9. Consolidación multi-fuente: deduplicación, cálculo de best_rank, n_sources y flags is_R1..is_R8.
10. Validaciones defensivas ante DataFrames vacíos o esquemas incompletos.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.candidates.generators import (  # noqa: E402
    consolidate_candidates,
    generate_age_group_popularity,
    generate_channel_popularity,
    generate_global_popularity,
    generate_item_cf,
    generate_product_family,
    generate_repurchase,
    generate_trending_items,
    generate_user_dept_popularity,
)
from src.utils.validation import split_transactions_temporal  # noqa: E402

# FIXTURES SINTÉTICAS PARA PRUEBAS ULTRA-RÁPIDAS (<0.1s)


@pytest.fixture
def synthetic_transactions() -> pl.DataFrame:
    """Genera transacciones sintéticas controladas para 3 clientes y 6 artículos."""
    dates = [
        datetime.date(2020, 9, 20),
        datetime.date(2020, 9, 19),
        datetime.date(2020, 9, 15),
        datetime.date(2020, 9, 10),
        datetime.date(2020, 9, 5),
    ]
    return pl.DataFrame(
        {
            "t_dat": [
                dates[0],
                dates[1],
                dates[2],
                dates[0],
                dates[1],
                dates[3],
                dates[4],
                dates[0],
                dates[1],
                dates[2],
            ],
            "customer_idx": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            "article_id": [101, 102, 101, 102, 103, 104, 104, 105, 106, 101],
            "price": [0.02, 0.03, 0.02, 0.03, 0.05, 0.01, 0.01, 0.04, 0.02, 0.02],
            "sales_channel_id": [2, 2, 2, 1, 1, 1, 1, 2, 1, 2],
        }
    )


@pytest.fixture
def synthetic_customers() -> pl.DataFrame:
    """Genera clientes sintéticos con cohortes demográficas asignadas."""
    return pl.DataFrame(
        {
            "customer_idx": [1, 2, 3],
            "customer_id": ["a" * 64, "b" * 64, "c" * 64],
            "age": [22, 30, 48],
            "age_bin": ["<25", "25-34", "45-54"],
        }
    )


@pytest.fixture
def synthetic_articles() -> pl.DataFrame:
    """Genera catálogo de artículos con metadatos de departamento."""
    return pl.DataFrame(
        {
            "article_id": [101, 102, 103, 104, 105, 106],
            "department_no": [1001, 1001, 1002, 1002, 1003, 1001],
            "product_type_no": [250, 250, 260, 260, 270, 250],
        }
    )


# TESTS DE CADA HEURÍSTICA DE RECALL (R1 A R8)


def test_r1_repurchase(synthetic_transactions: pl.DataFrame):
    """Valida que R1 recupere los artículos comprados previamente en orden cronológico."""
    res = generate_repurchase(synthetic_transactions, top_k=5)
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    assert (res["source"] == "R1_repurchase").all()

    # Cliente 1 compró 101 dos veces (la más reciente 2020-09-20) y 102 una vez (2020-09-19)
    c1_cands = res.filter(pl.col("customer_idx") == 1)["article_id"].to_list()
    assert c1_cands[0] == 101
    assert 102 in c1_cands


def test_r2_global_popularity(synthetic_transactions: pl.DataFrame):
    """Valida que R2 pondere con decaimiento y asigne los más populares a todos los usuarios."""
    res = generate_global_popularity(synthetic_transactions, top_k=3)
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    assert (res["source"] == "R2_popular_global").all()

    # Todos los clientes sintéticos deben recibir los top artículos globales
    unique_custs = set(res["customer_idx"].to_list())
    assert unique_custs == {1, 2, 3}
    for c in unique_custs:
        c_items = res.filter(pl.col("customer_idx") == c)["article_id"].to_list()
        assert len(c_items) <= 3


def test_r3_age_group_popularity(
    synthetic_transactions: pl.DataFrame,
    synthetic_customers: pl.DataFrame,
):
    """Valida que R3 asigne popularidad correspondiente al age_bin de cada cliente."""
    res = generate_age_group_popularity(synthetic_transactions, synthetic_customers, top_k=3)
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    assert (res["source"] == "R3_popular_age").all()

    # Cliente 1 (<25) adquirió 101 y 102
    c1_items = res.filter(pl.col("customer_idx") == 1)["article_id"].to_list()
    assert len(c1_items) > 0
    assert 101 in c1_items or 102 in c1_items


def test_r4_channel_popularity(synthetic_transactions: pl.DataFrame):
    """Valida que R4 identifique el canal preferente (físico vs online) de cada usuario."""
    res = generate_channel_popularity(synthetic_transactions, top_k=3)
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    assert (res["source"] == "R4_popular_channel").all()

    # Cliente 2 compró casi todo por canal 1 (tienda física)
    c2_items = res.filter(pl.col("customer_idx") == 2)["article_id"].to_list()
    assert len(c2_items) > 0


def test_r5_item_cf(synthetic_transactions: pl.DataFrame):
    """Valida que R5 proyecte co-ocurrencias entre artículos co-adquiridos."""
    res = generate_item_cf(synthetic_transactions, window_days=30, top_k=3)
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    if res.height > 0:
        assert (res["source"] == "R5_itemcf").all()


def test_r6_product_family(
    synthetic_transactions: pl.DataFrame,
    synthetic_articles: pl.DataFrame,
):
    """Valida que R6 recomiende artículos dentro de los departamentos afines al cliente."""
    res = generate_product_family(synthetic_transactions, synthetic_articles, top_k=3)
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    assert (res["source"] == "R6_product_family").all()

    # Cliente 1 compró en depto 1001 (artículos 101, 102). Debe recibir artículos del depto 1001 (101, 102, 106)
    c1_items = set(res.filter(pl.col("customer_idx") == 1)["article_id"].to_list())
    assert len(c1_items.intersection({101, 102, 106})) > 0


def test_r7_trending_items():
    """Valida que R7 calcule la velocidad relativa entre semanas reciente y previa."""
    t_now = datetime.date(2020, 9, 22)
    # Artículo 201 tiene aceleración (0 ventas previa semana, 4 en la reciente)
    tx_trending = pl.DataFrame(
        {
            "t_dat": [
                t_now - datetime.timedelta(days=1),
                t_now - datetime.timedelta(days=2),
                t_now - datetime.timedelta(days=3),
                t_now - datetime.timedelta(days=4),
                t_now - datetime.timedelta(days=10),
            ],
            "customer_idx": [1, 2, 3, 1, 2],
            "article_id": [201, 201, 201, 201, 202],
        }
    )
    res = generate_trending_items(tx_trending, top_k=2)
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    assert (res["source"] == "R7_trending").all()
    assert 201 in res["article_id"].to_list()


def test_r8_user_dept_popularity(
    synthetic_transactions: pl.DataFrame,
    synthetic_articles: pl.DataFrame,
):
    """Valida que R8 recomiende superventas recientes del departamento preferido del usuario."""
    res = generate_user_dept_popularity(
        transactions_df=synthetic_transactions,
        articles_df=synthetic_articles,
        top_k=3,
        recent_days=30,
    )
    assert list(res.columns) == ["customer_idx", "article_id", "source", "rank_in_source"]
    assert (res["source"] == "R8_user_dept_popularity").all()

    # Cliente 1 tiene como depto favorito el 1001
    c1_items = set(res.filter(pl.col("customer_idx") == 1)["article_id"].to_list())
    assert len(c1_items) > 0


# TESTS DE CONSOLIDACIÓN Y META-FEATURES


def test_consolidate_candidates_all_heuristics(
    synthetic_transactions: pl.DataFrame,
    synthetic_customers: pl.DataFrame,
    synthetic_articles: pl.DataFrame,
):
    """Valida la consolidación de las 8 heurísticas, deduplicación y meta-features."""
    r1 = generate_repurchase(synthetic_transactions, top_k=2)
    r2 = generate_global_popularity(synthetic_transactions, top_k=2)
    r3 = generate_age_group_popularity(synthetic_transactions, synthetic_customers, top_k=2)
    r4 = generate_channel_popularity(synthetic_transactions, top_k=2)
    r5 = generate_item_cf(synthetic_transactions, window_days=30, top_k=2)
    r6 = generate_product_family(synthetic_transactions, synthetic_articles, top_k=2)
    r7 = generate_trending_items(synthetic_transactions, top_k=2)
    r8 = generate_user_dept_popularity(
        synthetic_transactions, synthetic_articles, top_k=2, recent_days=30
    )

    cands_df = consolidate_candidates(r1, r2, r3, r4, r5, r6, r7, r8, max_per_user=10)

    # Verificar esquema exacto de consolidación
    expected_cols = [
        "customer_idx",
        "article_id",
        "best_rank",
        "n_sources",
        "is_R1",
        "is_R2",
        "is_R3",
        "is_R4",
        "is_R5",
        "is_R6",
        "is_R7",
        "is_R8",
    ]
    assert list(cands_df.columns) == expected_cols

    # Verificar unicidad (cero duplicados por cliente-artículo)
    dups = cands_df.group_by(["customer_idx", "article_id"]).len().filter(pl.col("len") > 1)
    assert dups.height == 0, f"Se encontraron {dups.height} pares duplicados en el pool consolidado"

    # Verificar meta-features
    assert (cands_df["best_rank"] >= 1).all()
    assert (cands_df["n_sources"] >= 1).all()
    for i in range(1, 9):
        assert cands_df[f"is_R{i}"].dtype == pl.Boolean

    # Si un candidato está en R1, is_R1 debe ser True
    r1_pairs = set(zip(r1["customer_idx"].to_list(), r1["article_id"].to_list(), strict=False))
    for row in cands_df.iter_rows(named=True):
        pair = (row["customer_idx"], row["article_id"])
        if pair in r1_pairs:
            assert row["is_R1"] is True


def test_generators_defensive_assertions():
    """Valida que los generadores lancen aserciones defensivas ante inputs defectuosos."""
    empty_df = pl.DataFrame(
        {
            "t_dat": pl.Series([], dtype=pl.Date),
            "customer_idx": pl.Series([], dtype=pl.Int32),
            "article_id": pl.Series([], dtype=pl.Int32),
        }
    )

    # DataFrame vacío debe fallar defensivamente
    with pytest.raises(AssertionError):
        generate_repurchase(empty_df)

    # Falta de columna obligatoria debe fallar
    missing_col_df = pl.DataFrame(
        {
            "customer_idx": [1],
            "article_id": [101],
        }
    )
    with pytest.raises(AssertionError):
        generate_repurchase(missing_col_df)


def test_candidate_generation_temporal_isolation():
    """Valida que los candidatos generados con partición temporal no contengan compras futuras."""
    # Base temporal sintética dinámica (14 días)
    start_date = datetime.date(2025, 4, 1)
    d_hist = start_date + datetime.timedelta(days=2)  # Día histórico (semana train)
    d_val = start_date + datetime.timedelta(days=12)  # Día futuro (semana validación retenida)

    # Cliente 1: compró artículo 1001 en el histórico y 2002 en la semana de validación
    # Cliente 2: compró artículo 3003 solo en la semana de validación
    tx_df = pl.DataFrame(
        {
            "customer_idx": [1, 1, 2],
            "article_id": [1001, 2002, 3003],
            "t_dat": [d_hist, d_val, d_val],
        }
    )

    # Partición temporal estricta de 7 días
    split = split_transactions_temporal(tx_df, val_days=7)
    train_tx = split.train_df

    # Generar candidatos de recompra solo con el histórico
    r1 = generate_repurchase(train_tx)

    # Artículos recomendados por recompra
    repurchased_pairs = set(
        zip(r1["customer_idx"].to_list(), r1["article_id"].to_list(), strict=False)
    )

    # El artículo 1001 debe estar presente para cliente 1
    assert (1, 1001) in repurchased_pairs
    # El artículo 2002 (comprado en val) NO debe estar presente para cliente 1
    assert (1, 2002) not in repurchased_pairs
    # El cliente 2 NO debe tener ningún candidato de recompra
    assert (2, 3003) not in repurchased_pairs

    # Popularidad global calculada sobre train_tx no debe contener el artículo 3003
    r2 = generate_global_popularity(train_tx)
    pop_articles = set(r2["article_id"].to_list())
    assert 1001 in pop_articles
    assert 2002 not in pop_articles
    assert 3003 not in pop_articles
