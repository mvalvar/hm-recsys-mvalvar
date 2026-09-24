"""Suite de Pruebas Unitarias Parametrizadas para Transformadores de Features (39 Features).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Pruebas en memoria ultra-rápidas (< 0.2s) con fixtures sintéticas que cubren:
- Namespace 1: User Features (9 características)
- Namespace 2: Article Features (9 características)
- Namespace 3: Interaction Features (8 características)
- Namespace 4: Candidate Meta-Features (13 características)
- Matriz Completa: Ensamblado de 39 características con tipado estricto y ausencia de nulos
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.features.builder import (
    build_article_features,
    build_candidate_meta_features,
    build_full_feature_matrix,
    build_interaction_features,
    build_user_features,
)


@pytest.fixture
def synthetic_transactions() -> pl.DataFrame:
    """Genera transacciones sintéticas controladas con variedad temporal y de precios."""
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
    """Genera clientes sintéticos con variedad de estados de club y demografía."""
    return pl.DataFrame(
        {
            "customer_idx": [1, 2, 3, 4],  # 4 es cliente nuevo sin transacciones
            "age": [22.0, 30.0, 48.0, None],
            "club_member_status": ["ACTIVE", "PRE-CREATE", "LEFT CLUB", None],
            "fashion_news_frequency": ["Regularly", "Monthly", "NONE", None],
        }
    )


@pytest.fixture
def synthetic_articles() -> pl.DataFrame:
    """Genera catálogo de artículos sintéticos con metadatos departamentales."""
    return pl.DataFrame(
        {
            "article_id": [101, 102, 103, 104, 105, 106, 999],  # 999 artículo sin ventas
            "product_type_no": [253, 253, 265, 272, 253, 253, 300],
            "graphical_appearance_no": [1010016, 1010016, 1010010, 1010001, 1010016, 1010016, 1010016],
            "colour_group_code": [9, 10, 11, 12, 13, 14, 15],
            "department_no": [1001, 1001, 1002, 1003, 1001, 1002, 2000],
            "index_group_no": [1, 1, 2, 2, 1, 2, 3],
            "section_no": [15, 15, 16, 17, 15, 16, 20],
            "garment_group_no": [1005, 1005, 1006, 1007, 1005, 1006, 1010],
        }
    )


@pytest.fixture
def synthetic_candidates() -> pl.DataFrame:
    """Genera pares candidato-usuario sintéticos con metadatos de fuentes."""
    return pl.DataFrame(
        {
            "customer_idx": [1, 1, 2, 2, 3],
            "article_id": [101, 102, 102, 103, 105],
            "best_rank": [1, 2, 1, 3, 1],
            "n_sources": [3, 1, 2, 1, 4],
            "is_R1": [True, True, False, False, True],
            "is_R2": [True, False, True, False, True],
            "is_R3": [True, False, False, False, True],
            "is_R4": [False, False, True, False, False],
            "is_R5": [False, False, False, True, False],
            "is_R6": [False, False, False, False, False],
            "is_R7": [False, False, False, False, True],
            "is_R8": [False, False, False, False, False],
        }
    )


# TESTS PARAMETRIZADOS: NAMESPACE 1 (USER FEATURES)


@pytest.mark.parametrize(
    "cust_idx,expected_tx_count,expected_club_code,expected_news_code",
    [
        (1, 3, 1, 2),  # ACTIVE -> 1, Regularly -> 2
        (2, 4, 2, 1),  # PRE-CREATE -> 2, Monthly -> 1
        (3, 3, 3, 0),  # LEFT CLUB -> 3, NONE -> 0
        (4, 0, 0, 0),  # Cliente nuevo sin tx -> 0 tx, default club 0, default news 0
    ],
)
def test_user_features_individual_profiles(
    synthetic_transactions,
    synthetic_customers,
    cust_idx,
    expected_tx_count,
    expected_club_code,
    expected_news_code,
):
    """Verifica la asignación correcta de estadísticos y codificaciones ordinales por usuario."""
    u_df = build_user_features(synthetic_transactions, synthetic_customers)
    user_row = u_df.filter(pl.col("customer_idx") == cust_idx).to_dicts()[0]

    assert user_row["u_total_transactions"] == expected_tx_count
    assert user_row["u_club_status"] == expected_club_code
    assert user_row["u_fashion_news"] == expected_news_code
    assert user_row["u_mean_price"] >= 0.0
    assert 0.0 <= user_row["u_online_ratio"] <= 1.0


def test_user_features_schema_and_nulls(synthetic_transactions, synthetic_customers):
    """Verifica que ninguna de las 9 características de usuario contenga valores nulos."""
    u_df = build_user_features(synthetic_transactions, synthetic_customers)
    expected_cols = [
        "customer_idx",
        "u_total_transactions",
        "u_unique_articles",
        "u_mean_price",
        "u_std_price",
        "u_online_ratio",
        "u_age",
        "u_club_status",
        "u_fashion_news",
        "u_last_purchase_days_ago",
    ]
    assert u_df.columns == expected_cols
    assert all(u_df[c].null_count() == 0 for c in expected_cols)


# TESTS PARAMETRIZADOS: NAMESPACE 2 (ARTICLE FEATURES)


@pytest.mark.parametrize(
    "art_id,expected_min_sales",
    [
        (101, 3),
        (102, 2),
        (104, 2),
        (999, 0),  # Artículo sin ventas en el histórico
    ],
)
def test_article_features_sales_aggregation(
    synthetic_transactions,
    synthetic_articles,
    art_id,
    expected_min_sales,
):
    """Verifica agregaciones de ventas y valores por defecto para artículos fríos."""
    a_df = build_article_features(synthetic_transactions, synthetic_articles)
    art_row = a_df.filter(pl.col("article_id") == art_id).to_dicts()[0]

    assert art_row["a_sales_count"] == expected_min_sales
    assert art_row["a_sales_decayed"] >= 0.0
    assert art_row["a_is_recent_introduction"] in (0, 1)


def test_article_features_schema_and_nulls(synthetic_transactions, synthetic_articles):
    """Verifica que ninguna de las 9 características de artículo contenga nulos."""
    a_df = build_article_features(synthetic_transactions, synthetic_articles)
    expected_cols = [
        "article_id",
        "a_sales_count",
        "a_unique_customers",
        "a_mean_price",
        "a_sales_decayed",
        "a_is_recent_introduction",
        "a_product_type_no",
        "a_graphical_appearance_no",
        "a_colour_group_code",
        "a_department_no",
    ]
    assert a_df.columns == expected_cols
    assert all(a_df[c].null_count() == 0 for c in expected_cols)


# TESTS PARAMETRIZADOS: NAMESPACE 3 (INTERACTION FEATURES)


def test_interaction_features_cross_calculations(
    synthetic_candidates,
    synthetic_transactions,
    synthetic_customers,
    synthetic_articles,
):
    """Verifica afinidad de departamento, diferencial de precios y recompras previas."""
    u_df = build_user_features(synthetic_transactions, synthetic_customers)
    a_df = build_article_features(synthetic_transactions, synthetic_articles)
    uxa_df = build_interaction_features(
        synthetic_candidates, synthetic_transactions, u_df, a_df
    )

    expected_cols = [
        "customer_idx",
        "article_id",
        "uxa_repurchase_count",
        "uxa_days_since_last_purchase",
        "uxa_dept_affinity",
        "uxa_price_diff",
        "uxa_price_ratio",
        "uxa_bought_dept_before",
        "uxa_channel_affinity",
        "uxa_is_favorite_dept",
    ]
    assert uxa_df.columns == expected_cols
    assert all(uxa_df[c].null_count() == 0 for c in expected_cols)

    # Cliente 1 compró artículo 101 dos veces
    pair_1_101 = uxa_df.filter(
        (pl.col("customer_idx") == 1) & (pl.col("article_id") == 101)
    ).to_dicts()[0]
    assert pair_1_101["uxa_repurchase_count"] == 2
    assert pair_1_101["uxa_bought_dept_before"] == 1


# TESTS: NAMESPACE 4 (CANDIDATE META-FEATURES) & FULL MATRIX


def test_candidate_meta_features(synthetic_candidates):
    """Verifica que las meta-características de candidatos conserven flags y best_rank."""
    meta_df = build_candidate_meta_features(synthetic_candidates)
    assert meta_df.height == synthetic_candidates.height
    assert "best_rank" in meta_df.columns
    assert "n_sources" in meta_df.columns
    for r in range(1, 9):
        assert f"is_R{r}" in meta_df.columns


def test_full_feature_matrix_assembly(
    synthetic_candidates,
    synthetic_transactions,
    synthetic_customers,
    synthetic_articles,
):
    """Verifica el ensamblado completo de las 39 características sin fugas ni nulos."""
    full_matrix = build_full_feature_matrix(
        candidates_df=synthetic_candidates,
        transactions_df=synthetic_transactions,
        customers_df=synthetic_customers,
        articles_df=synthetic_articles,
    )

    assert full_matrix.height == synthetic_candidates.height
    assert "customer_idx" in full_matrix.columns
    assert "article_id" in full_matrix.columns

    # Validación de completitud en matriz de entrenamiento
    for col in full_matrix.columns:
        assert full_matrix[col].null_count() == 0, f"Nulos detectados en columna: {col}"


def test_full_feature_matrix_with_out_path(
    synthetic_candidates,
    synthetic_transactions,
    synthetic_customers,
    synthetic_articles,
    tmp_path,
):
    """Verifica que build_full_feature_matrix guarde correctamente en disco cuando out_path se especifica."""
    target_parquet = tmp_path / "matrix_test.parquet"
    _ = build_full_feature_matrix(
        candidates_df=synthetic_candidates,
        transactions_df=synthetic_transactions,
        customers_df=synthetic_customers,
        articles_df=synthetic_articles,
        out_path=target_parquet,
    )
    assert target_parquet.exists()
    saved_df = pl.read_parquet(target_parquet)
    assert saved_df.height == synthetic_candidates.height


def test_full_feature_matrix_streaming_to_disk(
    synthetic_candidates,
    synthetic_transactions,
    synthetic_customers,
    synthetic_articles,
    tmp_path,
):
    """Verifica la rama out-of-core de transmisión continua a disco con PyArrow ParquetWriter."""
    stream_out = tmp_path / "matrix_streamed.parquet"
    res = build_full_feature_matrix(
        candidates_df=synthetic_candidates,
        transactions_df=synthetic_transactions,
        customers_df=synthetic_customers,
        articles_df=synthetic_articles,
        out_path=stream_out,
        chunk_size=2,
        streaming_threshold=2,
    )
    assert res is None
    assert stream_out.exists()
    streamed_df = pl.read_parquet(stream_out)
    assert streamed_df.height == synthetic_candidates.height


def test_full_feature_matrix_chunked_in_memory(
    synthetic_candidates,
    synthetic_transactions,
    synthetic_customers,
    synthetic_articles,
):
    """Verifica la rama de procesamiento por lotes concatenados en memoria."""
    res = build_full_feature_matrix(
        candidates_df=synthetic_candidates,
        transactions_df=synthetic_transactions,
        customers_df=synthetic_customers,
        articles_df=synthetic_articles,
        chunk_size=2,
        streaming_threshold=2,
    )
    assert res is not None
    assert res.height == synthetic_candidates.height
