"""Suite de Pruebas Unitarias y Defensivas para Ingeniería de Características.

Valida:
1. Exactitud matemática y ausencia de nulos en los 4 namespaces disjuntos (~35 features).
2. Cumplimiento estricto del tipado en memoria (Int8, Int16, Int32, Float32 - Cero Int64/Float64).
3. Inmunidad a fugas de datos temporales (Zero Data Leakage).
4. Valores centinela e imputaciones documentadas (999 en días sin compra, 0 en afinidad, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR
from src.features.builder import (
    build_article_features,
    build_candidate_meta_features,
    build_full_feature_matrix,
    build_interaction_features,
    build_user_features,
)
from src.utils.validation import split_transactions_temporal


@pytest.fixture(scope="module")
def loaded_data():
    """Carga los DataFrames optimizados de prueba desde data_processed/."""
    cand = pl.read_parquet(DATA_PROCESSED_DIR / "candidates.parquet").head(50000)
    tx = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    cust = pl.read_parquet(DATA_PROCESSED_DIR / "customers.parquet")
    art = pl.read_parquet(DATA_PROCESSED_DIR / "articles.parquet")
    return cand, tx, cust, art


def test_user_features_namespace(loaded_data):
    """Prueba unitaria para Namespace 1: Características de Usuario (9 features)."""
    _, tx, cust, _ = loaded_data
    u_feats = build_user_features(tx, cust)

    expected_user_cols = [
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
    assert u_feats.columns == expected_user_cols, f"Columnas inesperadas: {u_feats.columns}"

    assert u_feats.schema["u_total_transactions"] == pl.Int32
    assert u_feats.schema["u_unique_articles"] == pl.Int32
    assert u_feats.schema["u_mean_price"] == pl.Float32
    assert u_feats.schema["u_std_price"] == pl.Float32
    assert u_feats.schema["u_online_ratio"] == pl.Float32
    assert u_feats.schema["u_age"] == pl.Int16
    assert u_feats.schema["u_club_status"] == pl.Int8
    assert u_feats.schema["u_fashion_news"] == pl.Int8
    assert u_feats.schema["u_last_purchase_days_ago"] == pl.Int16

    assert all(u_feats[c].null_count() == 0 for c in expected_user_cols)

    # Rango de valores válidos
    assert u_feats["u_club_status"].min() >= 0 and u_feats["u_club_status"].max() <= 3
    assert u_feats["u_fashion_news"].min() >= 0 and u_feats["u_fashion_news"].max() <= 2
    assert u_feats["u_online_ratio"].min() >= 0.0 and u_feats["u_online_ratio"].max() <= 1.0
    assert (
        u_feats["u_last_purchase_days_ago"].min() >= 0
        and u_feats["u_last_purchase_days_ago"].max() <= 999
    )


def test_article_features_namespace(loaded_data):
    """Prueba unitaria para Namespace 2: Características de Artículo (9 features)."""
    _, tx, _, art = loaded_data
    a_feats = build_article_features(tx, art)

    expected_art_cols = [
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
    assert a_feats.columns == expected_art_cols, f"Columnas inesperadas: {a_feats.columns}"

    assert a_feats.schema["a_sales_count"] == pl.Int32
    assert a_feats.schema["a_unique_customers"] == pl.Int32
    assert a_feats.schema["a_mean_price"] == pl.Float32
    assert a_feats.schema["a_sales_decayed"] == pl.Float32
    assert a_feats.schema["a_is_recent_introduction"] == pl.Int8
    assert a_feats.schema["a_product_type_no"] == pl.Int32
    assert a_feats.schema["a_graphical_appearance_no"] == pl.Int32
    assert a_feats.schema["a_colour_group_code"] == pl.Int16
    assert a_feats.schema["a_department_no"] == pl.Int32

    assert all(a_feats[c].null_count() == 0 for c in expected_art_cols)
    assert (
        a_feats["a_is_recent_introduction"].min() >= 0
        and a_feats["a_is_recent_introduction"].max() <= 1
    )


def test_interaction_features_namespace(loaded_data):
    """Prueba unitaria para Namespace 3: Interacción Usuario x Artículo (8 features)."""
    cand, tx, cust, art = loaded_data
    u_feats = build_user_features(tx, cust)
    a_feats = build_article_features(tx, art)
    uxa_feats = build_interaction_features(cand, tx, u_feats, a_feats)

    expected_uxa_cols = [
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
    assert uxa_feats.columns == expected_uxa_cols, f"Columnas inesperadas: {uxa_feats.columns}"

    assert uxa_feats.schema["uxa_repurchase_count"] == pl.Int16
    assert uxa_feats.schema["uxa_days_since_last_purchase"] == pl.Int16
    assert uxa_feats.schema["uxa_dept_affinity"] == pl.Int16
    assert uxa_feats.schema["uxa_price_diff"] == pl.Float32
    assert uxa_feats.schema["uxa_price_ratio"] == pl.Float32
    assert uxa_feats.schema["uxa_bought_dept_before"] == pl.Int8
    assert uxa_feats.schema["uxa_channel_affinity"] == pl.Float32
    assert uxa_feats.schema["uxa_is_favorite_dept"] == pl.Int8

    assert all(uxa_feats[c].null_count() == 0 for c in expected_uxa_cols)

    # Validar centinela 999
    never_bought = uxa_feats.filter(pl.col("uxa_repurchase_count") == 0)
    assert (never_bought["uxa_days_since_last_purchase"] == 999).all()


def test_meta_features_namespace(loaded_data):
    """Prueba unitaria para Namespace 4: Meta-Features de Origen de Candidatos (13 features)."""
    cand, _, _, _ = loaded_data
    meta_feats = build_candidate_meta_features(cand)

    expected_meta_cols = [
        "customer_idx",
        "article_id",
        "is_R1",
        "is_R2",
        "is_R3",
        "is_R4",
        "is_R5",
        "is_R6",
        "is_R7",
        "is_R8",
        "n_sources",
        "best_rank",
        "source_diversity_score",
        "is_personal_candidate",
        "is_exploration_candidate",
    ]
    assert meta_feats.columns == expected_meta_cols, f"Columnas inesperadas: {meta_feats.columns}"

    for i in range(1, 9):
        assert meta_feats.schema[f"is_R{i}"] == pl.Int8
    assert meta_feats.schema["n_sources"] == pl.Int8
    assert meta_feats.schema["best_rank"] == pl.Int16
    assert meta_feats.schema["source_diversity_score"] == pl.Float32
    assert meta_feats.schema["is_personal_candidate"] == pl.Int8
    assert meta_feats.schema["is_exploration_candidate"] == pl.Int8

    assert all(meta_feats[c].null_count() == 0 for c in expected_meta_cols)

    # Consistencia de flags
    sum_flags = sum(meta_feats[f"is_R{i}"] for i in range(1, 9))
    assert (meta_feats["n_sources"] == sum_flags).all()


def test_full_feature_matrix_assembly_and_memory(loaded_data):
    """Prueba de integración: Ensamblado disjunto, 41 columnas totales y cero nulos."""
    cand, tx, cust, art = loaded_data
    sample_cand = cand.head(10000)
    feat_matrix = build_full_feature_matrix(sample_cand, tx, cust, art)

    # 2 claves + 39 características = 41 columnas
    assert len(feat_matrix.columns) == 41
    assert feat_matrix.height == sample_cand.height

    # Validación de completitud sin nulos
    for col in feat_matrix.columns:
        null_count = feat_matrix[col].null_count()
        assert null_count == 0, f"Columna {col} contiene {null_count} nulos!"

    # Sin tipos pesados (cero Float64 y cero Int64)
    for col, dt in feat_matrix.schema.items():
        assert dt not in (pl.Float64, pl.Int64), (
            f"Columna {col} tiene tipo pesado no downcasteado: {dt}"
        )


def test_zero_temporal_data_leakage(loaded_data):
    """Prueba de inmunidad a fugas de datos temporales (Zero Data Leakage)."""
    cand, tx, cust, art = loaded_data
    sample_cand = cand.head(10000)

    split = split_transactions_temporal(tx, val_days=7)
    train_tx = split.train_df
    val_tx = split.val_df

    # Construir features exclusivamente sobre train_tx
    feat_train = build_full_feature_matrix(sample_cand, train_tx, cust, art)

    # Validar que ningún artículo con compras únicamente en val_tx tenga a_sales_count > 0 en feat_train
    val_only_articles = (
        val_tx.select("article_id")
        .unique()
        .join(train_tx.select("article_id").unique(), on="article_id", how="anti")["article_id"]
        .to_list()
    )

    if val_only_articles:
        sample_future_art = val_only_articles[0]
        row_feat = feat_train.filter(pl.col("article_id") == sample_future_art)
        if row_feat.height > 0:
            assert row_feat[0, "a_sales_count"] == 0, (
                "¡Fuga detectada! a_sales_count > 0 para ítem futuro"
            )
            assert row_feat[0, "a_sales_decayed"] == 0.0, (
                "¡Fuga detectada! a_sales_decayed > 0 para ítem futuro"
            )


if __name__ == "__main__":
    print("=" * 70)
    print("  EJECUTANDO SUITE DE TESTS UNITARIOS: TEST_FEATURES.PY")
    print("=" * 70)
    cand_df = pl.read_parquet(DATA_PROCESSED_DIR / "candidates.parquet")
    tx_df = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    cust_df = pl.read_parquet(DATA_PROCESSED_DIR / "customers.parquet")
    art_df = pl.read_parquet(DATA_PROCESSED_DIR / "articles.parquet")
    data = (cand_df, tx_df, cust_df, art_df)

    test_user_features_namespace(data)
    print("[PASS] Namespace 1 (Usuario): 9 features validadas.")

    test_article_features_namespace(data)
    print("[PASS] Namespace 2 (Artículo): 9 features validadas.")

    test_interaction_features_namespace(data)
    print("[PASS] Namespace 3 (Interacción): 8 features validadas.")

    test_meta_features_namespace(data)
    print("[PASS] Namespace 4 (Meta-features): 13 features validadas.")

    test_full_feature_matrix_assembly_and_memory(data)
    print(
        "[PASS] Ensamblado Total: 41 columnas (2 claves + 39 features), 0 nulos, downcasting verificado."
    )

    test_zero_temporal_data_leakage(data)
    print("[PASS] Zero Data Leakage Temporal: Inmunidad arquitectónica confirmada.")

    print("=" * 70)
    print("[OK] Todos los tests de ingeniería de características superados.")
    print("=" * 70)
