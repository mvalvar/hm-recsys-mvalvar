"""Suite de Pruebas Unitarias para Preprocesamiento y Calidad de Datos (Fase 1).

Valida:
- Contratos de esquema y downcasting out-of-core (Polars/Parquet ZSTD).
- Límites de ventana temporal y prevención de fugas de información (zero-leakage).
- Validación de rangos numéricos de precios y edades.
- Biyección estricta en el mapeo de identificadores e integridad referencial.
- Disponibilidad de vectores de fallback para mitigación de cold-start.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, DATA_PROCESSED_SAMPLE_DIR  # noqa: E402
from src.candidates.generators import get_popular_fallback_items  # noqa: E402


# Fixtures para carga de artefactos
@pytest.fixture(scope="module")
def prod_artifacts():
    """Carga los artefactos de producción desde data_processed/."""
    tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
    cust_path = DATA_PROCESSED_DIR / "customers.parquet"
    art_path = DATA_PROCESSED_DIR / "articles.parquet"
    map_path = DATA_PROCESSED_DIR / "customer_id_mapping.parquet"
    agg_path = DATA_PROCESSED_DIR / "transactions_full_weekly_agg.parquet"

    if not (tx_path.exists() and cust_path.exists() and art_path.exists() and map_path.exists()):
        pytest.skip("Artefactos de producción en data_processed/ no disponibles para auditoría.")

    return {
        "transactions_5w": pl.read_parquet(tx_path),
        "customers": pl.read_parquet(cust_path),
        "articles": pl.read_parquet(art_path),
        "customer_id_mapping": pl.read_parquet(map_path),
        "weekly_agg": pl.read_parquet(agg_path) if agg_path.exists() else None,
    }


@pytest.fixture(scope="module")
def sample_artifacts():
    """Carga los artefactos de muestra desde data_processed/sample/."""
    tx_path = DATA_PROCESSED_SAMPLE_DIR / "transactions_5w.parquet"
    cust_path = DATA_PROCESSED_SAMPLE_DIR / "customers.parquet"
    art_path = DATA_PROCESSED_SAMPLE_DIR / "articles.parquet"
    map_path = DATA_PROCESSED_SAMPLE_DIR / "customer_id_mapping.parquet"

    if not (tx_path.exists() and cust_path.exists() and art_path.exists() and map_path.exists()):
        pytest.skip("Artefactos de muestra en data_processed/sample/ no disponibles.")

    return {
        "transactions_5w": pl.read_parquet(tx_path),
        "customers": pl.read_parquet(cust_path),
        "articles": pl.read_parquet(art_path),
        "customer_id_mapping": pl.read_parquet(map_path),
    }


# 1. CONTRATOS DE DATOS Y TIPOS DOWNCASTEADOS
def test_transactions_schema_and_types(prod_artifacts):
    """Verifica el contrato estricto de transactions_5w.parquet."""
    tx = prod_artifacts["transactions_5w"]
    expected_cols = ["t_dat", "customer_idx", "article_id", "price", "sales_channel_id"]
    assert tx.columns == expected_cols, f"Columnas inesperadas: {tx.columns}"

    # Downcasting estricto para ahorro de RAM
    assert tx.schema["t_dat"] == pl.Date, "t_dat debe ser pl.Date"
    assert tx.schema["customer_idx"] == pl.Int32, "customer_idx debe ser pl.Int32"
    assert tx.schema["article_id"] == pl.Int32, "article_id debe ser pl.Int32"
    assert tx.schema["price"] == pl.Float32, "price debe ser pl.Float32"
    assert tx.schema["sales_channel_id"] == pl.Int8, "sales_channel_id debe ser pl.Int8"


def test_customers_schema_and_types(prod_artifacts):
    """Verifica el contrato estricto de customers.parquet."""
    cust = prod_artifacts["customers"]
    expected_cols = [
        "customer_idx",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "age_bin",
        "postal_code",
    ]
    assert cust.columns == expected_cols, f"Columnas inesperadas: {cust.columns}"

    assert cust.schema["customer_idx"] == pl.Int32
    assert cust.schema["FN"] == pl.Int8
    assert cust.schema["Active"] == pl.Int8
    assert cust.schema["club_member_status"] == pl.Categorical
    assert cust.schema["fashion_news_frequency"] == pl.Categorical
    assert cust.schema["age"] == pl.Float32
    assert cust.schema["age_bin"] == pl.Categorical
    assert cust.schema["postal_code"] == pl.String


def test_articles_schema_and_types(prod_artifacts):
    """Verifica el contrato estricto de articles.parquet."""
    art = prod_artifacts["articles"]
    assert "article_id" in art.columns
    assert art.schema["article_id"] == pl.Int32
    # El catálogo completo oficial tiene exactamente 105.542 artículos
    assert art.height == 105_542, f"Se esperaban 105.542 artículos, se encontraron {art.height}"


# 2. INMUNIDAD TEMPORAL Y CERO FUGAS AL FUTURO (ZERO-LEAKAGE)
def test_temporal_window_bounds(prod_artifacts):
    """Verifica que transactions_5w cubra exactamente 5 semanas sin registros futuros."""
    tx = prod_artifacts["transactions_5w"]
    min_date = tx.select(pl.col("t_dat").min()).item()
    max_date = tx.select(pl.col("t_dat").max()).item()

    # Fecha máxima histórica de H&M: 2020-09-22
    assert max_date == datetime.date(2020, 9, 22), f"Fecha máxima inesperada: {max_date}"

    # Ventana de 5 semanas (35 días): inicio en 2020-08-18
    expected_cutoff = datetime.date(2020, 8, 18)
    assert min_date >= expected_cutoff, f"Fecha mínima {min_date} es anterior al corte {expected_cutoff}"

    # Aserción de ausencia de nulos en marcas temporales
    assert tx.filter(pl.col("t_dat").is_null()).height == 0


def test_weekly_agg_temporal_coherence(prod_artifacts):
    """Verifica que la agregación histórica tenga coherencia monótona de fechas."""
    agg = prod_artifacts["weekly_agg"]
    if agg is None:
        pytest.skip("transactions_full_weekly_agg.parquet no presente.")

    assert agg.height >= 100, f"Se esperaban al menos 100 semanas, encontradas {agg.height}"
    # Validar que las fechas sean monótonas crecientes
    dates = agg["week_date"].to_list()
    assert dates == sorted(dates), "Las semanas en weekly_agg no están ordenadas cronológicamente"


# 3. CALIDAD DE DATOS Y RANGO DE VALORES
def test_prices_positive_no_false_zeros(prod_artifacts):
    """Comprueba que no existan precios <= 0 ni valores atípicos imposibles."""
    tx = prod_artifacts["transactions_5w"]
    non_positive = tx.filter(pl.col("price") <= 0.0)
    assert non_positive.height == 0, f"Detectadas {non_positive.height} transacciones con precio <= 0"

    min_price = tx.select(pl.col("price").min()).item()
    max_price = tx.select(pl.col("price").max()).item()
    assert min_price > 0.0, f"Precio mínimo inválido: {min_price}"
    assert max_price < 1.0, f"Precio máximo inesperado (> 1.0 normalizado en H&M): {max_price}"


def test_customer_demographic_quality(prod_artifacts):
    """Comprueba la validez de las edades imputadas y la ausencia de ceros falsos."""
    cust = prod_artifacts["customers"]

    # Edades semánticamente válidas (entre 16 y 100 años)
    invalid_ages = cust.filter((pl.col("age") < 16.0) | (pl.col("age") > 100.0))
    assert invalid_ages.height == 0, f"Detectadas {invalid_ages.height} edades fuera de rango [16, 100]"

    # Aserción de completitud en metadatos obligatorios
    assert cust.filter(pl.col("age").is_null()).height == 0
    assert cust.filter(pl.col("age_bin").is_null()).height == 0
    assert cust.filter(pl.col("FN").is_null()).height == 0
    assert cust.filter(pl.col("Active").is_null()).height == 0

    # Categorías imputadas correctamente
    status_vals = cust["club_member_status"].unique().to_list()
    assert "UNKNOWN" in status_vals or "ACTIVE" in status_vals


# 4. INTEGRIDAD REFERENCIAL Y MAPEO BIYECTIVO
def test_bijective_customer_mapping(prod_artifacts):
    """Verifica que customer_id <-> customer_idx sea una biyección estricta 1:1."""
    mapping = prod_artifacts["customer_id_mapping"]
    n_rows = mapping.height

    # Unicidad absoluta de ambas claves
    assert mapping["customer_id"].n_unique() == n_rows, "Existen customer_id duplicados en el mapeo"
    assert mapping["customer_idx"].n_unique() == n_rows, "Existen customer_idx duplicados en el mapeo"

    # Rango contiguo estricto [0, N-1] sin huecos
    min_idx = mapping.select(pl.col("customer_idx").min()).item()
    max_idx = mapping.select(pl.col("customer_idx").max()).item()
    assert min_idx == 0, f"customer_idx no inicia en 0 (inicia en {min_idx})"
    assert max_idx == n_rows - 1, f"customer_idx tiene saltos (max={max_idx}, n_rows={n_rows})"

    assert mapping.filter(pl.col("customer_id").is_null()).height == 0
    assert mapping.filter(pl.col("customer_idx").is_null()).height == 0


def test_referential_integrity_transactions_and_customers(prod_artifacts):
    """Toda compra debe corresponder a un customer_idx existente y a un artículo del catálogo."""
    tx = prod_artifacts["transactions_5w"]
    cust = prod_artifacts["customers"]
    art = prod_artifacts["articles"]
    mapping = prod_artifacts["customer_id_mapping"]

    # Clientes en transacciones pertenecen a customer_id_mapping
    tx_cust_indices = set(tx["customer_idx"].unique().to_list())
    map_cust_indices = set(mapping["customer_idx"].to_list())
    orphan_customers = tx_cust_indices - map_cust_indices
    assert len(orphan_customers) == 0, f"Hay {len(orphan_customers)} clientes huérfanos en transactions_5w"

    # Clientes en customers.parquet coinciden exactamente con los de transacciones
    cust_table_indices = set(cust["customer_idx"].to_list())
    assert tx_cust_indices == cust_table_indices, "Discrepancia entre clientes en transacciones y customers.parquet"

    # Todo artículo comprado pertenece a articles.parquet
    tx_art_ids = set(tx["article_id"].unique().to_list())
    catalog_art_ids = set(art["article_id"].to_list())
    orphan_articles = tx_art_ids - catalog_art_ids
    assert len(orphan_articles) == 0, f"Hay {len(orphan_articles)} artículos comprados que no existen en el catálogo"


# 5. PARIDAD ENTRENAMIENTO-INFERENCIA Y RESILIENCIA ANTE COLD-START
def test_serving_fallback_resilience(prod_artifacts):
    """Verifica que el generador de fallback de inferencia funcione sobre transactions_5w."""
    fallback_items = get_popular_fallback_items(top_k=12, days_window=7, lambda_decay=0.05)
    assert len(fallback_items) == 12, f"Se esperaban 12 artículos de fallback, se obtuvieron {len(fallback_items)}"
    assert len(set(fallback_items)) == 12, "El fallback contiene artículos duplicados"

    # Verificar que los artículos recomendados pertenezcan al catálogo (artículo como entero canónico)
    catalog = set(prod_artifacts["articles"]["article_id"].to_list())
    for item in fallback_items:
        assert int(item) in catalog, f"Artículo de fallback {item} no está en el catálogo"


# 6. PARIDAD MULTIMODO (SAMPLE VS PRODUCCIÓN)
def test_schema_parity_sample_vs_prod(prod_artifacts, sample_artifacts):
    """Verifica que la muestra (--sample) tenga estructura compatible con producción."""
    for table_name in ["transactions_5w", "customers", "articles", "customer_id_mapping"]:
        prod_df = prod_artifacts[table_name]
        sample_df = sample_artifacts[table_name]

        # Mismas columnas en idéntico orden
        assert prod_df.columns == sample_df.columns, (
            f"Discrepancia de columnas en {table_name}: {prod_df.columns} vs {sample_df.columns}"
        )

        # Compatibilidad de tipos
        for col_name in prod_df.columns:
            prod_type = prod_df.schema[col_name]
            sample_type = sample_df.schema[col_name]

            # Ambos deben ser del mismo supertipo (ambos enteros, ambos floats, ambos Date, etc.)
            if prod_type.is_integer():
                assert sample_type.is_integer(), f"{table_name}.{col_name} debe ser entero en sample"
            elif prod_type.is_float():
                assert sample_type.is_float(), f"{table_name}.{col_name} debe ser float en sample"
            else:
                assert prod_type == sample_type, (
                    f"Discrepancia de tipo en {table_name}.{col_name}: {prod_type} vs {sample_type}"
                )
