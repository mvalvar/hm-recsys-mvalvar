"""Pipeline de preprocesamiento e ingesta out-of-core (DuckDB -> Polars -> Parquet).

Implementa la arquitectura de ingesta eficiente para el TFM:
1. Lectura Out-of-Core con DuckDB: Consulta directa sobre disco limitando memoria a 6 GB.
2. Agregación Histórica Única: Genera transactions_full_weekly_agg.parquet (104 semanas) en un solo pase.
3. Filtrado Temporal Estricto: Aísla las 5 semanas recientes (TEMPORAL_WINDOW_WEEKS) sin fugas.
4. Downcasting Defensivo:
   - customer_id (hash 64B) -> customer_idx (Int32 contiguo): Ahorro del 93.75% de RAM.
   - article_id (string 10B) -> Int32: Ahorro del 60% sin pérdida semántica.
   - price -> Float32, sales_channel_id -> Int8, t_dat -> Date.
5. Imputación e Ingeniería Demográfica: Imputación de edad con mediana y categorización por age_bin.
6. Serialización Columnar: Checkpoints Parquet con compresión ZSTD.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import NamedTuple

import duckdb
import polars as pl

from config.settings import (
    AGE_BIN_LABELS,
    AGE_BINS,
    DATA_RAW_DIR,
    DATA_SAMPLE_DIR,
    TEMPORAL_WINDOW_WEEKS,
    get_processed_dir,
)
from src.utils.memory import downcast_polars, log_memory_usage

logger = logging.getLogger(__name__)


class PreprocessSummary(NamedTuple):
    """Resumen de métricas de la fase de ingesta y compresión."""

    transactions_rows: int
    customers_rows: int
    articles_rows: int
    min_date: str
    max_date: str
    memory_rss_mb: float
    raw_size_mb: float
    parquet_size_mb: float
    compression_ratio_pct: float


def run_preprocess(
    use_sample: bool = False,
    raw_dir: Path = DATA_RAW_DIR,
    sample_dir: Path = DATA_SAMPLE_DIR,
    output_dir: Path | None = None,
    weeks_window: int = TEMPORAL_WINDOW_WEEKS,
) -> PreprocessSummary:
    """Ejecuta el pipeline completo de preprocesamiento e ingesta out-of-core.

    Parameters
    ----------
    use_sample : bool, optional
        Si es True, procesa los CSVs de data_sample/; si es False, los datos crudos masivos.
    raw_dir : Path, optional
        Directorio donde reside el dataset original de H&M (h&m_data/).
    sample_dir : Path, optional
        Directorio donde reside la muestra representativa (data_sample/).
    output_dir : Path, optional
        Directorio destino para los checkpoints Parquet (resuelve dinámicamente según use_sample si es None).
    weeks_window : int, optional
        Número de semanas recientes a extraer para entrenamiento y ranking (default 5).

    Returns
    -------
    PreprocessSummary
        Métricas exhaustivas de filas, fechas, consumo de RAM y compresión.
    """
    if output_dir is None:
        output_dir = get_processed_dir(use_sample=use_sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = log_memory_usage("Inicio de preprocesamiento")

    # Resolución de rutas de entrada
    source_dir = sample_dir if use_sample else raw_dir
    prefix = "sample_" if use_sample else ""
    tx_file = (
        source_dir / f"{prefix}transactions_train.csv"
        if not use_sample
        else source_dir / "sample_transactions.csv"
    )
    cust_file = (
        source_dir / f"{prefix}customers.csv"
        if not use_sample
        else source_dir / "sample_customers.csv"
    )
    art_file = (
        source_dir / f"{prefix}articles.csv"
        if not use_sample
        else source_dir / "sample_articles.csv"
    )

    assert tx_file.exists(), f"Error crítico: Archivo transacciones no encontrado en '{tx_file}'"
    assert cust_file.exists(), f"Error crítico: Archivo clientes no encontrado en '{cust_file}'"
    assert art_file.exists(), f"Error crítico: Archivo artículos no encontrado en '{art_file}'"

    raw_tx_bytes = tx_file.stat().st_size
    raw_size_mb = raw_tx_bytes / (1024.0 * 1024.0)

    # Inicialización de DuckDB con control de presupuesto de RAM (máximo 6 GB)
    logger.info("-> Conectando con DuckDB embebido (modo streaming out-of-core)...")
    con = duckdb.connect()
    con.execute("SET memory_limit = '6GB';")
    con.execute("SET threads = 6;")

    # Agregación Histórica Semanal (104 semanas) y Detección de Fecha Máxima
    # Agregación out-of-core de métricas macro en una sola pasada sobre transacciones
    logger.info("-> Calculando agregación semanal histórica y fecha de corte...")
    weekly_agg_query = f"""
        CREATE OR REPLACE TABLE weekly_agg AS
        SELECT
            DATE_TRUNC('week', CAST(t_dat AS DATE)) AS week_date,
            COUNT(*) AS n_transactions,
            COUNT(DISTINCT article_id) AS n_unique_articles,
            COUNT(DISTINCT customer_id) AS n_unique_customers,
            ROUND(SUM(price), 2) AS total_revenue,
            ROUND(AVG(price), 4) AS avg_price,
            MAX(CAST(t_dat AS DATE)) AS max_t_dat,
            MIN(CAST(t_dat AS DATE)) AS min_t_dat
        FROM read_csv_auto('{tx_file.as_posix()}', header=True)
        GROUP BY 1
        ORDER BY 1;
    """
    con.execute(weekly_agg_query)

    # Extraer fechas y guardar agregación histórica
    weekly_agg_df = con.execute("SELECT * FROM weekly_agg ORDER BY week_date ASC").pl()
    weekly_agg_df.write_parquet(
        output_dir / "transactions_full_weekly_agg.parquet", compression="zstd"
    )

    max_date: datetime.date = con.execute("SELECT max(max_t_dat) FROM weekly_agg").fetchone()[0]
    cutoff_date: datetime.date = max_date - datetime.timedelta(weeks=weeks_window)
    logger.info(f"-> Ventana activa: [{cutoff_date} a {max_date}] ({weeks_window} semanas)")

    # Ingesta filtrada de las últimas 5 semanas con DuckDB
    logger.info(
        f"-> Ingestando transacciones recientes (>= {cutoff_date}) directamente a Polars..."
    )
    tx_filter_query = f"""
        SELECT
            CAST(t_dat AS DATE) AS t_dat,
            customer_id,
            CAST(article_id AS INTEGER) AS article_id,
            CAST(price AS FLOAT) AS price,
            CAST(sales_channel_id AS TINYINT) AS sales_channel_id
        FROM read_csv_auto('{tx_file.as_posix()}', header=True)
        WHERE CAST(t_dat AS DATE) >= DATE '{cutoff_date}'
    """
    transactions_df = con.execute(tx_filter_query).pl()
    assert transactions_df.height > 0, (
        "Error crítico: El conjunto de transacciones de 5 semanas está vacío"
    )

    # Generación de mapeo determinista y bidireccional: customer_id (hash 64B) -> customer_idx (Int32)
    # Compactación hash hex (64B) -> Int32 (4B) para reducir huella de usuarios a 126 MB (-93.7% RAM)
    logger.info("-> Generando mapeo determinista customer_id -> customer_idx (Int32)...")
    unique_customers = (
        transactions_df.select("customer_id")
        .unique()
        .sort("customer_id")
        .with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("customer_idx"))
    )
    unique_customers.write_parquet(output_dir / "customer_id_mapping.parquet", compression="zstd")

    # Unir customer_idx a transacciones y descartar el hash original
    logger.info("-> Aplicando mapeo a transacciones y serializando transactions_5w.parquet...")
    transactions_df = (
        transactions_df.join(unique_customers, on="customer_id", how="inner")
        .drop("customer_id")
        .select(["t_dat", "customer_idx", "article_id", "price", "sales_channel_id"])
    )

    # Validación de columnas requeridas
    assert transactions_df.filter(pl.col("customer_idx").is_null()).height == 0, (
        "Nulos en customer_idx"
    )
    assert transactions_df.filter(pl.col("article_id").is_null()).height == 0, "Nulos en article_id"
    assert transactions_df.filter(pl.col("t_dat").is_null()).height == 0, "Nulos en t_dat"

    transactions_df = downcast_polars(transactions_df)
    transactions_df.write_parquet(output_dir / "transactions_5w.parquet", compression="zstd")

    min_tx_date = transactions_df.select(pl.col("t_dat").min()).item()
    max_tx_date = transactions_df.select(pl.col("t_dat").max()).item()

    # 6b. Ingesta filtrada de 10 semanas para experimentos de horizonte temporal (A1)
    logger.info("-> Generando transactions_10w.parquet para experimentos de horizonte temporal...")
    cutoff_10w = max_date - datetime.timedelta(weeks=10)
    tx_10w_query = f"""
        SELECT
            CAST(t_dat AS DATE) AS t_dat,
            customer_id,
            CAST(article_id AS INTEGER) AS article_id,
            CAST(price AS FLOAT) AS price,
            CAST(sales_channel_id AS TINYINT) AS sales_channel_id
        FROM read_csv_auto('{tx_file.as_posix()}', header=True)
        WHERE CAST(t_dat AS DATE) >= DATE '{cutoff_10w}'
    """
    tx_10w_df = con.execute(tx_10w_query).pl()
    tx_10w_df = (
        tx_10w_df.join(unique_customers, on="customer_id", how="inner")
        .drop("customer_id")
        .select(["t_dat", "customer_idx", "article_id", "price", "sales_channel_id"])
    )
    tx_10w_df = downcast_polars(tx_10w_df)
    tx_10w_df.write_parquet(output_dir / "transactions_10w.parquet", compression="zstd")
    del tx_10w_df

    # Procesamiento de Metadatos de Clientes (customers.csv)
    logger.info(f"-> Procesando metadatos de clientes desde {cust_file.name}...")
    customers_raw_df = pl.read_csv(
        cust_file,
        columns=[
            "customer_id",
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "postal_code",
        ],
    )

    # Imputación de edad con mediana y binning demográfico estándar
    median_age = customers_raw_df.select(pl.col("age").median()).item() or 32.0
    customers_df = (
        customers_raw_df.with_columns(
            pl.col("FN").fill_null(0).cast(pl.Int8),
            pl.col("Active").fill_null(0).cast(pl.Int8),
            pl.col("club_member_status").fill_null("UNKNOWN").cast(pl.Categorical),
            pl.col("fashion_news_frequency").fill_null("NONE").cast(pl.Categorical),
            pl.col("age").fill_null(median_age).cast(pl.Float32),
        )
        .with_columns(
            pl.col("age")
            .cut(breaks=AGE_BINS[1:-1], labels=AGE_BIN_LABELS)
            .cast(pl.Categorical)
            .alias("age_bin")
        )
        .join(unique_customers, on="customer_id", how="inner")
        .drop("customer_id")
        .select(
            [
                "customer_idx",
                "FN",
                "Active",
                "club_member_status",
                "fashion_news_frequency",
                "age",
                "age_bin",
                "postal_code",
            ]
        )
    )
    customers_df = downcast_polars(customers_df)
    customers_df.write_parquet(output_dir / "customers.parquet", compression="zstd")

    # Procesamiento de Metadatos de Artículos (articles.csv)
    logger.info(f"-> Procesando metadatos de catálogo desde {art_file.name}...")
    articles_df = pl.read_csv(
        art_file,
        columns=[
            "article_id",
            "product_code",
            "prod_name",
            "product_type_no",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "department_name",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no",
            "garment_group_name",
            "detail_desc",
        ],
    ).with_columns(pl.col("article_id").cast(pl.Int32))

    articles_df = downcast_polars(articles_df)
    articles_df.write_parquet(output_dir / "articles.parquet", compression="zstd")

    # Cálculo de telemetría y compresión
    rss_end = log_memory_usage("Fin de preprocesamiento")

    parquet_files = [
        output_dir / "transactions_5w.parquet",
        output_dir / "customers.parquet",
        output_dir / "articles.parquet",
        output_dir / "customer_id_mapping.parquet",
        output_dir / "transactions_full_weekly_agg.parquet",
    ]
    parquet_bytes = sum(p.stat().st_size for p in parquet_files if p.exists())
    parquet_size_mb = parquet_bytes / (1024.0 * 1024.0)

    compression_pct = (1.0 - (parquet_bytes / raw_tx_bytes)) * 100.0

    return PreprocessSummary(
        transactions_rows=transactions_df.height,
        customers_rows=customers_df.height,
        articles_rows=articles_df.height,
        min_date=str(min_tx_date),
        max_date=str(max_tx_date),
        memory_rss_mb=rss_end,
        raw_size_mb=raw_size_mb,
        parquet_size_mb=parquet_size_mb,
        compression_ratio_pct=compression_pct,
    )
