"""Módulo de extracción estratificada de muestra representativa (Zero-Orphans & < 3 MB).

Implementa la arquitectura de muestreo defensivo:
1. Ingesta selectiva con DuckDB directamente contra disco (evita cargar 3.49 GB en RAM).
2. Filtrado temporal en las últimas 5 semanas y selección de clientes con al menos 2 transacciones.
3. Muestreo estratificado por frecuencia de compra (800 con 2 tx, 700 con 3-4 tx, 350 con 5-6 tx, 150 con >=7 tx).
4. Extracción relacional consistente de 3 entidades (Transactions, Customers, Articles) con CERO huérfanos.
5. Validación estricta de integridad referencial y comprobación de cota de tamaño para Git (< 3.0 MB).
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import NamedTuple

import duckdb
import polars as pl

from config.settings import (
    DATA_RAW_DIR,
    DATA_SAMPLE_DIR,
    RANDOM_SEED,
    TEMPORAL_WINDOW_WEEKS,
)

logger = logging.getLogger(__name__)


class SampleExtractionResult(NamedTuple):
    """Métricas y estadísticas de la muestra relacional generada."""

    n_transactions: int
    n_customers: int
    n_articles: int
    n_unique_customers_in_tx: int
    n_unique_articles_in_tx: int
    transactions_kb: float
    customers_kb: float
    articles_kb: float
    total_size_mb: float


def extract_consistent_sample(
    n_customers: int = 2000,
    raw_dir: Path = DATA_RAW_DIR,
    output_dir: Path = DATA_SAMPLE_DIR,
    weeks_window: int = TEMPORAL_WINDOW_WEEKS,
    seed: int = RANDOM_SEED,
    max_size_mb: float = 3.0,
) -> SampleExtractionResult:
    """Extrae un subconjunto relacionalmente consistente sin claves foráneas huérfanas.

    Parameters
    ----------
    n_customers : int, optional
        Número exacto de clientes a muestrear (default 2000).
    raw_dir : Path, optional
        Directorio donde residen los CSVs crudos de H&M.
    output_dir : Path, optional
        Directorio destino para almacenar los 3 CSVs generados.
    weeks_window : int, optional
        Ventana temporal de historia reciente a considerar (default 5 semanas).
    seed : int, optional
        Semilla para el hashing determinista (default 42).
    max_size_mb : float, optional
        Cota máxima admisible de almacenamiento combinado en disco (default 3.0 MB).

    Returns
    -------
    SampleExtractionResult
        Tupla con las métricas y pesos de los archivos generados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tx_raw = raw_dir / "transactions_train.csv"
    cust_raw = raw_dir / "customers.csv"
    art_raw = raw_dir / "articles.csv"

    # Comprobación de existencia de archivos fuente
    assert tx_raw.exists(), f"Error: No se encontró el archivo de transacciones en '{tx_raw}'"
    assert cust_raw.exists(), f"Error: No se encontró el archivo de clientes en '{cust_raw}'"
    assert art_raw.exists(), f"Error: No se encontró el archivo de artículos en '{art_raw}'"

    logger.info("-> Conectando con motor DuckDB embebido (lectura out-of-core)...")
    con = duckdb.connect()

    # Calcular la fecha máxima del dataset y el corte temporal de 5 semanas
    max_date_query = f"SELECT max(CAST(t_dat AS DATE)) FROM read_csv_auto('{tx_raw.as_posix()}')"
    max_date: datetime.date = con.execute(max_date_query).fetchone()[0]
    cutoff_date: datetime.date = max_date - datetime.timedelta(weeks=weeks_window)

    logger.info(
        f"-> Ventana temporal seleccionada: [{cutoff_date} a {max_date}] ({weeks_window} semanas)"
    )
    logger.info(
        f"-> Muestreando de forma estratificada {n_customers} clientes con >= 2 compras (semilla={seed})..."
    )

    # Muestreo estratificado por niveles de actividad transaccional
    # Proporciones estratificadas que reflejan la distribución real de retail:
    # Estrato 1: 40% (800) compradores con 2 compras
    # Estrato 2: 35% (700) compradores con 3-4 compras
    # Estrato 3: 17.5% (350) compradores con 5-6 compras
    # Estrato 4: 7.5% (150) compradores frecuentes (>=7 compras)
    n_s1 = int(n_customers * 0.40)
    n_s2 = int(n_customers * 0.35)
    n_s3 = int(n_customers * 0.175)
    n_s4 = n_customers - (n_s1 + n_s2 + n_s3)

    sampling_query = f"""
        WITH recent_transactions AS (
            SELECT customer_id, count(*) AS purchase_count
            FROM read_csv_auto('{tx_raw.as_posix()}')
            WHERE CAST(t_dat AS DATE) >= DATE '{cutoff_date}'
            GROUP BY customer_id
            HAVING count(*) >= 2
        ),
        customer_strata AS (
            SELECT customer_id, purchase_count,
                CASE
                    WHEN purchase_count = 2 THEN 'stratum_1'
                    WHEN purchase_count BETWEEN 3 AND 4 THEN 'stratum_2'
                    WHEN purchase_count BETWEEN 5 AND 6 THEN 'stratum_3'
                    ELSE 'stratum_4'
                END AS stratum
            FROM recent_transactions
        ),
        sampled AS (
            (SELECT customer_id FROM customer_strata WHERE stratum = 'stratum_1' ORDER BY hash(customer_id || '{seed}') LIMIT {n_s1})
            UNION ALL
            (SELECT customer_id FROM customer_strata WHERE stratum = 'stratum_2' ORDER BY hash(customer_id || '{seed}') LIMIT {n_s2})
            UNION ALL
            (SELECT customer_id FROM customer_strata WHERE stratum = 'stratum_3' ORDER BY hash(customer_id || '{seed}') LIMIT {n_s3})
            UNION ALL
            (SELECT customer_id FROM customer_strata WHERE stratum = 'stratum_4' ORDER BY hash(customer_id || '{seed}') LIMIT {n_s4})
        )
        SELECT customer_id FROM sampled
    """
    sampled_customers_df = con.execute(sampling_query).pl()
    assert sampled_customers_df.height == n_customers, (
        f"Se esperaban {n_customers} clientes, pero DuckDB devolvió {sampled_customers_df.height}"
    )

    con.register("sampled_customers", sampled_customers_df)

    # Extracción de Transacciones asociadas
    logger.info("-> Extrayendo transacciones de los clientes seleccionados...")
    tx_extract_query = f"""
        SELECT
            CAST(t.t_dat AS VARCHAR) AS t_dat,
            t.customer_id,
            CAST(t.article_id AS VARCHAR) AS article_id,
            t.price,
            t.sales_channel_id
        FROM read_csv_auto('{tx_raw.as_posix()}') t
        INNER JOIN sampled_customers sc ON t.customer_id = sc.customer_id
        WHERE CAST(t.t_dat AS DATE) >= DATE '{cutoff_date}'
        ORDER BY t.t_dat ASC, t.customer_id ASC
    """
    sample_tx_df = con.execute(tx_extract_query).pl()
    assert sample_tx_df.height > 0, "El subconjunto de transacciones extraído no puede estar vacío"

    con.register("sample_transactions", sample_tx_df)

    # Extracción de Clientes (perfiles exactos de los 2.000 muestreados)
    logger.info("-> Extrayendo perfiles demográficos de clientes...")
    cust_extract_query = f"""
        SELECT c.*
        FROM read_csv_auto('{cust_raw.as_posix()}') c
        INNER JOIN sampled_customers sc ON c.customer_id = sc.customer_id
        ORDER BY c.customer_id ASC
    """
    sample_cust_df = con.execute(cust_extract_query).pl()
    assert sample_cust_df.height == n_customers, (
        f"Discrepancia en clientes: {sample_cust_df.height} vs esperado {n_customers}"
    )

    # Extracción de Artículos (únicamente los comprados por los clientes en la muestra)
    logger.info("-> Extrayendo metadatos de artículos comprados...")
    art_extract_query = f"""
        SELECT a.*
        FROM read_csv_auto('{art_raw.as_posix()}') a
        WHERE a.article_id IN (
            SELECT DISTINCT article_id
            FROM sample_transactions
        )
        ORDER BY a.article_id ASC
    """
    sample_art_df = con.execute(art_extract_query).pl()
    assert sample_art_df.height > 0, "El catálogo de artículos filtrado no puede estar vacío"

    # COMPROBACIÓN DE INTEGRIDAD RELACIONAL (SIN REGISTROS HUÉRFANOS)
    logger.info("-> Verificando integridad referencial (auditoría anti-huérfanos)...")
    tx_cust_set = set(sample_tx_df["customer_id"].to_list())
    cust_set = set(sample_cust_df["customer_id"].to_list())
    tx_art_set = set(sample_tx_df["article_id"].to_list())
    art_set = set(sample_art_df["article_id"].to_list())

    # Aserción 1: Todo cliente en transacciones existe en customers y viceversa
    assert tx_cust_set == cust_set, (
        f"Huérfanos detectados en clientes! En TX: {len(tx_cust_set)}, en CUST: {len(cust_set)}"
    )

    # Aserción 2: Todo artículo comprado en transacciones existe en articles
    missing_articles = tx_art_set - art_set
    assert len(missing_articles) == 0, (
        f"Huérfanos detectados en artículos! {len(missing_articles)} artículos en TX no existen en sample_articles"
    )

    # Aserción 3: Cada cliente tiene al menos 2 compras
    tx_per_user = sample_tx_df.group_by("customer_id").len()
    min_tx = tx_per_user.select(pl.col("len").min()).item()
    assert min_tx >= 2, f"Error: Se detectaron clientes con menos de 2 compras (mínimo: {min_tx})"

    sample_tx_path = output_dir / "sample_transactions.csv"
    sample_cust_path = output_dir / "sample_customers.csv"
    sample_art_path = output_dir / "sample_articles.csv"

    sample_tx_df.write_csv(sample_tx_path)
    sample_cust_df.write_csv(sample_cust_path)
    sample_art_df.write_csv(sample_art_path)

    # Control de tamaño en disco (< 3.0 MB para Git)
    tx_bytes = sample_tx_path.stat().st_size
    cust_bytes = sample_cust_path.stat().st_size
    art_bytes = sample_art_path.stat().st_size

    tx_kb = tx_bytes / 1024.0
    cust_kb = cust_bytes / 1024.0
    art_kb = art_bytes / 1024.0
    total_mb = (tx_bytes + cust_bytes + art_bytes) / (1024.0 * 1024.0)

    assert total_mb <= max_size_mb, (
        f"Violación de presupuesto de Git: tamaño total ({total_mb:.2f} MB) excede {max_size_mb} MB"
    )

    return SampleExtractionResult(
        n_transactions=sample_tx_df.height,
        n_customers=sample_cust_df.height,
        n_articles=sample_art_df.height,
        n_unique_customers_in_tx=len(tx_cust_set),
        n_unique_articles_in_tx=len(tx_art_set),
        transactions_kb=tx_kb,
        customers_kb=cust_kb,
        articles_kb=art_kb,
        total_size_mb=total_mb,
    )
