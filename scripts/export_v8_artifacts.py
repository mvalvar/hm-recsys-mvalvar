"""Script: Precomputación y Exportación de Artefactos de Servicio V8 en Parquet ZSTD.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Genera los tres artefactos optimizados para inferencia en tiempo real en la API (Modo 2):
1. data_processed/v8_basket_affinity.parquet (Pares causales P(B|A) ponderados con n >= 3)
2. data_processed/v8_bestsellers_age.parquet (Top-12 por cohorte y global con gamma=0.12)
3. data_processed/v8_customer_history_28d.parquet (Historial de 28 días indexado por customer_idx)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, DATA_PROCESSED_SAMPLE_DIR

logger = logging.getLogger("export_v8_artifacts")


def export_v8_artifacts(
    processed_dir: Path = DATA_PROCESSED_DIR,
    ref_date: str = "2020-09-22",
    pop_decay: float = 0.12,
    window_days: int = 28,
    cooccur_window_days: int = 35,
    cooccur_min_support: int = 3,
) -> dict[str, Any]:
    start_t = time.perf_counter()
    tx_path = processed_dir / "transactions_5w.parquet"
    cust_path = processed_dir / "customers.parquet"

    assert tx_path.exists(), f"Falta {tx_path}"
    assert cust_path.exists(), f"Falta {cust_path}"

    tx_str = str(tx_path).replace("\\", "/")
    cust_str = str(cust_path).replace("\\", "/")

    con = duckdb.connect()

    logger.info("-> [1/3] Generando v8_bestsellers_age.parquet...")
    # Bestsellers multi-semana globales y por cohorte de edad
    bs_df = con.execute(f"""
        WITH tx AS (
            SELECT t.article_id,
                   datediff('day', cast(t.t_dat as date), date '{ref_date}') as days_ago,
                   coalesce(c.age_bin, 'ALL') as age_bin
            FROM read_parquet('{tx_str}') t
            LEFT JOIN read_parquet('{cust_str}') c ON t.customer_idx = c.customer_idx
            WHERE t.t_dat >= '2020-09-02' AND t.t_dat <= '{ref_date}'
        ),
        by_age AS (
            SELECT age_bin,
                   article_id,
                   SUM(POWER({pop_decay}, days_ago / 7.0)) as pop_score
            FROM tx
            GROUP BY age_bin, article_id
        ),
        ranked_age AS (
            SELECT age_bin,
                   article_id,
                   pop_score,
                   row_number() OVER (PARTITION BY age_bin ORDER BY pop_score DESC, article_id ASC) as rank
            FROM by_age
        ),
        global_pop AS (
            SELECT 'ALL' as age_bin,
                   article_id,
                   SUM(POWER({pop_decay}, days_ago / 7.0)) as pop_score
            FROM tx
            GROUP BY article_id
        ),
        ranked_global AS (
            SELECT age_bin,
                   article_id,
                   pop_score,
                   row_number() OVER (PARTITION BY age_bin ORDER BY pop_score DESC, article_id ASC) as rank
            FROM global_pop
        )
        SELECT age_bin, article_id, cast(pop_score as float) as pop_score, cast(rank as smallint) as rank
        FROM ranked_age WHERE rank <= 12
        UNION ALL
        SELECT age_bin, article_id, cast(pop_score as float) as pop_score, cast(rank as smallint) as rank
        FROM ranked_global WHERE rank <= 12
        ORDER BY age_bin, rank
    """).pl()

    bs_out = processed_dir / "v8_bestsellers_age.parquet"
    bs_df.write_parquet(bs_out, compression="zstd")
    logger.info(f"   [OK] Guardado {bs_out.name} ({len(bs_df)} filas, {bs_out.stat().st_size / 1024:.1f} KB)")

    logger.info("-> [2/3] Generando v8_basket_affinity.parquet...")
    # Co-ocurrencias causales en cesta
    cooccur_df = con.execute(f"""
        WITH recent_tx AS (
            SELECT customer_idx, t_dat, article_id,
                   datediff('day', cast(t_dat as date), date '{ref_date}') as days_ago
            FROM read_parquet('{tx_str}')
            WHERE t_dat >= date '{ref_date}' - interval '{cooccur_window_days} days'
              AND t_dat <= date '{ref_date}'
        ),
        pairs AS (
            SELECT b1.article_id as art_a, b2.article_id as art_b,
                   SUM(POWER(0.85, (b1.days_ago + b2.days_ago) / 14.0)) as pair_weight,
                   count(*) as pair_count
            FROM recent_tx b1
            JOIN recent_tx b2 ON b1.customer_idx = b2.customer_idx AND b1.t_dat = b2.t_dat AND b1.article_id != b2.article_id
            GROUP BY b1.article_id, b2.article_id
            HAVING count(*) >= {cooccur_min_support}
        ),
        ranked_pairs AS (
            SELECT art_a as article_id,
                   art_b as complement_id,
                   cast(pair_weight as float) as pair_weight,
                   cast(row_number() OVER (PARTITION BY art_a ORDER BY pair_weight DESC, art_b ASC) as smallint) as pair_rank
            FROM pairs
        )
        SELECT article_id, complement_id, pair_weight, pair_rank
        FROM ranked_pairs
        WHERE pair_rank <= 12
        ORDER BY article_id, pair_rank
    """).pl()

    cooccur_out = processed_dir / "v8_basket_affinity.parquet"
    cooccur_df.write_parquet(cooccur_out, compression="zstd")
    logger.info(f"   [OK] Guardado {cooccur_out.name} ({len(cooccur_df)} pares, {cooccur_out.stat().st_size / 1024:.1f} KB)")

    logger.info("-> [3/3] Generando v8_customer_history_28d.parquet...")
    # Historial personal de clientes activos en 28 días
    history_df = con.execute(f"""
        WITH tx AS (
            SELECT customer_idx,
                   article_id,
                   datediff('day', cast(t_dat as date), date '{ref_date}') as days_ago
            FROM read_parquet('{tx_str}')
            WHERE t_dat >= date '{ref_date}' - interval '{window_days} days'
              AND t_dat <= date '{ref_date}'
        ),
        recency AS (
            SELECT customer_idx,
                   article_id,
                   min(days_ago) as days_ago,
                   count(*) as freq
            FROM tx
            GROUP BY customer_idx, article_id
        ),
        ranked AS (
            SELECT customer_idx,
                   article_id,
                   cast(days_ago as smallint) as days_ago,
                   cast(row_number() OVER (
                       PARTITION BY customer_idx
                       ORDER BY days_ago ASC, freq DESC, article_id ASC
                   ) as smallint) as purchase_rank
            FROM recency
        )
        SELECT customer_idx, article_id, days_ago, purchase_rank
        FROM ranked
        WHERE purchase_rank <= 12
        ORDER BY customer_idx, purchase_rank
    """).pl()

    history_out = processed_dir / "v8_customer_history_28d.parquet"
    history_df.write_parquet(history_out, compression="zstd")
    logger.info(f"   [OK] Guardado {history_out.name} ({len(history_df)} registros, {history_out.stat().st_size / (1024*1024):.2f} MB)")

    total_sec = time.perf_counter() - start_t
    logger.info(f"[EXITO] Artefactos nativos de V8 exportados en {total_sec:.2f} s")

    return {
        "bestsellers_rows": len(bs_df),
        "cooccur_rows": len(cooccur_df),
        "history_rows": len(history_df),
        "total_seconds": total_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar artefactos de servicio V8")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Exportar a espacio de muestra data_processed/sample/",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    target_dir = DATA_PROCESSED_SAMPLE_DIR if args.sample else DATA_PROCESSED_DIR
    export_v8_artifacts(processed_dir=target_dir)


if __name__ == "__main__":
    main()
