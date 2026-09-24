"""Script 02: Generación y Consolidación de Candidatos (8 Heurísticas de Recall).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

- Escalado masivo para los 278.275 clientes activos de transactions_5w.parquet.
- 8 heurísticas complementarias: R1 Recompra, R2 Popularidad Global, R3 Popularidad por Edad,
  R4 Popularidad por Canal, R5 Item-CF, R6 Familias de Producto, R7 Trending,
  R8 Popularidad por Departamento Favorito.
- Consolidación deduplicada en memoria acotada (< 1.8 GB RSS) mediante particionamiento
  por rangos de customer_idx y recolección de basura con gc.collect().
- Salida serializada en Parquet ZSTD: data_processed/candidates.parquet.

Uso:
    python scripts/02_candidates.py          # Procesa los 278.275 clientes activos completos
    python scripts/02_candidates.py --sample # Modo muestra reducida (2.000 clientes de data_sample)
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATA_SAMPLE_DIR,
    N_CANDIDATES_PER_USER,
    get_processed_dir,
)
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
from src.utils.memory import log_memory_usage  # noqa: E402
from src.utils.validation import split_transactions_temporal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generación y consolidación out-of-core de candidatos multi-heurística"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Ejecuta en modo muestra reducida (2.000 clientes activos)",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["offline", "inference"],
        default="offline",
        help="Modo de evaluación: 'offline' aplica corte temporal estricto aislando la semana de prueba; 'inference' utiliza todo el historial.",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Alias de conveniencia para --evaluation-mode inference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta de guardado para el archivo Parquet de candidatos generados",
    )
    args = parser.parse_args()
    if args.full_history:
        args.evaluation_mode = "inference"

    mode_str = (
        "MUESTRA RÁPIDA (2.000 CLIENTES)"
        if args.sample
        else "CATÁLOGO COMPLETO ACTIVO (278.275 CLIENTES)"
    )
    eval_str = (
        "OFFLINE (ANTI-LEAKAGE)"
        if args.evaluation_mode == "offline"
        else "INFERENCIA (TODO EL HISTORIAL)"
    )
    print("=" * 80)
    print("  FASE 2: GENERACIÓN DE CANDIDATOS OUT-OF-CORE (8 HEURÍSTICAS)")
    print(f"  Modo Operacional: {mode_str} | Evaluación: {eval_str}")
    print("=" * 80)

    t0 = time.perf_counter()
    log_memory_usage("Inicio de generación de candidatos")

    processed_dir = get_processed_dir(args.sample)
    tx_path = processed_dir / "transactions_5w.parquet"
    if not tx_path.exists() and args.sample:
        processed_dir = DATA_PROCESSED_DIR
        tx_path = processed_dir / "transactions_5w.parquet"

    cust_path = processed_dir / "customers.parquet"
    art_path = processed_dir / "articles.parquet"

    for p in [tx_path, cust_path, art_path]:
        assert p.exists(), f"Debe ejecutar scripts/01_preprocess.py primero. No existe {p}"

    print(f"-> Cargando checkpoints Parquet desde {processed_dir}...")
    tx_df = pl.read_parquet(tx_path)
    cust_df = pl.read_parquet(cust_path)
    art_df = pl.read_parquet(art_path)

    if args.sample:
        # Subconjunto estratificado para verificación rápida de pipeline
        sample_cust_path = DATA_SAMPLE_DIR / "sample_customers.csv"
        if sample_cust_path.exists():
            sample_ids = pl.read_csv(sample_cust_path, columns=["customer_id"])[
                "customer_id"
            ].to_list()
            mapping_path = processed_dir / "customer_id_mapping.parquet"
            if not mapping_path.exists() and args.sample:
                mapping_path = DATA_PROCESSED_DIR / "customer_id_mapping.parquet"
            if mapping_path.exists():
                mapping_df = pl.read_parquet(mapping_path).filter(
                    pl.col("customer_id").is_in(sample_ids)
                )
                valid_indices = set(mapping_df["customer_idx"].to_list())
                cust_df = cust_df.filter(pl.col("customer_idx").is_in(valid_indices))
                tx_df = tx_df.filter(pl.col("customer_idx").is_in(valid_indices))
        else:
            cust_df = cust_df.head(2000)
            valid_indices = set(cust_df["customer_idx"].to_list())
            tx_df = tx_df.filter(pl.col("customer_idx").is_in(valid_indices))
        print(
            f"  * Muestra acotada a {cust_df.height:,} clientes y {tx_df.height:,} transacciones."
        )

    active_customers_count = cust_df["customer_idx"].n_unique()
    print(f"  * Clientes activos a procesar : {active_customers_count:,}")
    print(f"  * Transacciones de 5 semanas  : {tx_df.height:,}")
    print(f"  * Catálogo de artículos       : {art_df.height:,}")

    # Time-based split: semana W104 aislada como holdout set para prevenir data leakage (look-ahead bias)
    if args.evaluation_mode == "offline":
        print(
            "-> Aplicando partición temporal anti-leakage (reserva estricta de la semana de prueba)..."
        )
        split = split_transactions_temporal(tx_df, val_days=7)
        cand_tx_df = split.train_df
        print(
            f"  * Transacciones para candidatos : {cand_tx_df.height:,} ({split.val_start_date} excluida)"
        )
    else:
        print(
            "-> [MODO INFERENCIA] Utilizando historial completo sin corte temporal para predicción final."
        )
        cand_tx_df = tx_df

    log_memory_usage("Datos cargados en memoria")

    # R1: Recompra histórica reciente
    t_start = time.perf_counter()
    print("\n-> Generando R1: Recompra histórica reciente...")
    r1 = generate_repurchase(cand_tx_df)
    print(f"  * R1 completado: {r1.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # R2: Popularidad global con decaimiento temporal
    t_start = time.perf_counter()
    print("-> Generando R2: Popularidad global con decaimiento exponencial...")
    r2 = generate_global_popularity(cand_tx_df)
    print(f"  * R2 completado: {r2.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # R3: Popularidad segmentada por edad
    t_start = time.perf_counter()
    print("-> Generando R3: Popularidad por cohorte de edad (age_bin)...")
    r3 = generate_age_group_popularity(cand_tx_df, cust_df)
    print(f"  * R3 completado: {r3.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # R4: Popularidad discriminada por canal dominante
    t_start = time.perf_counter()
    print("-> Generando R4: Popularidad por canal dominante de venta...")
    r4 = generate_channel_popularity(cand_tx_df)
    print(f"  * R4 completado: {r4.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # R5: Filtrado colaborativo ítem-ítem (Item-CF)
    t_start = time.perf_counter()
    print("-> Generando R5: Filtrado colaborativo ítem-ítem (co-compras)...")
    r5 = generate_item_cf(cand_tx_df)
    print(f"  * R5 completado: {r5.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # R6: Afinidad por familias de producto y departamentos
    t_start = time.perf_counter()
    print("-> Generando R6: Afinidad por familias de producto...")
    r6 = generate_product_family(cand_tx_df, art_df)
    print(f"  * R6 completado: {r6.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # R7: Artículos en tendencia y aceleración de ventas
    t_start = time.perf_counter()
    print("-> Generando R7: Artículos en tendencia y aceleración semanal...")
    r7 = generate_trending_items(cand_tx_df)
    print(f"  * R7 completado: {r7.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # R8: Popularidad estacional por departamento favorito
    t_start = time.perf_counter()
    print("-> Generando R8: Popularidad estacional por departamento favorito...")
    r8 = generate_user_dept_popularity(cand_tx_df, art_df)
    print(f"  * R8 completado: {r8.height:,} pares en {time.perf_counter() - t_start:.2f} s")

    # Liberar dataframes base para optimizar memoria antes de la consolidación
    del tx_df, cand_tx_df, cust_df, art_df
    gc.collect()
    log_memory_usage("Heurísticas generadas (pre-consolidación)")

    # Consolidación deduplicada y meta-features con control de memoria
    print(
        f"\n-> Consolidando candidatos (Top-{N_CANDIDATES_PER_USER} por cliente) con particionamiento..."
    )
    t_cons = time.perf_counter()
    candidates = consolidate_candidates(
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        r7,
        r8,
        max_per_user=N_CANDIDATES_PER_USER,
        n_partitions=4,
    )
    del r1, r2, r3, r4, r5, r6, r7, r8
    gc.collect()
    print(f"  * Consolidación finalizada en {time.perf_counter() - t_cons:.2f} s")

    # Validación de cobertura
    unique_users = candidates["customer_idx"].n_unique()
    assert unique_users == active_customers_count, (
        f"Cobertura de clientes incompleta: {unique_users:,} != {active_customers_count:,}"
    )

    # Serialización en Parquet ZSTD
    out_path = (
        Path(args.output)
        if args.output is not None
        else (get_processed_dir(args.sample) / "candidates.parquet")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> Escribiendo candidatos en {out_path} (compresión ZSTD)...")
    candidates.write_parquet(out_path, compression="zstd")
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    total_duration = time.perf_counter() - t0
    final_rss = log_memory_usage("Fin de generación de candidatos")

    # Reporte ejecutivo
    print("\n" + "=" * 80)
    print("  REPORTE EJECUTIVO DE CANDIDATOS (FASE 2 ESCALADA)")
    print("=" * 80)
    print(f"  * Total de Pares Candidatos    : {candidates.height:>12,d}")
    print(f"  * Clientes Únicos Cubiertos    : {unique_users:>12,d} (100.0% de clientes activos)")
    print(f"  * Promedio Candidatos / Cliente: {candidates.height / unique_users:>12.1f}")
    print(f"  * Columnas del Esquema         : {len(candidates.columns):>12d}")
    print(f"  * Archivo Generado             : {out_path.name} ({file_size_mb:.2f} MB)")
    print(f"  * Consumo Máximo de RAM (RSS)  : {final_rss:>12.1f} MB (Límite: 2,000 MB)")
    print(f"  * Tiempo Total de Ejecución    : {total_duration:>12.2f} segundos")
    print("=" * 80)
    print(f"[EXITO] Candidatos a escala completa generados y certificados en: {out_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
