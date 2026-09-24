"""Script 01: Preprocesamiento e Ingesta Out-of-Core (DuckDB -> Polars -> Parquet).

Procesa eficientemente los CSVs de H&M convirtiéndolos en checkpoints Parquet
comprimidos con ZSTD y tipado downcasteado, sin superar los 12 GB de RAM.

Uso:
    python scripts/01_preprocess.py          # Procesa dataset completo (data/)
    python scripts/01_preprocess.py --sample # Procesa muestra rápida (data_sample/)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import get_processed_dir
from src.ingestion.preprocess import run_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocesamiento e ingesta out-of-core con DuckDB y Polars"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Utiliza la muestra ligera data_sample/ en lugar del dataset crudo masivo",
    )
    args = parser.parse_args()

    mode_label = (
        "MUESTRA RÁPIDA (data_sample/)" if args.sample else "DATASET COMPLETO MASIVO (data/)"
    )

    print("=" * 75)
    print("  FASE 1: PIPELINE PRINCIPAL DE INGESTA Y OPTIMIZACIÓN OUT-OF-CORE")
    print(f"  Modo Operacional: {mode_label}")
    print("=" * 75)

    try:
        summary = run_preprocess(use_sample=args.sample)

        print("\n" + "-" * 75)
        print("  REPORTE TÉCNICO DE INGESTA Y COMPRESIÓN COLUMNAR")
        print("-" * 75)
        print(f"  * Transacciones de 5 semanas   : {summary.transactions_rows:>10,d} filas")
        print(f"  * Clientes únicos mapeados     : {summary.customers_rows:>10,d} usuarios")
        print(f"  * Artículos en catálogo        : {summary.articles_rows:>10,d} productos")
        print(f"  * Cobertura temporal activa    : {summary.min_date} a {summary.max_date}")
        print(f"  * Consumo final de RAM (RSS)   : {summary.memory_rss_mb:>10.1f} MB")
        print(f"  * Tamaño CSV original fuente   : {summary.raw_size_mb:>10.2f} MB")
        print(f"  * Tamaño combinado Parquets    : {summary.parquet_size_mb:>10.2f} MB")
        print(f"  * Reducción neta de espacio    : {summary.compression_ratio_pct:>9.2f}%")
        print("-" * 75)
        target_dir = get_processed_dir(args.sample)
        print(f"[OK] Checkpoints Parquet generados exitosamente en {target_dir}/.")
        print("=" * 75)

    except AssertionError as err:
        print(f"\n[ERROR DEFENSIVO] {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[ERROR CRÍTICO] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
