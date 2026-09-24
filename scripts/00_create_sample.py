"""Script 00: Generación determinista del subconjunto local consistente en data_sample/.

Permite crear, después de descargar los CSV oficiales desde Kaggle, una muestra local
para ejecutar el pipeline de recomendación con el flag `--sample`.

Especificaciones técnicas:
1. Ingesta out-of-core con DuckDB sobre transactions_train.csv (últimas 5 semanas).
2. Muestreo estratificado de exactamente 2.000 clientes activos con al menos 2 compras (semilla 42).
3. Extracción relacional consistente (cero huérfanos entre transacciones, clientes y artículos).
4. Verificación de peso estricto (< 3.0 MB) para evaluación rápida local.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_RAW_DIR, DATA_SAMPLE_DIR, RANDOM_SEED, TEMPORAL_WINDOW_WEEKS
from src.ingestion.create_sample import extract_consistent_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generación determinista del mini-dataset representativo para data_sample/ (<3 MB)"
    )
    parser.add_argument(
        "--n-customers",
        type=int,
        default=2000,
        help="Número exacto de clientes activos a muestrear (default: 2000)",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=TEMPORAL_WINDOW_WEEKS,
        help="Ventana temporal de semanas recientes a considerar (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Semilla para muestreo pseudo-aleatorio determinista (default: 42)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=3.0,
        help="Límite superior admisible en disco para evaluación local (default: 3.0 MB)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  TFM H&M RECSYS: GENERADOR DETERMINISTA DE MUESTRA (data_sample/)")
    print(f"  Fuente: {DATA_RAW_DIR}")
    print(f"  Destino: {DATA_SAMPLE_DIR}")
    print(f"  Clientes: {args.n_customers} | Semanas: {args.weeks} | Semilla: {args.seed}")
    print("=" * 70)

    try:
        result = extract_consistent_sample(
            n_customers=args.n_customers,
            raw_dir=DATA_RAW_DIR,
            output_dir=DATA_SAMPLE_DIR,
            weeks_window=args.weeks,
            seed=args.seed,
            max_size_mb=args.max_size_mb,
        )

        print("\n" + "-" * 70)
        print("  RESUMEN DE EXTRACCIÓN RELACIONAL (CERO HUÉRFANOS)")
        print("-" * 70)
        print(
            f"  1. sample_transactions.csv : {result.n_transactions:>7,d} filas | {result.transactions_kb:>8.2f} KB"
        )
        print(
            f"  2. sample_customers.csv    : {result.n_customers:>7,d} filas | {result.customers_kb:>8.2f} KB"
        )
        print(
            f"  3. sample_articles.csv     : {result.n_articles:>7,d} filas | {result.articles_kb:>8.2f} KB"
        )
        print("-" * 70)
        print(
            f"  * Clientes únicos en TX    : {result.n_unique_customers_in_tx:>7,d} (coincidencia 100% con clientes)"
        )
        print(
            f"  * Artículos únicos en TX   : {result.n_unique_articles_in_tx:>7,d} (coincidencia 100% con artículos)"
        )
        print(
            f"  * Peso total en disco      : {result.total_size_mb:>7.2f} MB (Límite local: {args.max_size_mb} MB)"
        )
        print("-" * 70)
        print("[OK] Subconjunto representativo generado y validado con éxito.")
        print("=" * 70)

    except AssertionError as err:
        print(f"\n[ERROR DE INTEGRIDAD DEFENSIVA] {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[ERROR INESPERADO] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
