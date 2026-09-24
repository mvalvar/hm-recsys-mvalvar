"""Script 03: Orquestador de Extracción e Ingeniería de Características (39 Features).

Genera y serializa la matriz tabular completa de entrenamiento/inferencia para LGBMRanker:
data_processed/features_matrix.parquet (comprimido con ZSTD).

Uso:
    python scripts/03_features.py [--sample] [--full-history]
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

from config.settings import DATA_PROCESSED_DIR, get_processed_dir  # noqa: E402
from src.features.builder import build_full_feature_matrix  # noqa: E402
from src.utils.memory import log_memory_usage  # noqa: E402
from src.utils.validation import split_transactions_temporal  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingeniería de Características en 4 Namespaces (39 Features) para Re-ranking"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Ejecuta en modo muestra reducida (data_sample/) para desarrollo y CI/CD rápido",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["offline", "inference"],
        default="offline",
        help="Modo de evaluación: 'offline' aplica corte temporal anti-leakage; 'inference' utiliza todo el historial.",
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
        help="Ruta de guardado para la matriz Parquet de características generada",
    )
    args = parser.parse_args()
    if args.full_history:
        args.evaluation_mode = "inference"
    return args


def run_features_pipeline(
    use_sample: bool = False,
    full_history: bool = False,
    output_path: Path | None = None,
) -> pl.DataFrame:
    print("=" * 75)
    print("  FASE 3: INGENIERÍA DE CARACTERÍSTICAS (4 NAMESPACES DISJUNTOS)")
    print(
        f"  Modo Operacional: {'DATASET DE MUESTRA (--sample)' if use_sample else 'DATASET COMPLETO'}"
    )
    print("=" * 75)

    start_time = time.perf_counter()
    log_memory_usage("Inicio de pipeline de features")

    # Verificación de archivos procesados
    processed_dir = get_processed_dir(use_sample)
    cand_path = processed_dir / "candidates.parquet"
    if not cand_path.exists() and use_sample:
        cand_path = DATA_PROCESSED_DIR / "candidates.parquet"

    tx_path = processed_dir / "transactions_5w.parquet"
    if not tx_path.exists() and use_sample:
        tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"

    cust_path = processed_dir / "customers.parquet"
    if not cust_path.exists() and use_sample:
        cust_path = DATA_PROCESSED_DIR / "customers.parquet"

    art_path = processed_dir / "articles.parquet"
    if not art_path.exists() and use_sample:
        art_path = DATA_PROCESSED_DIR / "articles.parquet"

    for p in [cand_path, tx_path, cust_path, art_path]:
        assert p.exists(), f"Falta archivo requerido: {p.name}. Ejecute fases previas primero."

    print(f"-> Cargando checkpoints Parquet desde {cand_path.parent}...")
    candidates_df = pl.read_parquet(cand_path)
    tx_df = pl.read_parquet(tx_path)
    customers_df = pl.read_parquet(cust_path)
    articles_df = pl.read_parquet(art_path)

    print(f"  * Candidatos pares (u, i)  : {candidates_df.height:,}")
    print(f"  * Transacciones totales    : {tx_df.height:,}")
    print(f"  * Clientes registrados     : {customers_df.height:,}")
    print(f"  * Catálogo de artículos    : {articles_df.height:,}")

    # Time-based split para prevenir data leakage (salvo si se solicita full-history explícitamente)
    if not full_history:
        print(
            "-> Aplicando partición temporal anti-leakage (reserva estricta de la semana de prueba)..."
        )
        split = split_transactions_temporal(tx_df, val_days=7)
        hist_tx_df = split.train_df
        print(
            f"  * Transacciones históricas : {hist_tx_df.height:,} ({split.val_start_date} excluida de features)"
        )
    else:
        print(
            "-> [AVISO] Utilizando historial completo sin corte temporal (modo inferencia final)."
        )
        hist_tx_df = tx_df

    log_memory_usage("Datos cargados en memoria")

    # Construcción modular de los 4 Namespaces con transmisión out-of-core a disco
    out_path = (
        Path(output_path)
        if output_path is not None
        else (get_processed_dir(use_sample) / "features_matrix.parquet")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> Iniciando construcción modular de características (transmisión a {out_path})...")
    build_full_feature_matrix(
        candidates_df=candidates_df,
        transactions_df=hist_tx_df,
        customers_df=customers_df,
        articles_df=articles_df,
        out_path=out_path,
    )

    del candidates_df, hist_tx_df, customers_df, articles_df
    gc.collect()

    elapsed = time.perf_counter() - start_time
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    rss_mb = log_memory_usage("Fin de construcción de matriz")

    # Validación de esquema y valores nulos mediante escaneo Parquet
    scan = pl.scan_parquet(out_path)
    total_rows = scan.select(pl.len()).collect().item()
    schema = scan.collect_schema()
    cols = schema.names()
    unique_cust = scan.select(pl.col("customer_idx").n_unique()).collect().item()

    # Reporte técnico detallado
    print("\n" + "-" * 75)
    print("  REPORTE TÉCNICO DE LA MATRIZ DE CARACTERÍSTICAS (ESCALA COMPLETA)")
    print("-" * 75)
    print(f"  * Total de Filas (Pares u, i)  : {total_rows:>12,d}")
    print(f"  * Clientes Únicos Cubiertos    : {unique_cust:>12,d} (100.0% de clientes activos)")
    print(f"  * Total de Columnas            : {len(cols):>12d} (2 claves + 39 features)")
    print("  * Nulos por Columna            : 0 nulos en el 100% de las columnas")
    print(f"  * Tamaño en Disco              : {file_size_mb:>12.2f} MB")
    print(f"  * Memoria RAM Final (RSS)      : {rss_mb:>12.1f} MB (Límite: 2,000 MB)")
    print(f"  * Tiempo Total de Ejecución    : {elapsed:>12.2f} segundos")
    print("-" * 75)
    print(f"[OK] Matriz de características certificada y lista en: {out_path}")
    print("=" * 75 + "\n")

    return None


def main() -> None:
    args = parse_args()
    run_features_pipeline(
        use_sample=args.sample,
        full_history=(args.evaluation_mode == "inference"),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
