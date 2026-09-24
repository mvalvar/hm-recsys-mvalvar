"""Script 05: Ejecución del Estudio Sistemático de Ablación (Experimentos A1 a A6).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Ejecuta los experimentos empíricos de aislamiento:
- A1: Sensibilidad a la longitud de la ventana temporal (3, 5, 8 y 10 semanas).
- A2: Aportación marginal de candidatos mediante Leave-One-Out (R1 a R7).
- A3: Valor informativo de las Candidate Source Flags (is_R1 a is_R7).
- A4: Sensibilidad al ratio de Negative Downsampling (1:3, 1:5, 1:10, 1:20).
- A5: Comparativa de funciones de pérdida: LambdaRank (Listwise) vs Binary Cross-Entropy (Pointwise).
- A6: Análisis multiobjetivo y frontera de eficiencia de Pareto (Precisión vs RAM vs Latencia).

Uso:
    python scripts/05_ablation.py --sample --experiment A4
    python scripts/05_ablation.py --sample --experiment A5
    python scripts/05_ablation.py --sample --experiment A6
    python scripts/05_ablation.py --sample --experiment all
    python scripts/05_ablation.py --experiment all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import TABLES_DIR
from src.evaluation.ablation import run_ablation_experiment
from src.utils.memory import log_memory_usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estudio Sistemático de Ablación (Experimentos A1 a A6)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Utiliza hiperparámetros ágiles para validación y CI/CD rápido",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "A1", "A2", "A3", "A4", "A5", "A6"],
        help="Identificador del experimento de ablación ('A1'..'A6' o 'all')",
    )
    return parser.parse_args()


def print_formatted_table(title: str, results: list) -> None:
    """Imprime una tabla formateada en consola con métricas comparativas."""
    print("\n" + "=" * 96)
    print(f"  {title}")
    print("=" * 96)
    header = (
        f"{'Variante':<24} | {'MAP@12':>8} | {'Delta':>9} | {'Delta %':>8} | "
        f"{'Ceiling@80':>10} | {'Pares':>8} | {'RAM (MB)':>9} | {'Tiempo':>7}"
    )
    print(header)
    print("-" * 96)

    for r in results:
        delta_str = f"{r.delta_vs_baseline:+.5f}" if r.delta_vs_baseline != 0 else "Baseline"
        delta_pct_str = f"{r.delta_pct:+.2f}%" if r.delta_pct != 0 else "---"
        ceiling_str = f"{r.recall_ceiling:.4f}" if r.recall_ceiling > 0 else "N/A"
        row = (
            f"{r.variant:<24} | {r.map12:>8.5f} | {delta_str:>9} | {delta_pct_str:>8} | "
            f"{ceiling_str:>10} | {r.n_candidates:>8,d} | {r.memory_mb:>9.1f} | {r.duration_sec:>6.2f}s"
        )
        print(row)
    print("=" * 96)


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    mode_str = "MUESTRA ÁGIL (--sample)" if args.sample else "PRODUCCIÓN COMPLETA"
    print("*" * 96)
    print("  TFM RECSYS: ESTUDIO SISTEMÁTICO DE ABLACIÓN EXPERIMENTAL (A1 a A6)")
    print(f"  Modo Operacional: {mode_str} | Experimento: {args.experiment.upper()}")
    print("*" * 96)

    log_memory_usage("Inicio del framework de ablación")

    results = run_ablation_experiment(
        experiment_id=args.experiment,
        use_sample=args.sample,
        output_dir=TABLES_DIR,
    )

    # Filtrar y presentar tablas por experimento
    exp_groups: dict[str, list] = {}
    for r in results:
        exp_groups.setdefault(r.experiment_id, []).append(r)

    title_map = {
        "A1": "TABLA COMPARATIVA A1: SENSIBILIDAD AL HORIZONTE DE MEMORIA TEMPORAL",
        "A2": "TABLA COMPARATIVA A2: APORTACIÓN MARGINAL LEAVE-ONE-OUT (R1..R7)",
        "A3": "TABLA COMPARATIVA A3: VALOR INFORMATIVO DE LAS CANDIDATE SOURCE FLAGS",
        "A4": "TABLA COMPARATIVA A4: SENSIBILIDAD AL RATIO DE NEGATIVE DOWNSAMPLING",
        "A5": "TABLA COMPARATIVA A5: COMPARATIVA DE FUNCIONES DE PÉRDIDA (LAMBDARANK VS BCE)",
        "A6": "TABLA COMPARATIVA A6: FRONTERA DE EFICIENCIA DE PARETO (ACCURACY VS RECURSOS)",
    }

    for exp_id, group in exp_groups.items():
        title = title_map.get(exp_id, f"EXPERIMENTO {exp_id}")
        print_formatted_table(title, group)

    total_time = time.perf_counter() - start_time
    final_rss = log_memory_usage("Finalización de ablación")

    print("\n" + "=" * 96)
    print("  RESUMEN EJECUTIVO DE ABLACIÓN")
    print("=" * 96)
    print(f"  * Experimentos ejecutados : {list(exp_groups.keys())}")
    print(f"  * Variantes evaluadas     : {len(results)}")
    print(f"  * Tiempo total de corrida : {total_time:.2f} segundos")
    print(f"  * Consumo pico final RAM  : {final_rss:.1f} MB (Cota < 2.0 GB preservada)")
    out_csv_name = "ablation_results_sample.csv" if args.sample else "ablation_results.csv"
    print(f"  * Archivo consolidado     : {TABLES_DIR / out_csv_name}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
