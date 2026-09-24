"""Generador de la Figura de Importancia de Características (Capítulo 7).

Genera results/figures/fig_07_feature_importance.png (y su alias fig_cap07_01)
a partir de la tabla oficial results/tables/feature_importance_lgbm.csv
o del modelo serializado models/lgbm_ranker.txt.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import FIGURES_DIR, MODELS_DIR, TABLES_DIR
from src.visualization.model_plots import (
    plot_feature_importance_from_model,
    plot_feature_importance_from_table,
)


def main() -> None:
    print("=" * 80)
    print("  GENERACIÓN DE FIGURA DE IMPORTANCIA DE CARACTERÍSTICAS (CAPÍTULO 7)")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIGURES_DIR / "fig_cap07_01_feature_importance.png"

    table_path = TABLES_DIR / "feature_importance_lgbm.csv"
    model_path = MODELS_DIR / "lgbm_ranker.txt"

    if table_path.exists():
        print(f"-> Generando gráfico desde tabla: {table_path.name}...")
        plot_feature_importance_from_table(table_path, save_path=out_fig)
    elif model_path.exists():
        print(f"-> Generando gráfico desde modelo: {model_path.name}...")
        plot_feature_importance_from_model(model_path, save_path=out_fig)
    else:
        print(f"[ERROR] No se encontró ni {table_path} ni {model_path}.", file=sys.stderr)
        sys.exit(1)

    print(f"[EXITO] Generado exitosamente: {out_fig.name}")


if __name__ == "__main__":
    main()
