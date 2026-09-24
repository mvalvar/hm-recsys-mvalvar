"""Generación de las 12 figuras del análisis exploratorio (EDA) del TFM.

Exporta los paneles visuales para el análisis descriptivo y sincroniza
la nomenclatura dual para los capítulos de la memoria (fig_cap04_XX).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR
from src.visualization.eda_plots import (
    plot_basket_cooccurrence_matrix,
    plot_customer_demographics_age,
    plot_lorenz_curve_gini,
    plot_microsegmentation_distribution,
    plot_nlp_metadata_signal,
    plot_power_law_long_tail,
    plot_product_taxonomy_revenue,
    plot_repurchase_lag_distribution,
    plot_sales_channel_distribution,
    plot_sparsity_interaction_space,
    plot_weekly_temporal_trend,
)


def main() -> None:
    print("=" * 80)
    print("  GENERANDO FIGURAS DE ANÁLISIS EXPLORATORIO (EDA)")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("-> Cargando datasets procesados desde data_processed/...")
    tx = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    cust = pl.read_parquet(DATA_PROCESSED_DIR / "customers.parquet")
    art = pl.read_parquet(DATA_PROCESSED_DIR / "articles.parquet")
    agg = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_full_weekly_agg.parquet")

    audit_metrics_path = TABLES_DIR / "eda_raw_audit_metrics.json"
    audit_metrics = {}
    if audit_metrics_path.exists():
        with open(audit_metrics_path, encoding="utf-8") as f:
            audit_metrics = json.load(f)

    print("-> Generando Figura 01: Sparsity y Espacio de Interacción...")
    plot_sparsity_interaction_space(
        tx, cust, art, audit_metrics, save_path=FIGURES_DIR / "fig_cap04_01_sparsity_interaction_space.png"
    )

    print("-> Generando Figura 02: Serie Temporal Histórica...")
    plot_weekly_temporal_trend(agg, save_path=FIGURES_DIR / "fig_cap04_02_weekly_temporal_trend.png")

    print("-> Generando Figura 03: Ciclo de Vida y Repetición de Compra...")
    plot_repurchase_lag_distribution(
        tx, save_path=FIGURES_DIR / "fig_cap04_03_repurchase_lag_distribution.png"
    )

    print("-> Generando Figura 04: Ley de Potencias y Long Tail...")
    plot_power_law_long_tail(tx, save_path=FIGURES_DIR / "fig_cap04_04_power_law_long_tail.png")

    print("-> Generando Figura 05: Curva de Lorenz e Índice de Gini Dual...")
    plot_lorenz_curve_gini(tx, art, save_path=FIGURES_DIR / "fig_cap04_05_lorenz_curve_gini.png")

    print("-> Generando Figura 06: Demografía y Cohortes de Edad...")
    plot_customer_demographics_age(
        cust, save_path=FIGURES_DIR / "fig_cap04_06_customer_demographics_age.png"
    )

    print("-> Generando Figura 07: Análisis de Canal de Venta...")
    plot_sales_channel_distribution(
        tx, cust, save_path=FIGURES_DIR / "fig_cap04_07_sales_channel_distribution.png"
    )

    print("-> Generando Figura 08: Taxonomía de Producto y Departamentos...")
    plot_product_taxonomy_revenue(
        tx, art, save_path=FIGURES_DIR / "fig_cap04_08_product_taxonomy_revenue.png"
    )

    print("-> Generando Figura 09: Análisis de Señal NLP en Metadatos...")
    plot_nlp_metadata_signal(art, save_path=FIGURES_DIR / "fig_cap04_09_nlp_metadata_signal.png")

    print("-> Generando Figura 11: Matriz de Co-ocurrencia en Cesta Transaccional...")
    plot_basket_cooccurrence_matrix(
        tx, art, save_path=FIGURES_DIR / "fig_cap04_11_basket_cooccurrence_matrix.png"
    )

    print("-> Generando Figura 12: Micro-Segmentación Sociodemográfica y Dispersión de Precios...")
    plot_microsegmentation_distribution(
        tx, cust, save_path=FIGURES_DIR / "fig_cap04_12_microsegmentation_distribution.png"
    )

    print(
        f"[EXITO] Las 11 figuras del EDA fueron generadas y exportadas a {FIGURES_DIR}."
    )


if __name__ == "__main__":
    main()
