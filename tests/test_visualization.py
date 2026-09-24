"""Tests unitarios y de integración para el módulo src.visualization."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR
from src.visualization import (
    COLOR_PRIMARY,
    PALETTE,
    plot_basket_cooccurrence_matrix,
    plot_customer_demographics_age,
    plot_feature_importance_from_df,
    plot_feature_importance_from_table,
    plot_leave_one_out,
    plot_lorenz_curve_gini,
    plot_microsegmentation_distribution,
    plot_nlp_metadata_signal,
    plot_pareto_frontier,
    plot_power_law_long_tail,
    plot_product_taxonomy_revenue,
    plot_repurchase_lag_distribution,
    plot_sales_channel_distribution,
    plot_sparsity_interaction_space,
    plot_temporal_window,
    plot_weekly_temporal_trend,
    set_editorial_style,
)


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Cierra figuras abiertas después de cada test para evitar consumo de memoria."""
    yield
    plt.close("all")


def test_set_editorial_style():
    """Verifica que el estilo editorial configure correctamente los parámetros de matplotlib."""
    set_editorial_style()
    assert plt.rcParams["figure.dpi"] == 300
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["axes.grid"] is True
    assert COLOR_PRIMARY in PALETTE.values()


def test_eda_plots_execution():
    """Verifica que todas las funciones de EDA generen figuras válidas."""
    set_editorial_style()

    # Si los parquets de data_processed existen, usamos una muestra pequeña para verificación rápida
    tx_file = DATA_PROCESSED_DIR / "transactions_5w.parquet"
    cust_file = DATA_PROCESSED_DIR / "customers.parquet"
    art_file = DATA_PROCESSED_DIR / "articles.parquet"
    agg_file = DATA_PROCESSED_DIR / "transactions_full_weekly_agg.parquet"

    if tx_file.exists() and cust_file.exists() and art_file.exists() and agg_file.exists():
        tx = pl.read_parquet(tx_file).head(300)
        cust = pl.read_parquet(cust_file).head(300)
        art = pl.read_parquet(art_file).head(300)
        agg = pl.read_parquet(agg_file).head(30)
    else:
        # Fallback sintético con tipos de datos y columnas exactas
        base_date = datetime.date(2020, 8, 1)
        tx = pl.DataFrame({
            "customer_idx": [i % 10 for i in range(100)],
            "article_id": [f"0{100000 + (i % 15)}" for i in range(100)],
            "t_dat": [base_date + datetime.timedelta(days=i % 30) for i in range(100)],
            "price": [0.01 + (i % 5) * 0.01 for i in range(100)],
            "sales_channel_id": [1 if i % 2 == 0 else 2 for i in range(100)],
        })
        cust = pl.DataFrame({
            "customer_idx": list(range(20)),
            "age": [20 + (i * 2) for i in range(20)],
            "club_member_status": ["ACTIVE", "ACTIVE", "LEFT CLUB", "PRE-CREATE"] * 5,
            "fashion_news_frequency": ["NONE", "Regularly", "Monthly", "NONE"] * 5,
        })
        art = pl.DataFrame({
            "article_id": [f"0{100000 + i}" for i in range(15)],
            "prod_name": [f"Garment Item {i}" for i in range(15)],
            "product_group_name": ["Garment Upper body", "Garment Lower body", "Accessories"] * 5,
            "index_group_name": ["Ladieswear", "Divided", "Menswear"] * 5,
            "detail_desc": [f"Detailed description of article {i} made of organic cotton." for i in range(15)],
        })
        agg = pl.DataFrame({
            "week_date": [base_date + datetime.timedelta(weeks=w) for w in range(10)],
            "tx_count": [1000 + w * 50 for w in range(10)],
            "active_customers": [500 + w * 20 for w in range(10)],
            "active_articles": [200 + w * 10 for w in range(10)],
            "revenue": [50000.0 + w * 1000.0 for w in range(10)],
        })

    # Sparsity
    fig1 = plot_sparsity_interaction_space(tx, cust, art)
    assert isinstance(fig1, plt.Figure)

    # Weekly trend
    fig2 = plot_weekly_temporal_trend(agg)
    assert isinstance(fig2, plt.Figure)

    # Repurchase lag
    fig3 = plot_repurchase_lag_distribution(tx)
    assert isinstance(fig3, plt.Figure)

    # Power law
    fig4 = plot_power_law_long_tail(tx)
    assert isinstance(fig4, plt.Figure)

    # Lorenz Gini
    fig5 = plot_lorenz_curve_gini(tx, art)
    assert isinstance(fig5, plt.Figure)

    # Demographics
    fig6 = plot_customer_demographics_age(cust)
    assert isinstance(fig6, plt.Figure)

    # Sales channel
    fig7 = plot_sales_channel_distribution(tx, cust)
    assert isinstance(fig7, plt.Figure)

    # Product taxonomy
    fig8 = plot_product_taxonomy_revenue(tx, art)
    assert isinstance(fig8, plt.Figure)

    # NLP metadata
    fig9 = plot_nlp_metadata_signal(art)
    assert isinstance(fig9, plt.Figure)

    # Basket co-occurrence
    fig11 = plot_basket_cooccurrence_matrix(tx, art)
    assert isinstance(fig11, plt.Figure)

    # Microsegmentation
    fig12 = plot_microsegmentation_distribution(tx, cust)
    assert isinstance(fig12, plt.Figure)


def test_ablation_plots_synthetic():
    """Verifica que las funciones de ablación grafiquen correctamente."""
    set_editorial_style()

    df_a1 = pl.DataFrame({
        "variant": ["3w", "5w", "8w", "10w"],
        "map12": [0.025, 0.024, 0.018, 0.016],
        "memory_mb": [2044.0, 2104.0, 2123.0, 2132.0],
        "n_candidates": [72000, 113000, 125000, 129000],
    })

    df_a2 = pl.DataFrame({
        "variant": [
            "Sin R1 (Repurchase)",
            "Sin R2 (Pop Global)",
            "Sin R3 (Pop Edad)",
            "Sin R4 (Pop Canal)",
            "Sin R5 (Item-CF)",
        ],
        "delta_pct": [-16.9, -20.0, -14.8, 4.6, -4.3],
        "delta_vs_baseline": [-0.005, -0.006, -0.004, 0.001, -0.0002],
    })

    df_a6 = pl.DataFrame({
        "variant": ["3w, LTR 1:5", "5w, LTR 1:5", "8w, LTR 1:5", "Ratio 1:20"],
        "memory_mb": [2044.0, 2104.0, 2123.0, 2500.0],
        "map12": [0.025, 0.0285, 0.018, 0.019],
        "duration_sec": [0.3, 0.5, 0.6, 1.2],
        "description": ["Pareto-Óptimo: True", "Pareto-Óptimo: True", "Dominada", "Dominada"],
    })

    fig1 = plot_temporal_window(df_a1)
    assert isinstance(fig1, plt.Figure)

    fig2 = plot_leave_one_out(df_a2)
    assert isinstance(fig2, plt.Figure)

    fig3 = plot_pareto_frontier(df_a6)
    assert isinstance(fig3, plt.Figure)


def test_feature_importance_plots(tmp_path):
    """Verifica las funciones de importancia de features tanto desde DF como desde CSV."""
    set_editorial_style()

    df_feat = pl.DataFrame({
        "feature": [f"uxa_feature_{i}" if i % 2 == 0 else f"item_stat_{i}" for i in range(20)],
        "importance": [1000 - i * 40 for i in range(20)],
        "gain_pct": [(1000 - i * 40) / 100.0 for i in range(20)],
    })

    # Test desde DataFrame
    fig1 = plot_feature_importance_from_df(df_feat, max_features=10)
    assert isinstance(fig1, plt.Figure)

    # Test desde CSV temporal
    csv_path = tmp_path / "test_feat_imp.csv"
    df_feat.write_csv(csv_path)
    fig2 = plot_feature_importance_from_table(csv_path, max_features=10)
    assert isinstance(fig2, plt.Figure)

    # Test con CSV oficial del repositorio si existe
    official_csv = TABLES_DIR / "feature_importance_lgbm.csv"
    if official_csv.exists():
        fig3 = plot_feature_importance_from_table(official_csv, max_features=15)
        assert isinstance(fig3, plt.Figure)


def test_figures_artifacts_integrity():
    """Verifica que las figuras clave existan en results/figures con resolución y tamaño adecuados."""
    expected_figures = [
        "fig_cap04_01_sparsity_interaction_space.png",
        "fig_cap04_02_weekly_temporal_trend.png",
        "fig_cap05_01_recall_ceiling_curve.png",
        "fig_cap07_01_feature_importance.png",
        "fig_cap09_01_ablation_temporal_window.png",
        "fig_cap09_02_ablation_leave_one_out.png",
        "fig_cap09_03_pareto_frontier_ram_map12.png",
        "fig_cap10_01_shap_summary_global.png",
    ]

    for fname in expected_figures:
        fpath = FIGURES_DIR / fname
        assert fpath.exists(), f"Figura {fname} no existe en {FIGURES_DIR}"
        assert fpath.stat().st_size > 10 * 1024, f"Figura {fname} es sospechosamente pequeña (<10KB)"

        with Image.open(fpath) as img:
            dpi = img.info.get("dpi", (0, 0))
            assert round(dpi[0]) >= 295, f"DPI insuficiente en {fname}: {dpi}"
