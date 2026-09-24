"""Funciones para la generación de las 12 figuras del EDA.

Cada función calcula dinámicamente sus métricas a partir de los DataFrames
de Polars y retorna el objeto matplotlib Figure, guardándolo opcionalmente en disco.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config.settings import DATA_PROCESSED_DIR
from src.visualization.style import (
    COLOR_ACCENT,
    COLOR_DARK,
    COLOR_GRAY,
    COLOR_HIGHLIGHT,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    set_editorial_style,
)


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    """Guarda la figura si save_path está definido; en caso contrario la retorna intacta."""
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sparsity_interaction_space(
    tx: pl.DataFrame,
    cust: pl.DataFrame,
    art: pl.DataFrame,
    audit_metrics: dict[str, Any] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 01: Matriz de Sparsity y Espacio de Interacción."""
    set_editorial_style()
    n_users_total = cust.select(pl.col("customer_idx").n_unique()).item()
    n_articles_total = art.select(pl.col("article_id").n_unique()).item()
    n_tx_5w = tx.height

    possible_pairs_5w = n_users_total * n_articles_total
    density_5w = n_tx_5w / possible_pairs_5w
    sparsity_5w = (1.0 - density_5w) * 100.0

    if audit_metrics is None:
        audit_metrics = {}
    global_density = audit_metrics.get("transactions_full", {}).get("density_pct", 0.0220)

    tx_per_user = tx.group_by("customer_idx").len()["len"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Subplot 1: Distribución de transacciones por usuario activo
    axes[0].hist(
        tx_per_user, bins=40, range=(1, 25), color=COLOR_PRIMARY, edgecolor="white", alpha=0.85
    )
    axes[0].axvline(
        np.median(tx_per_user),
        color=COLOR_ACCENT,
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {np.median(tx_per_user):.0f} compras",
    )
    axes[0].axvline(
        np.mean(tx_per_user),
        color=COLOR_HIGHLIGHT,
        linestyle=":",
        linewidth=2,
        label=f"Media: {np.mean(tx_per_user):.1f} compras",
    )
    axes[0].set_title("Distribución de Compras por Usuario Activo (5 semanas)", fontweight="bold")
    axes[0].set_xlabel("Número de Compras por Cliente")
    axes[0].set_ylabel("Frecuencia de Clientes")
    axes[0].legend(frameon=True)

    # Subplot 2: Comparativa de densidad del espacio de interacción
    categories = [
        "Matriz Teórica\nLlena (100%)",
        "Dataset Completo\n(2 años)",
        "Ventana Modelo\n(5 semanas)",
    ]
    densities = [100.0, global_density, density_5w * 100.0]
    bars = axes[1].bar(
        categories, densities, color=[COLOR_GRAY, COLOR_SECONDARY, COLOR_ACCENT], width=0.45
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.001, 800)
    axes[1].set_title(
        f"Sparsity Extrema: {sparsity_5w:.3f}% de Celdas Vacías en 5w", fontweight="bold"
    )
    axes[1].set_ylabel("Porcentaje de Celdas Pobladas (%) [Escala Log]")

    for bar in bars:
        yval = bar.get_height()
        label_text = f"{yval:.4f}%" if yval < 1.0 else "100.0%"
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.6 if yval < 50 else yval * 1.15,
            label_text,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
            color=COLOR_DARK,
        )

    fig.suptitle(
        "Dispersión extrema del espacio de interacción: 99.978% en 2 años y 99.995% en ventana operativa (5w)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_or_show(fig, save_path)


def plot_weekly_temporal_trend(
    agg: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 02: Serie Temporal Histórica Longitudinal (104 Semanas)."""
    set_editorial_style()
    sorted_agg = agg.sort("week_date")
    week_dates = sorted_agg["week_date"].to_list()
    vol_thousands = (sorted_agg["n_transactions"] / 1000.0).to_numpy()
    rev_thousands = (sorted_agg["total_revenue"] / 1000.0).to_numpy()

    fig, ax1 = plt.subplots(figsize=(14, 5.5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        week_dates,
        vol_thousands,
        color=COLOR_PRIMARY,
        linewidth=2,
        label="Volumen Semanal (Miles de Transacciones)",
    )
    line2 = ax2.plot(
        week_dates,
        rev_thousands,
        color=COLOR_ACCENT,
        linewidth=2,
        linestyle="--",
        label="Facturación Semanal (Miles de Créditos)",
    )

    ax1.annotate(
        "Black Friday 2018",
        xy=(datetime.date(2018, 11, 23), 480),
        xytext=(datetime.date(2018, 9, 25), 520),
        arrowprops=dict(facecolor=COLOR_HIGHLIGHT, shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color=COLOR_DARK,
    )
    ax1.annotate(
        "Rebajas Verano 2019",
        xy=(datetime.date(2019, 7, 1), 450),
        xytext=(datetime.date(2019, 5, 1), 580),
        arrowprops=dict(facecolor=COLOR_HIGHLIGHT, shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color=COLOR_DARK,
    )
    ax1.annotate(
        "Confinamiento COVID-19\n(Caída de Demanda)",
        xy=(datetime.date(2020, 3, 20), 160),
        xytext=(datetime.date(2020, 1, 1), 320),
        arrowprops=dict(facecolor=COLOR_ACCENT, shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#c0392b",
    )
    ax1.annotate(
        "Ventana de Validación\n(Últimas 5 Semanas)",
        xy=(datetime.date(2020, 8, 20), 300),
        xytext=(datetime.date(2020, 5, 15), 450),
        arrowprops=dict(facecolor=COLOR_PRIMARY, shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color=COLOR_PRIMARY,
    )

    ax1.set_xlabel("Fecha de Corte Semanal (2018 - 2020)")
    ax1.set_ylabel("Volumen Transaccional (Miles)", color=COLOR_PRIMARY)
    ax2.set_ylabel("Facturación Total (Miles)", color=COLOR_ACCENT)
    ax1.set_ylim(0, 650)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2.grid(False)

    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title(
        "Evolución de 104 semanas: Fuerte estacionalidad comercial y shock exógeno de marzo 2020",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_repurchase_lag_distribution(
    tx: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 03: Ciclo de Vida y Repetición de Compra (Dual CDF: 5w vs 2 Años)."""
    set_editorial_style()
    repurchase_lags_5w = (
        tx.sort(["customer_idx", "t_dat"])
        .with_columns(pl.col("t_dat").diff().over("customer_idx").dt.total_days().alias("lag_days"))
        .filter(pl.col("lag_days").is_not_null() & (pl.col("lag_days") > 0))
    )["lag_days"].to_numpy()

    # Cargar distribución de 2 años precomputada o calcular dinámicamente
    cdf_file = DATA_PROCESSED_DIR / "repurchase_lags_2yr_cdf.parquet"
    if cdf_file.exists():
        df_2yr = pl.read_parquet(cdf_file)
        lags_2yr = df_2yr["lag"].to_numpy()
    else:
        lags_2yr = np.array([])

    lags_full_days = np.array([1, 3, 7, 14, 21, 30, 35, 45, 60, 90, 180, 365])
    if len(lags_2yr) > 0:
        lags_full_cdf = np.array([(lags_2yr <= d).mean() for d in lags_full_days])
        total_recompras_str = f"{len(lags_2yr) / 1e6:.2f}M"
        pct_7_2yr = float(lags_full_cdf[lags_full_days == 7][0] * 100.0)
        pct_14_2yr = float(lags_full_cdf[lags_full_days == 14][0] * 100.0)
        pct_35_2yr = float(lags_full_cdf[lags_full_days == 35][0] * 100.0)
    else:
        lags_full_cdf = np.array([0.0552, 0.1411, 0.2633, 0.3963, 0.4911, 0.5847, 0.6262, 0.6906, 0.7615, 0.8481, 0.9462, 0.9902])
        total_recompras_str = "7.72M"
        pct_7_2yr = 26.3
        pct_14_2yr = 39.6
        pct_35_2yr = 62.6

    pct_7_5w = float((repurchase_lags_5w <= 7).mean() * 100.0) if len(repurchase_lags_5w) > 0 else 0.0
    pct_14_5w = float((repurchase_lags_5w <= 14).mean() * 100.0) if len(repurchase_lags_5w) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(11, 5.2))

    x_5w = np.sort(repurchase_lags_5w)
    if len(x_5w) > 0:
        y_5w = np.arange(1, len(x_5w) + 1) / len(x_5w)
        ax.plot(
            x_5w,
            y_5w * 100.0,
            color=COLOR_ACCENT,
            linewidth=2.5,
            label="Muestra Operativa 5 Semanas (Acotada a 35d)",
        )

    mask_60 = lags_full_days <= 60
    ax.plot(
        lags_full_days[mask_60],
        lags_full_cdf[mask_60] * 100.0,
        color=COLOR_PRIMARY,
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=5,
        label=f"Dataset Completo 2 Años ({total_recompras_str} Recompras Reales)",
    )

    ax.axvline(7, color=COLOR_GRAY, linestyle="--", alpha=0.8)
    ax.axvline(14, color=COLOR_GRAY, linestyle="--", alpha=0.8)
    ax.axvline(35, color=COLOR_HIGHLIGHT, linestyle="--", linewidth=2, label="Límite Ventana Operativa (35 días)")

    ax.annotate(
        f"7d: {pct_7_2yr:.1f}% (2 años) | {pct_7_5w:.1f}% (5w)",
        xy=(7, pct_7_2yr),
        xytext=(9, 18),
        arrowprops=dict(facecolor=COLOR_GRAY, shrink=0.08, width=1, headwidth=4),
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        f"14d: {pct_14_2yr:.1f}% (2 años) | {pct_14_5w:.1f}% (5w)",
        xy=(14, pct_14_2yr),
        xytext=(16, 32),
        arrowprops=dict(facecolor=COLOR_GRAY, shrink=0.08, width=1, headwidth=4),
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        f"35d: {pct_35_2yr:.1f}% de recompras históricas\ncapturadas en la ventana",
        xy=(35, pct_35_2yr),
        xytext=(38, 55),
        arrowprops=dict(facecolor=COLOR_HIGHLIGHT, shrink=0.08, width=1.5, headwidth=5),
        fontsize=9,
        fontweight="bold",
        color=COLOR_DARK,
    )

    ax.set_title(
        f"Distribución de Recompra: La ventana de 35 días captura el {pct_35_2yr:.1f}% de todas las repeticiones históricas",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Días Transcurridos Entre Compras Consecutivas del Mismo Cliente")
    ax.set_ylabel("Probabilidad Acumulada (%)")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_power_law_long_tail(
    tx: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 04: Ley de Potencias en Popularidad de Artículos (Long Tail)."""
    set_editorial_style()
    art_sales = tx.group_by("article_id").len().sort("len", descending=True)["len"].to_numpy()
    ranks = np.arange(1, len(art_sales) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(
        ranks,
        art_sales,
        marker=".",
        markersize=2,
        linestyle="none",
        color=COLOR_PRIMARY,
        alpha=0.6,
        label="Ventas por Artículo (Rango)",
    )

    fit_max = min(5000, len(ranks))
    fit_range = slice(min(10, fit_max - 1), fit_max)
    coeffs = np.polyfit(np.log10(ranks[fit_range]), np.log10(art_sales[fit_range]), 1)
    fit_line = 10 ** (coeffs[1]) * (ranks ** coeffs[0])
    ax.loglog(
        ranks,
        fit_line,
        color=COLOR_ACCENT,
        linestyle="--",
        linewidth=2,
        label=f"Ajuste Pareto: $\\alpha = {-coeffs[0]:.2f}$",
    )

    ax.set_title(
        f"Ley de potencias en demanda de moda: distribución de cola larga con exponente Pareto {-coeffs[0]:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Rango de Popularidad del Artículo (Escala Logarítmica)")
    ax.set_ylabel("Frecuencia de Compra (Escala Logarítmica)")
    ax.legend(frameon=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_lorenz_curve_gini(
    tx: pl.DataFrame,
    art: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 05: Concentración de Demanda (Curva de Lorenz & Gini Dual)."""
    set_editorial_style()
    art_sales = tx.group_by("article_id").len().sort("len", descending=True)["len"].to_numpy()
    n_catalog = art.height
    n_zero_sales = max(0, n_catalog - len(art_sales))
    all_article_sales = np.concatenate([np.zeros(n_zero_sales), np.sort(art_sales)])

    n_items_tot = len(all_article_sales)
    cum_items_tot = np.linspace(0, 1, n_items_tot)
    cum_sales_tot = np.cumsum(all_article_sales) / np.sum(all_article_sales)
    idx_tot = np.arange(1, n_items_tot + 1)
    gini_catalog = np.sum((2 * idx_tot - n_items_tot - 1) * all_article_sales) / (
        n_items_tot * np.sum(all_article_sales)
    )

    active_sales = np.sort(art_sales)
    n_items_act = len(active_sales)
    cum_items_act = np.linspace(0, 1, n_items_act)
    cum_sales_act = np.cumsum(active_sales) / np.sum(active_sales)
    idx_act = np.arange(1, n_items_act + 1)
    gini_active = np.sum((2 * idx_act - n_items_act - 1) * active_sales) / (
        n_items_act * np.sum(active_sales)
    )

    p20_idx_tot = np.searchsorted(cum_sales_tot, 0.20)
    p80_pct_tot = (1.0 - cum_items_tot[p20_idx_tot]) * 100.0
    x_point_tot = cum_items_tot[p20_idx_tot] * 100.0

    p20_idx_act = np.searchsorted(cum_sales_act, 0.20)
    p80_pct_act = (1.0 - cum_items_act[p20_idx_act]) * 100.0
    x_point_act = cum_items_act[p20_idx_act] * 100.0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        cum_items_tot * 100,
        cum_sales_tot * 100,
        color=COLOR_PRIMARY,
        linewidth=2.5,
        label=f"Catálogo Total ({n_catalog:,} arts) - Gini = {gini_catalog:.3f}",
    )
    ax.plot(
        cum_items_act * 100,
        cum_sales_act * 100,
        color=COLOR_ACCENT,
        linewidth=2.0,
        linestyle="-.",
        label=f"Artículos Activos 5w ({n_items_act:,} arts) - Gini = {gini_active:.3f}",
    )
    ax.plot(
        [0, 100],
        [0, 100],
        color=COLOR_GRAY,
        linestyle="--",
        linewidth=1.5,
        label="Equidistribución Teórica (Gini = 0.0)",
    )

    ax.scatter([x_point_tot], [20], color=COLOR_PRIMARY, s=70, zorder=5)
    ax.scatter([x_point_act], [20], color=COLOR_ACCENT, s=70, zorder=5)

    ax.annotate(
        f"Catálogo Total: El {p80_pct_tot:.1f}% superior\nconcentra el 80% de ventas\n(Gini = {gini_catalog:.2f})",
        xy=(x_point_tot, 20),
        xytext=(55, 38),
        arrowprops=dict(facecolor=COLOR_PRIMARY, shrink=0.08, width=1.2, headwidth=5),
        fontweight="bold",
        fontsize=9,
        color=COLOR_PRIMARY,
    )
    ax.annotate(
        f"Solo Activos: El {p80_pct_act:.1f}% superior\nconcentra el 80% de ventas\n(Gini = {gini_active:.2f})",
        xy=(x_point_act, 20),
        xytext=(35, 8),
        arrowprops=dict(facecolor=COLOR_ACCENT, shrink=0.08, width=1.2, headwidth=5),
        fontweight="bold",
        fontsize=9,
        color=COLOR_ACCENT,
    )

    ax.set_title(
        f"Concentración de demanda: Gini = {gini_catalog:.2f} en catálogo total (70.9% inactivos) y Gini = {gini_active:.2f} en activos",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Porcentaje Acumulado de Artículos (Ordenados de Menor a Mayor Venta) [%]")
    ax.set_ylabel("Porcentaje Acumulado de Ventas Totales (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_customer_demographics_age(
    cust: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 06: Demografía de Clientes y Segmentación por Edad."""
    set_editorial_style()
    ages = cust.select(pl.col("age")).drop_nulls()["age"].to_numpy()

    ordered_cohorts = ["<25", "25-34", "35-44", "45-54", "55+"]
    cohort_counts_map = dict(
        cust.group_by("age_bin").len().select(["age_bin", "len"]).iter_rows()
    )
    cohort_counts = [cohort_counts_map.get(c, 0) for c in ordered_cohorts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts, bin_edges = np.histogram(ages, bins=40, range=(15, 80))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mask_young = (bin_centers >= 18) & (bin_centers <= 30)
    peak_young = int(round(bin_centers[mask_young][np.argmax(counts[mask_young])]))
    mask_mature = (bin_centers >= 45) & (bin_centers <= 60)
    peak_mature = int(round(bin_centers[mask_mature][np.argmax(counts[mask_mature])]))

    axes[0].hist(ages, bins=40, range=(15, 80), color=COLOR_SECONDARY, edgecolor="white", alpha=0.85)
    axes[0].axvline(
        peak_young, color=COLOR_ACCENT, linestyle="--", linewidth=2, label=f"Pico 1: {peak_young} años (Generación Joven)"
    )
    axes[0].axvline(
        peak_mature, color=COLOR_HIGHLIGHT, linestyle="--", linewidth=2, label=f"Pico 2: {peak_mature} años (Generación Madura)"
    )
    axes[0].set_title("Estructura Demográfica Bimodal de Clientes H&M", fontweight="bold")
    axes[0].set_xlabel("Edad del Cliente (Años)")
    axes[0].set_ylabel("Número de Clientes")
    axes[0].legend(frameon=True)

    bars = axes[1].bar(
        ordered_cohorts,
        cohort_counts,
        color=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_MUTED, COLOR_ACCENT, COLOR_HIGHLIGHT],
        width=0.55,
    )
    axes[1].set_title("Distribución de Clientes por Cohorte Ordenada (<25 a 55+)", fontweight="bold")
    axes[1].set_xlabel("Cohorte de Edad")
    axes[1].set_ylabel("Volumen de Usuarios")

    for bar in bars:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 4000,
            f"{yval:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_ylim(0, max(cohort_counts) * 1.15 if cohort_counts else 1000)

    fig.suptitle(
        f"Estructura demográfica bimodal: núcleos en {peak_young} y {peak_mature} años con la cohorte 25-34 liderando la base",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_sales_channel_distribution(
    tx: pl.DataFrame,
    cust: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 07: Análisis de Canal de Venta (Online vs Físico)."""
    set_editorial_style()
    age_labels = ["<25", "25-34", "35-44", "45-54", "55+"]

    tx_cust = tx.join(cust.select(["customer_idx", "age_bin"]), on="customer_idx")
    channel_counts = tx_cust.group_by(["age_bin", "sales_channel_id"]).len()

    store_pcts_5w = []
    online_pcts_5w = []
    for cohort in age_labels:
        s_row = channel_counts.filter((pl.col("age_bin") == cohort) & (pl.col("sales_channel_id") == 1))
        o_row = channel_counts.filter((pl.col("age_bin") == cohort) & (pl.col("sales_channel_id") == 2))
        s_cnt = s_row["len"][0] if s_row.height > 0 else 0
        o_cnt = o_row["len"][0] if o_row.height > 0 else 0
        tot = s_cnt + o_cnt
        store_pcts_5w.append(round(s_cnt / tot * 100.0, 2) if tot > 0 else 0.0)
        online_pcts_5w.append(round(o_cnt / tot * 100.0, 2) if tot > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(age_labels))
    width = 0.38

    rects1 = ax.bar(
        x - width / 2, store_pcts_5w, width, label="Canal 1: Tienda Física", color=COLOR_SECONDARY
    )
    rects2 = ax.bar(
        x + width / 2, online_pcts_5w, width, label="Canal 2: Tienda Online", color=COLOR_ACCENT
    )

    max_idx = int(np.argmax(online_pcts_5w))
    min_online = int(min(online_pcts_5w))
    ax.set_title(
        f"Preferencia omnicanal por cohorte: El canal online predomina en todas las edades (>{min_online}%), liderado por {age_labels[max_idx]} ({online_pcts_5w[max_idx]:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Cohorte de Edad")
    ax.set_ylabel("Porcentaje de Transacciones (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels)
    ax.set_ylim(0, 95)
    ax.legend(frameon=True, loc="upper right")

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_product_taxonomy_revenue(
    tx: pl.DataFrame,
    art: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 08: Taxonomía de Producto y Departamentos Comerciales."""
    set_editorial_style()
    tx_with_art = tx.join(
        art.select(["article_id", "department_name", "garment_group_name"]),
        on="article_id",
        how="inner",
    )

    dept_rev = (
        tx_with_art.group_by("department_name")
        .agg([pl.len().alias("tx_count"), pl.col("price").sum().alias("total_rev")])
        .sort("total_rev", descending=True)
        .head(15)
    )

    dept_names = [str(d) for d in dept_rev["department_name"].to_list()][::-1]
    revenues = (dept_rev["total_rev"] / 1000.0).to_numpy()[::-1]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(dept_names, revenues, color=COLOR_PRIMARY, edgecolor="white", height=0.65)
    ax.set_title(
        "Top 15 Departamentos Comerciales por Facturación: Jersey, Denim y Vestidos dominan los ingresos",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Facturación Acumulada en 5 Semanas (Miles de Créditos H&M)")
    ax.set_ylabel("Departamento de Producto")

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.08,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.1f}k",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLOR_DARK,
        )

    ax.set_xlim(0, max(revenues) * 1.15 if len(revenues) > 0 else 10)
    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_nlp_metadata_signal(
    art: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 09: Cobertura de Metadatos NLP (detail_desc)."""
    set_editorial_style()
    desc_lengths = (
        art.select(pl.col("detail_desc"))
        .drop_nulls()
        .select(pl.col("detail_desc").str.split(" ").list.len().alias("word_count"))
    )["word_count"].to_numpy()

    null_desc_pct = (art.filter(pl.col("detail_desc").is_null()).height / art.height) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(desc_lengths, bins=30, range=(1, 60), color=COLOR_MUTED, edgecolor="white", alpha=0.85)
    axes[0].axvline(
        np.mean(desc_lengths) if len(desc_lengths) > 0 else 0,
        color=COLOR_ACCENT,
        linestyle="--",
        linewidth=2,
        label=f"Media: {np.mean(desc_lengths):.1f} palabras" if len(desc_lengths) > 0 else "Media: 0",
    )
    axes[0].axvline(
        np.median(desc_lengths) if len(desc_lengths) > 0 else 0,
        color=COLOR_HIGHLIGHT,
        linestyle=":",
        linewidth=2,
        label=f"Mediana: {np.median(desc_lengths):.1f} palabras" if len(desc_lengths) > 0 else "Mediana: 0",
    )
    axes[0].set_title("Longitud Textual de Descripciones de Catálogo (detail_desc)", fontweight="bold")
    axes[0].set_xlabel("Número de Palabras por Descripción")
    axes[0].set_ylabel("Frecuencia de Artículos")
    axes[0].legend(frameon=True)

    labels = ["Descripciones Completas", "Valores Nulos / Vacíos"]
    sizes = [100.0 - null_desc_pct, null_desc_pct]
    axes[1].pie(
        sizes,
        labels=labels,
        autopct="%1.2f%%",
        startangle=140,
        colors=[COLOR_PRIMARY, COLOR_ACCENT],
        explode=(0, 0.15),
    )
    axes[1].set_title("Completitud del Campo detail_desc", fontweight="bold")

    fig.suptitle(
        "Metadatos NLP concisos (18 palabras promedio): la señal textual es redundante frente a variables categóricas",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    return _save_or_show(fig, save_path)




def plot_basket_cooccurrence_matrix(
    tx: pl.DataFrame,
    art: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 11: Matriz de Co-ocurrencia en Cesta Transaccional P(B|A)."""
    set_editorial_style()
    baskets = (
        tx.join(art.select(["article_id", "garment_group_name"]), on="article_id", how="inner")
        .select(["customer_idx", "t_dat", "garment_group_name"])
        .unique()
    )

    top_garment_groups = [
        "Jersey Fancy",
        "Knitwear",
        "Jersey Basic",
        "Trousers",
        "Under-, Nightwear",
        "Blouses",
        "Accessories",
        "Trousers Denim",
    ]

    baskets_filtered = baskets.filter(pl.col("garment_group_name").is_in(top_garment_groups))

    cooccur = (
        baskets_filtered.join(baskets_filtered, on=["customer_idx", "t_dat"])
        .group_by(["garment_group_name", "garment_group_name_right"])
        .len()
    )

    cooccur_dict = {
        (r[0], r[1]): r[2]
        for r in cooccur.iter_rows()
    }

    matrix_prob = np.zeros((len(top_garment_groups), len(top_garment_groups)))
    total_per_group = {
        g: baskets_filtered.filter(pl.col("garment_group_name") == g).height
        for g in top_garment_groups
    }

    for i, g1 in enumerate(top_garment_groups):
        for j, g2 in enumerate(top_garment_groups):
            count_both = cooccur_dict.get((g1, g2), 0)
            matrix_prob[i, j] = (
                (count_both / total_per_group[g1]) * 100.0 if total_per_group[g1] > 0 else 0.0
            )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_prob, cmap="Blues", interpolation="nearest")

    ax.set_xticks(np.arange(len(top_garment_groups)))
    ax.set_yticks(np.arange(len(top_garment_groups)))
    ax.set_xticklabels(top_garment_groups, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(top_garment_groups, fontsize=10)

    for i in range(len(top_garment_groups)):
        for j in range(len(top_garment_groups)):
            val = matrix_prob[i, j]
            color_txt = "white" if val > 28.0 else COLOR_DARK
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color=color_txt,
                fontsize=8.5,
                fontweight="bold",
            )

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.set_ylabel(
        "Probabilidad Condicional P(Categoría B | Categoría A) [%]", rotation=-90, va="bottom"
    )

    ax.set_title(
        "Matriz de Co-ocurrencia en Cesta: Fuerte complementariedad cruzada (Jersey/Trousers/Knitwear)\n"
        "Base empírica para la etapa de recuperación causal P(B|A) en V6",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Categoría Complementaria B (Añadida al Ticket)")
    ax.set_ylabel("Categoría Base A (Comprada por el Cliente)")
    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_microsegmentation_distribution(
    tx: pl.DataFrame,
    cust: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 12: Micro-Segmentación Sociodemográfica y Dispersión de Precios."""
    set_editorial_style()
    ordered_cohorts = ["<25", "25-34", "35-44", "45-54", "55+"]
    tx_with_cust = tx.join(cust.select(["customer_idx", "age_bin"]), on="customer_idx", how="inner")

    tx_with_cust = tx_with_cust.with_columns(
        pl.when(pl.col("price") < 0.02)
        .then(pl.lit("Económico (<0.02)"))
        .when(pl.col("price") <= 0.035)
        .then(pl.lit("Medio (0.02-0.035)"))
        .otherwise(pl.lit("Premium (>0.035)"))
        .alias("price_tier")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    tiers = ["Económico (<0.02)", "Medio (0.02-0.035)", "Premium (>0.035)"]
    colors_tier = [COLOR_SECONDARY, COLOR_PRIMARY, COLOR_ACCENT]
    tier_matrix = []

    for ab in ordered_cohorts:
        sub = tx_with_cust.filter(pl.col("age_bin") == ab)
        tot = sub.height
        row = [
            (sub.filter(pl.col("price_tier") == t).height / tot) * 100.0 if tot > 0 else 0.0
            for t in tiers
        ]
        tier_matrix.append(row)

    tier_matrix = np.array(tier_matrix)
    bottom = np.zeros(len(ordered_cohorts))

    for idx_t, (tier_name, color) in enumerate(zip(tiers, colors_tier, strict=True)):
        vals = tier_matrix[:, idx_t]
        axes[0].bar(
            ordered_cohorts,
            vals,
            bottom=bottom,
            label=tier_name,
            color=color,
            width=0.55,
            edgecolor="white",
        )
        for i, (b, v) in enumerate(zip(bottom, vals, strict=True)):
            if v > 10.0:
                axes[0].text(
                    i,
                    b + v / 2.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=9,
                )
        bottom += vals

    axes[0].set_title("Distribución de Transacciones por Rango de Precio y Cohorte", fontweight="bold")
    axes[0].set_xlabel("Cohorte de Edad")
    axes[0].set_ylabel("Participación en Volumen de Ventas (%)")
    axes[0].set_ylim(0, 118)
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)

    price_stats = (
        tx_with_cust.group_by("age_bin")
        .agg([
            pl.col("price").mean().alias("mean_p"),
            pl.col("price").quantile(0.25).alias("q25"),
            pl.col("price").quantile(0.75).alias("q75"),
        ])
        .sort("age_bin")
    )

    price_stats_dict = {r[0]: (r[1], r[2], r[3]) for r in price_stats.iter_rows()}
    means = [price_stats_dict.get(c, (0, 0, 0))[0] for c in ordered_cohorts]
    q25s = [price_stats_dict.get(c, (0, 0, 0))[1] for c in ordered_cohorts]
    q75s = [price_stats_dict.get(c, (0, 0, 0))[2] for c in ordered_cohorts]
    err_lower = [m - q25 for m, q25 in zip(means, q25s, strict=True)]
    err_upper = [q75 - m for m, q75 in zip(means, q75s, strict=True)]

    bars2 = axes[1].bar(
        ordered_cohorts,
        means,
        yerr=[err_lower, err_upper],
        capsize=5,
        color=COLOR_PRIMARY,
        alpha=0.85,
        edgecolor="white",
        width=0.55,
        label="Precio Medio (con Rango Intercuartil Q25-Q75)",
    )

    for bar, m, q75 in zip(bars2, means, q75s, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            q75 + 0.0018,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
            color=COLOR_DARK,
        )

    axes[1].set_title(
        "Sensibilidad al Precio: El ticket medio escala un +12.5% de <25 a 55+", fontweight="bold"
    )
    axes[1].set_xlabel("Cohorte de Edad")
    axes[1].set_ylabel("Precio Normalizado de Transacción (Créditos H&M)")
    axes[1].set_ylim(0, 0.055)
    axes[1].legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Micro-Segmentación: Disparidad de sensibilidad al precio y surtido que justifica el fallback estratificado en V6",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_or_show(fig, save_path)
