"""Script de Generación de Figuras para el Capítulo 5 (Generación de Candidatos / Retrieval).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Produce:
1. results/figures/fig_cap05_01_recall_ceiling_curve.png (y fig_04_recall_ceiling_curve.png)
2. results/figures/fig_cap05_02_candidate_sources_overlap.png (y fig_05_candidate_sources_overlap.png)
3. results/figures/fig_cap05_03_marginal_recall_gain.png (y fig_06_marginal_recall_gain.png)

Uso:
    python scripts/generate_candidate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, FIGURES_DIR
from src.candidates.generators import (
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
from src.evaluation.metrics import hit_rate_at_k, recall_at_k
from src.utils.validation import split_transactions_temporal


def main() -> None:
    print("=" * 80)
    print("  GENERACIÓN DE FIGURAS DEL CAPÍTULO 5 (POOL DE CANDIDATOS / RECALL)")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    primary_color = "#003f5c"
    secondary_color = "#2f4b7c"
    accent_color = "#ff6361"

    print("-> Cargando datasets desde data_processed/...")
    tx = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    cust = pl.read_parquet(DATA_PROCESSED_DIR / "customers.parquet")
    art = pl.read_parquet(DATA_PROCESSED_DIR / "articles.parquet")

    # Partición temporal estricta (últimos 7 días para validación)
    split = split_transactions_temporal(tx, val_days=7, history_days=35)
    train_df = split.train_df
    val_df = split.val_df

    print(f"  * Transacciones train : {train_df.height:,}")
    print(f"  * Transacciones val   : {val_df.height:,}")

    # Generación de las 8 heurísticas sobre el historial de entrenamiento
    print("-> Computando heurísticas de recall (R1 a R8)...")
    r1 = generate_repurchase(train_df)
    r2 = generate_global_popularity(train_df)
    r3 = generate_age_group_popularity(train_df, cust)
    r4 = generate_channel_popularity(train_df)
    r5 = generate_item_cf(train_df)
    r6 = generate_product_family(train_df, art)
    r7 = generate_trending_items(train_df)
    r8 = generate_user_dept_popularity(train_df, art)

    # Consolidar a los mejores 80 candidatos para evaluación de recall ceiling
    print("-> Consolidando candidatos deduplicados...")
    cand = consolidate_candidates(r1, r2, r3, r4, r5, r6, r7, r8, max_per_user=80)

    val_pairs = (
        val_df.select(["customer_idx", "article_id"])
        .unique()
        .with_columns(pl.lit(1).alias("is_target"))
    )
    cand_hits = cand.join(val_pairs, on=["customer_idx", "article_id"], how="left").with_columns(
        pl.col("is_target").fill_null(0)
    )

    val_truth = val_df.group_by("customer_idx").agg(
        pl.col("article_id").unique().alias("actual_items")
    )
    cand_lists = cand.group_by("customer_idx").agg(pl.col("article_id").alias("predicted_items"))
    eval_df = val_truth.join(cand_lists, on="customer_idx", how="inner")

    actuals = dict(
        zip(eval_df["customer_idx"].to_list(), eval_df["actual_items"].to_list(), strict=False)
    )
    preds = dict(
        zip(
            eval_df["customer_idx"].to_list(),
            eval_df["predicted_items"].to_list(),
            strict=False,
        )
    )

    # FIGURA 5.1: Curva de Techo de Recall en Función de k
    print("-> Generando Figura 5.1: Curva de Techo de Recall...")
    k_range = list(range(1, 81))
    rec_curve = [recall_at_k(actuals, preds, k=k) * 100.0 for k in k_range]
    hit_curve = [hit_rate_at_k(actuals, preds, k=k) * 100.0 for k in k_range]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        k_range,
        rec_curve,
        color=primary_color,
        lw=2.5,
        label="Techo de Recall@k (%)",
    )
    ax2.plot(
        k_range,
        hit_curve,
        color=accent_color,
        lw=2.0,
        ls="--",
        label="Hit Rate@k (%)",
    )

    ax1.set_xlabel("Profundidad de Corte del Pool ($k$)", fontsize=11)
    ax1.set_ylabel("Recall Teórico (%)", color=primary_color, fontsize=11)
    ax2.set_ylabel("Hit Rate (%)", color=accent_color, fontsize=11)
    ax1.set_title(
        "Figura 5.1: Curva de Techo de Recall (Cobertura vs. Tamaño del Pool $k$)",
        fontweight="bold",
        pad=12,
        fontsize=12,
    )

    # Anotación en k=80
    ax1.scatter([80], [rec_curve[-1]], color=primary_color, s=50, zorder=5)
    ax1.annotate(
        f"k=80: Recall={rec_curve[-1]:.2f}%",
        (80, rec_curve[-1]),
        textcoords="offset points",
        xytext=(-80, 10),
        arrowprops=dict(arrowstyle="->", color=primary_color),
        fontweight="bold",
        fontsize=9,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    plt.tight_layout()

    out_5_1 = FIGURES_DIR / "fig_cap05_01_recall_ceiling_curve.png"
    fig.savefig(out_5_1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Guardado: {out_5_1.name}")

    # FIGURA 5.2: Heatmap de Correlación Inter-Heurística
    print("-> Generando Figura 5.2: Heatmap de Correlación...")
    flag_cols = [c for c in cand.columns if c.startswith("is_R")]
    corr_matrix = cand.select(flag_cols).to_pandas().astype(int).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "Figura 5.2: Matriz de Correlación de Pearson Inter-Heurística (Ortogonalidad de Fuentes)",
        fontweight="bold",
        pad=12,
        fontsize=11,
    )
    plt.tight_layout()

    out_5_2 = FIGURES_DIR / "fig_cap05_02_candidate_sources_overlap.png"
    fig.savefig(out_5_2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Guardado: {out_5_2.name}")

    # FIGURA 5.3: Aciertos Totales y Exclusivos por Heurística
    print("-> Generando Figura 5.3: Aciertos Totales y Exclusivos...")
    heuristics = [c.replace("is_", "") for c in flag_cols]
    res_rows = []
    for h in heuristics:
        flag = f"is_{h}"
        total_h = cand_hits.filter(pl.col(flag) & (pl.col("is_target") == 1)).height
        other_flags = [f"is_{other}" for other in heuristics if other != h]
        excl_filter = pl.col(flag) & (pl.col("is_target") == 1)
        for oflag in other_flags:
            excl_filter = excl_filter & (~pl.col(oflag))
        excl_h = cand_hits.filter(excl_filter).height
        res_rows.append(
            {
                "Heurística": h,
                "Aciertos Totales": total_h,
                "Aciertos Exclusivos": excl_h,
            }
        )

    contrib_df = pl.DataFrame(res_rows)
    labels = contrib_df["Heurística"].to_list()
    totales = contrib_df["Aciertos Totales"].to_list()
    exclusivos = contrib_df["Aciertos Exclusivos"].to_list()

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(
        x - width / 2,
        totales,
        width,
        label="Aciertos Totales",
        color=secondary_color,
    )
    rects2 = ax.bar(
        x + width / 2,
        exclusivos,
        width,
        label="Aciertos Exclusivos",
        color=accent_color,
    )

    ax.set_ylabel("Número de Aciertos en Validación", fontsize=11)
    ax.set_xlabel("Heurística de Recall", fontsize=11)
    ax.set_title(
        "Figura 5.3: Aciertos Totales vs. Exclusivos por Regla de Recuperación",
        fontweight="bold",
        pad=12,
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.legend(loc="upper right")

    # Etiquetas sobre las barras
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_5_3 = FIGURES_DIR / "fig_cap05_03_marginal_recall_gain.png"
    fig.savefig(out_5_3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Guardado: {out_5_3.name}")

    print("=" * 80)
    print("  [OK] TODAS LAS FIGURAS DEL CAPÍTULO 5 GENERADAS EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
