"""Funciones canónicas de visualización de modelos de ranking (Feature Importance).

Permite graficar la importancia por ganancia tanto desde un DataFrame precalculado,
desde un archivo CSV de importancia, o desde el modelo booster serializado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch

from src.visualization.style import set_editorial_style


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    """Guarda la figura si save_path está definido; en caso contrario la retorna intacta."""
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_feature_importance_from_df(
    imp_df: pl.DataFrame,
    save_path: str | Path | None = None,
    max_features: int = 15,
) -> plt.Figure:
    """Genera el gráfico editorial de importancia de características a partir de un DataFrame."""
    set_editorial_style()
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    top_df = imp_df.head(max_features).reverse()
    features = top_df["feature"].to_list()
    gain_pcts = top_df["gain_pct"].to_list()

    colors = [
        "#1b4965" if any(k in f for k in ["uxa_", "n_sources", "best_rank", "is_R"]) else "#48cae4"
        for f in features
    ]

    bars = ax.barh(
        features, gain_pcts, color=colors, height=0.65, edgecolor="#0f2b3c", linewidth=0.8
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.2f}%",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )

    ax.set_title(
        "Importancia de Características por Ganancia (LGBMRanker):\n"
        "Interacciones Usuario × Prenda y Consenso Heurístico Dominan la Capacidad Predictiva",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
        color="#111111",
    )
    ax.set_xlabel(
        "Ganancia Relativa en Reducción de Pérdida LambdaRank (%)", fontsize=10.5, labelpad=10
    )
    ax.set_xlim(0, max(gain_pcts) * 1.18 if gain_pcts else 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    legend_elements = [
        Patch(
            facecolor="#1b4965",
            edgecolor="#0f2b3c",
            label="Interacción Usuario × Artículo & Meta-Features",
        ),
        Patch(
            facecolor="#48cae4",
            edgecolor="#0f2b3c",
            label="Atributos Univariantes (Usuario / Artículo)",
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right", frameon=True, facecolor="white", framealpha=0.9
    )

    plt.tight_layout()
    return _save_or_show(fig, save_path)


def plot_feature_importance_from_table(
    table_path: str | Path,
    save_path: str | Path | None = None,
    max_features: int = 15,
) -> plt.Figure:
    """Lee el CSV de feature importance y genera el gráfico."""
    df = pl.read_csv(Path(table_path))
    return plot_feature_importance_from_df(df, save_path=save_path, max_features=max_features)


def plot_feature_importance_from_model(
    model_path: str | Path,
    save_path: str | Path | None = None,
    max_features: int = 15,
) -> plt.Figure:
    """Carga un LGBMRanker serializado y genera su gráfico de importancia."""
    from src.modeling.ranker import LGBMRankerModel

    ranker = LGBMRankerModel.load(Path(model_path))
    imp_df = ranker.get_feature_importance()
    return plot_feature_importance_from_df(imp_df, save_path=save_path, max_features=max_features)
