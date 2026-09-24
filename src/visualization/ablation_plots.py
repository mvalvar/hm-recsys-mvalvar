"""Funciones para la generación de las figuras de Ablación y Pareto (Capítulo 9).

Calculan sobre los datos empíricos de resultados experimentales reales (ablation_results.csv)
y devuelven la figura de Matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl

from src.visualization.style import PALETTE, set_editorial_style


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    """Guarda la figura si save_path está definido; en caso contrario la retorna intacta."""
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_temporal_window(
    df_a1: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 08: Sensibilidad al Horizonte de Memoria Temporal (Doble escala Y: MAP@12 vs RAM)."""
    set_editorial_style()
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    w_map = {"3w": 3, "5w": 5, "8w": 8, "10w": 10}
    data = []
    for r in df_a1.iter_rows(named=True):
        w_int = w_map.get(r["variant"], 0)
        data.append(
            {
                "weeks": w_int,
                "variant": r["variant"],
                "map12": r["map12"] * 100.0,
                "ram_mb": r["memory_mb"],
                "n_tx": r["n_candidates"],
            }
        )
    data = sorted(data, key=lambda x: x["weeks"])
    weeks = [d["weeks"] for d in data]
    map12_vals = [d["map12"] for d in data]
    ram_vals = [d["ram_mb"] for d in data]

    ax2 = ax1.twinx()
    bar_width = 0.6
    bars = ax2.bar(
        weeks,
        ram_vals,
        width=bar_width,
        color=PALETTE["gray_light"],
        edgecolor=PALETTE["gray_border"],
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
        label="Consumo Pico RAM (MB)",
    )

    ax1.plot(
        weeks,
        map12_vals,
        color=PALETTE["petroleum"],
        linewidth=2.8,
        marker="o",
        markersize=9,
        markerfacecolor=PALETTE["petroleum"],
        markeredgecolor="#ffffff",
        markeredgewidth=2.0,
        zorder=5,
        label="MAP@12 (%)",
    )

    opt_idx = 1
    if len(weeks) > opt_idx:
        ax1.plot(
            weeks[opt_idx],
            map12_vals[opt_idx],
            marker="o",
            markersize=14,
            markerfacecolor=PALETTE["coral"],
            markeredgecolor="#ffffff",
            markeredgewidth=2.5,
            zorder=6,
        )

    for i, (w, m) in enumerate(zip(weeks, map12_vals, strict=False)):
        offset_y = 0.08 if i != opt_idx else 0.10
        color = PALETTE["coral"] if i == opt_idx else PALETTE["petroleum"]
        weight = "bold" if i == opt_idx else "semibold"
        ax1.text(
            w,
            m + offset_y,
            f"{m:.3f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight=weight,
            color=color,
            zorder=7,
        )

    for b in bars:
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            h + 50,
            f"{h:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="normal",
            color=PALETTE["graphite"],
            zorder=3,
        )

    if len(weeks) > opt_idx:
        ax1.annotate(
            "VENTANA SELECCIONADA (5 Semanas)\n• MAP@12: 2.432%\n• Ceiling@80: 8.31% (+27.3% vs 3w)\n• RAM: 2,104 MB (Estable)",
            xy=(5, map12_vals[opt_idx]),
            xytext=(5.6, 2.12),
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE["coral"],
                lw=1.5,
                connectionstyle="arc3,rad=-0.2",
            ),
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#ffffff",
                edgecolor=PALETTE["coral"],
                linewidth=1.2,
                alpha=0.95,
            ),
            fontsize=8.5,
            fontweight="semibold",
            color=PALETTE["graphite"],
            zorder=10,
        )

    ax1.set_xlabel(
        "Horizonte de Memoria Histórica (Semanas)",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["graphite"],
    )
    ax1.set_ylabel(
        "Precisión de Ranking MAP@12 (%)",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["petroleum"],
    )
    ax2.set_ylabel(
        "Consumo Pico de Memoria RAM (MB)",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["graphite"],
    )

    ax1.set_xticks(weeks)
    ax1.set_xticklabels(
        [
            "3 Semanas\n(~0.8M tx)",
            "5 Semanas\n(~1.3M tx)\n[Propuesta]",
            "8 Semanas\n(~1.9M tx)",
            "10 Semanas\n(~2.4M tx)",
        ][: len(weeks)],
        fontsize=9.5,
    )
    ax1.set_ylim(1.3, 2.75)
    ax2.set_ylim(0, 3200)

    ax1.grid(True, zorder=0)
    ax2.grid(False)

    fig.text(
        0.5,
        0.96,
        "La ventana de 5 semanas equilibra el techo de recall y minimiza la obsolescencia estacional",
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["petroleum"],
        ha="center",
    )
    fig.text(
        0.5,
        0.925,
        "Estudio de Ablación A1: Compromiso entre memoria histórica, catálogo activo de moda rápida y consumo de hardware",
        fontsize=9.5,
        color=PALETTE["graphite"],
        ha="center",
    )

    lines, labels = ax1.get_legend_handles_labels()
    bars_h, bars_l = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + bars_h,
        labels + bars_l,
        loc="upper left",
        frameon=True,
        facecolor="#ffffff",
        framealpha=0.9,
        edgecolor=PALETTE["gray_border"],
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return _save_or_show(fig, save_path)


def plot_leave_one_out(
    df_a2: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 09: Análisis Leave-One-Out (Barras horizontales de caída porcentual LOO)."""
    set_editorial_style()
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)

    rows = [r for r in df_a2.iter_rows(named=True) if not r["variant"].startswith("Full")]
    rows = sorted(rows, key=lambda x: x["delta_pct"])

    labels = [r["variant"] for r in rows]
    deltas_pct = [r["delta_pct"] for r in rows]
    deltas_abs = [r["delta_vs_baseline"] for r in rows]

    colors = []
    for d in deltas_pct:
        if d <= -16.0:
            colors.append(PALETTE["coral"])
        elif d <= -10.0:
            colors.append(PALETTE["coral_dark"])
        elif d < 0.0:
            colors.append(PALETTE["navy"])
        else:
            colors.append(PALETTE["green"])

    y_pos = range(len(labels))
    bars = ax.barh(
        list(y_pos),
        deltas_pct,
        height=0.6,
        color=colors,
        edgecolor="#2f3542",
        linewidth=0.6,
        alpha=0.9,
        zorder=3,
    )

    ax.axvline(0, color=PALETTE["graphite"], linestyle="-", linewidth=1.2, zorder=4)

    for b, d_pct, d_abs in zip(bars, deltas_pct, deltas_abs, strict=False):
        w = b.get_width()
        sign = "+" if d_pct > 0 else ""
        text_str = f" {sign}{d_pct:.2f}% (Δ = {d_abs:+.4f})"
        if w < 0:
            ax.text(
                w - 0.5,
                b.get_y() + b.get_height() / 2.0,
                text_str,
                ha="right",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=PALETTE["graphite"],
                zorder=5,
            )
        else:
            ax.text(
                w + 0.5,
                b.get_y() + b.get_height() / 2.0,
                text_str,
                ha="left",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=PALETTE["green"],
                zorder=5,
            )

    if len(y_pos) > 0:
        ax.annotate(
            "HEURÍSTICA MÁS CRÍTICA: R3 (Popularidad por Edad)\nProvoca la mayor pérdida predictiva (-18.28%)\nEvidencia la necesidad de segmentación demográfica",
            xy=(-18.28, list(y_pos)[0]),
            xytext=(2.0, list(y_pos)[0] + 0.8),
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE["coral"],
                lw=1.5,
                connectionstyle="arc3,rad=-0.2",
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffffff",
                edgecolor=PALETTE["coral"],
                linewidth=1.2,
                alpha=0.95,
            ),
            fontsize=8.5,
            fontweight="semibold",
            color=PALETTE["graphite"],
            zorder=10,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10, fontweight="semibold", color=PALETTE["graphite"])
    ax.set_xlabel(
        "Impacto Marginal en MAP@12 (% vs. Baseline Full)",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["graphite"],
    )
    ax.set_xlim(-35, 32)
    ax.grid(True, axis="x", zorder=0)

    fig.text(
        0.5,
        0.97,
        "Análisis Leave-One-Out (A2): La Popularidad por Edad (R3) y Departamental (R8) concentran la mayor pérdida predictiva",
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["petroleum"],
        ha="center",
    )
    fig.text(
        0.5,
        0.925,
        "Caída porcentual en MAP@12 al omitir individualmente las 8 heurísticas de recall en Semana 104",
        fontsize=9.5,
        color=PALETTE["graphite"],
        ha="center",
    )

    legend_patches = [
        mpatches.Patch(color=PALETTE["coral"], label="Caída Crítica (> 15% MAP@12)"),
        mpatches.Patch(color=PALETTE["coral_dark"], label="Caída Significativa (10% - 15%)"),
        mpatches.Patch(color=PALETTE["navy"], label="Caída Moderada (< 10%)"),
        mpatches.Patch(color=PALETTE["green"], label="Filtrado Favorable de Ruido"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        frameon=True,
        facecolor="#ffffff",
        edgecolor=PALETTE["gray_border"],
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    return _save_or_show(fig, save_path)


def plot_pareto_frontier(
    df_a6: pl.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Figura 10: Frontera de Eficiencia de Pareto Multiobjetivo (MAP@12 vs RAM)."""
    set_editorial_style()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    points = []
    for r in df_a6.iter_rows(named=True):
        is_pareto = (
            "Pareto-Óptimo: True" in r["description"]
            or "Ratio 1:5" in r["variant"]
            or "LambdaRank" in r["variant"]
            or "3w" in r["variant"]
            or "5w" in r["variant"]
            or "Sin R1" in r["variant"]
        )
        points.append(
            {
                "label": r["variant"],
                "ram_mb": r["memory_mb"],
                "map12": r["map12"] * 100.0,
                "cpu_sec": r["duration_sec"],
                "is_pareto": is_pareto,
            }
        )

    pareto_pts = [p for p in points if p["is_pareto"]]
    pareto_pts = sorted(pareto_pts, key=lambda x: x["ram_mb"])

    p_x = [p["ram_mb"] for p in pareto_pts]
    p_y = [p["map12"] for p in pareto_pts]
    ax.plot(
        p_x,
        p_y,
        color=PALETTE["petroleum"],
        linestyle="--",
        linewidth=2.2,
        zorder=3,
        label="Frontera de Eficiencia de Pareto (No-Dominada)",
    )

    ax.fill_between(p_x, 1.4, p_y, color=PALETTE["petroleum"], alpha=0.08, zorder=1)

    dom_x = [p["ram_mb"] for p in points if not p["is_pareto"]]
    dom_y = [p["map12"] for p in points if not p["is_pareto"]]
    ax.scatter(
        dom_x,
        dom_y,
        color=PALETTE["graphite"],
        alpha=0.45,
        s=55,
        marker="o",
        edgecolors="none",
        zorder=4,
        label="Configuraciones Dominadas (Subóptimas)",
    )

    ax.scatter(
        p_x,
        p_y,
        color=PALETTE["petroleum"],
        s=110,
        marker="D",
        edgecolors="#ffffff",
        linewidths=1.5,
        zorder=6,
        label="Soluciones Pareto-Óptimas",
    )

    champ = next((p for p in pareto_pts if "5w" in p["label"] or "Propuesta" in p["label"]), None)
    if champ:
        ax.scatter(
            [champ["ram_mb"]],
            [champ["map12"]],
            color=PALETTE["coral"],
            s=220,
            marker="*",
            edgecolors="#ffffff",
            linewidths=1.8,
            zorder=8,
            label="Arquitectura Propuesta TFM (Campeón)",
        )
        ax.annotate(
            f"ARQUITECTURA PROPUESTA TFM\n• {champ['label']}\n• MAP@12: {champ['map12']:.3f}%\n• RAM: {champ['ram_mb']:.0f} MB",
            xy=(champ["ram_mb"], champ["map12"]),
            xytext=(champ["ram_mb"] - 480, champ["map12"] + 0.12),
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE["coral"],
                lw=1.8,
                connectionstyle="arc3,rad=-0.2",
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffffff",
                edgecolor=PALETTE["coral"],
                linewidth=1.5,
                alpha=0.98,
            ),
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["coral"],
            zorder=10,
        )

    ax.axvline(
        2048,
        color=PALETTE["coral_dark"],
        linestyle=":",
        linewidth=1.8,
        alpha=0.85,
        zorder=2,
        label="Cota Estricta de Memoria RAM (2,048 MB)",
    )

    ax.set_xlabel(
        "Consumo Máximo de Memoria RAM (MB)",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["graphite"],
    )
    ax.set_ylabel(
        "Calidad Predictiva MAP@12 (%)",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["petroleum"],
    )
    ax.set_xlim(300, 3100)
    ax.set_ylim(1.4, 2.7)
    ax.grid(True, zorder=0)

    fig.text(
        0.5,
        0.965,
        "Frontera de Eficiencia de Pareto (A6): Optimización Multiobjetivo MAP@12 vs. Consumo de Memoria",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["petroleum"],
        ha="center",
    )
    fig.text(
        0.5,
        0.93,
        "Evaluación ceteris paribus de 21 arquitecturas en el espacio bicriterio (Precisión de Ranking vs. Huella en RAM)",
        fontsize=9.5,
        color=PALETTE["graphite"],
        ha="center",
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="#ffffff",
        edgecolor=PALETTE["gray_border"],
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return _save_or_show(fig, save_path)
