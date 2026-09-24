"""Estilo visual y paleta cromática centralizada para el TFM.

Garantiza coherencia visual estética (paleta sobria, tipografía limpia)
en todos los scripts de generación y notebooks interactivos.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

COLOR_PRIMARY = "#003f5c"
COLOR_SECONDARY = "#2f4b7c"
COLOR_ACCENT = "#ff6361"
COLOR_HIGHLIGHT = "#ffa600"
COLOR_MUTED = "#665191"
COLOR_GRAY = "#a4b0be"
COLOR_DARK = "#1e272e"
COLOR_GREEN = "#2ed573"

PALETTE = {
    "petroleum": COLOR_PRIMARY,
    "navy": COLOR_SECONDARY,
    "purple": COLOR_MUTED,
    "magenta": "#a05195",
    "coral_dark": "#d45087",
    "coral_light": "#f95d6a",
    "coral": COLOR_ACCENT,
    "amber": COLOR_HIGHLIGHT,
    "graphite": "#2f3542",
    "gray_light": "#f1f2f6",
    "gray_border": "#dfe4ea",
    "green": COLOR_GREEN,
}


def set_editorial_style() -> None:
    """Configura los parámetros globales de Matplotlib para publicaciones de alta calidad."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
            "figure.titlesize": 13,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": "#747d8c",
            "axes.linewidth": 0.8,
            "grid.color": "#e4e7eb",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
