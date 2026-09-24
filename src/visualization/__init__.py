"""Módulo de visualización para el sistema de recomendación.

Provee estilos centralizados y funciones de trazado para análisis exploratorio,
candidatos, importancia de features y ablación.
"""

from src.visualization.ablation_plots import (
    plot_leave_one_out,
    plot_pareto_frontier,
    plot_temporal_window,
)
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
from src.visualization.model_plots import (
    plot_feature_importance_from_df,
    plot_feature_importance_from_model,
    plot_feature_importance_from_table,
)
from src.visualization.style import (
    COLOR_ACCENT,
    COLOR_DARK,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_HIGHLIGHT,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    PALETTE,
    set_editorial_style,
)

__all__ = [
    "COLOR_ACCENT",
    "COLOR_DARK",
    "COLOR_GRAY",
    "COLOR_GREEN",
    "COLOR_HIGHLIGHT",
    "COLOR_MUTED",
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "PALETTE",
    "set_editorial_style",
    "plot_sparsity_interaction_space",
    "plot_weekly_temporal_trend",
    "plot_repurchase_lag_distribution",
    "plot_power_law_long_tail",
    "plot_lorenz_curve_gini",
    "plot_customer_demographics_age",
    "plot_sales_channel_distribution",
    "plot_product_taxonomy_revenue",
    "plot_nlp_metadata_signal",
    "plot_basket_cooccurrence_matrix",
    "plot_microsegmentation_distribution",
    "plot_temporal_window",
    "plot_leave_one_out",
    "plot_pareto_frontier",
    "plot_feature_importance_from_df",
    "plot_feature_importance_from_table",
    "plot_feature_importance_from_model",
]
