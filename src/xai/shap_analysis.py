r"""Módulo de Interpretabilidad y Explicabilidad Algorítmica (XAI) mediante SHAP TreeExplainer.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Fundamento Matemático:
Implementa la teoría de juegos cooperativos (Valores de Shapley) adaptada a modelos de árboles
gradient boosted mediante TreeSHAP (Lundberg et al., Nature Machine Intelligence, 2020).
Garantiza la propiedad aditiva local exacta:
    f(x) = E[f(x)] + \sum_{i=1}^M \phi_i(x)
donde:
- f(x): Predicción continua del modelo (ranking score o log-odds).
- E[f(x)]: Valor base esperado sobre el conjunto de referencia (esperanza matemática del prior).
- \phi_i(x): Contribución marginal atribuida a la característica i-ésima libre de sesgos de orden.

Estructura Funcional:
1. Explicabilidad Global: Summary Plot (beeswarm) ordenado por media absoluta |SHAP| sobre N=2.000 observaciones.
2. Explicabilidad Local: Descomposición Waterfall en 3 arquetipos de clientes de moda:
   - Arquetipo A: Cliente Fiel / Comprador de Básicos (alta recompra y afinidad departamental).
   - Arquetipo B: Explorador Joven / Tendencias (<25 años, digital, candidate sources R3 y R7).
   - Arquetipo C: Cliente Esporádico / Cold-Start Ligero (1 compra previa, anclado en popularidad de canal R4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap

from config.settings import FIGURES_DIR
from src.modeling.ranker import LGBMRankerModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchetypeProfile:
    """Perfil estructurado de un arquetipo de comprador para explicabilidad local."""

    archetype_id: str
    customer_idx: int
    article_id: int
    persona_label: str
    business_segment: str
    description: str


def compute_shap_values(
    ranker_model: LGBMRankerModel,
    X_sample: pl.DataFrame | np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    r"""Calcula la descomposición aditiva exacta de Shapley mediante TreeSHAP en C++.

    Parameters
    ----------
    ranker_model : LGBMRankerModel
        Modelo entrenado o cargado que contiene el booster de LightGBM.
    X_sample : pl.DataFrame | np.ndarray
        Matriz de características de tamaño (N, M).
    feature_names : list[str]
        Lista de nombres de las M características evaluadas.

    Returns
    -------
    shap.Explanation
        Objeto Explanation de SHAP con valores, valores base y datos brutos.
    """
    assert ranker_model.booster is not None, "El modelo debe contener un booster válido."

    explainer = shap.TreeExplainer(ranker_model.booster)
    X_eval = (
        X_sample.select(feature_names).to_numpy()
        if isinstance(X_sample, pl.DataFrame)
        else X_sample
    )

    shap_values = explainer(X_eval)
    shap_values.feature_names = feature_names
    return shap_values


def get_top_shap_features(
    shap_values: shap.Explanation,
    feature_names: list[str],
    top_n: int = 10,
) -> pl.DataFrame:
    r"""Extrae el ranking de características ordenado por impacto absoluto medio $|\text{SHAP}|$.

    $$\text{Mean } |\text{SHAP}|_j = \frac{1}{N} \sum_{i=1}^N |\phi_{i,j}|$$

    Parameters
    ----------
    shap_values : shap.Explanation
        Explicación generada por compute_shap_values.
    feature_names : list[str]
        Nombres de las características.
    top_n : int, optional
        Número de características principales a retornar (por defecto 10).

    Returns
    -------
    pl.DataFrame
        DataFrame con ['rank', 'feature', 'mean_abs_shap', 'relative_pct'].
    """
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    total_impact = float(np.sum(mean_abs_shap)) or 1.0
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    ranks: list[int] = []
    features: list[str] = []
    values: list[float] = []
    percentages: list[float] = []

    for rank, idx in enumerate(sorted_indices[:top_n], 1):
        ranks.append(rank)
        features.append(feature_names[idx])
        val = float(mean_abs_shap[idx])
        values.append(val)
        percentages.append(float(val / total_impact * 100.0))

    return pl.DataFrame(
        {
            "rank": ranks,
            "feature": features,
            "mean_abs_shap": values,
            "relative_pct": percentages,
        }
    )


def plot_shap_summary(
    shap_values: shap.Explanation,
    X_sample: pl.DataFrame | np.ndarray,
    feature_names: list[str],
    output_path: Path | str = FIGURES_DIR / "fig_11_shap_summary_global.png",
    max_display: int = 15,
) -> Path:
    """Genera y exporta el gráfico de resumen global tipo beeswarm.

    Cumple con los estándares de visualización:
    - Título conclusivo sobre la dinámica empírica del modelo.
    - Exportación nítida sin solapamiento ni corte de etiquetas.

    Parameters
    ----------
    shap_values : shap.Explanation
        Explicación SHAP global calculada.
    X_sample : pl.DataFrame | np.ndarray
        Matriz de características emparejada con shap_values.
    feature_names : list[str]
        Lista de nombres de variables.
    output_path : Path | str, optional
        Ruta de destino del archivo PNG.
    max_display : int, optional
        Máximo número de variables graficadas (por defecto 15).

    Returns
    -------
    Path
        Ruta final del archivo generado.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    X_eval = (
        X_sample.select(feature_names).to_numpy()
        if isinstance(X_sample, pl.DataFrame)
        else X_sample
    )

    plt.figure(figsize=(11, 7), dpi=300)
    shap.summary_plot(
        shap_values,
        X_eval,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )

    plt.title(
        "Impacto Global SHAP en el Re-Ranking (LGBMRanker):\n"
        "Recompra Reciente (uxa_repurchase_count) y Consenso Multi-Fuente Gobiernan la Prioridad",
        fontsize=11.5,
        fontweight="bold",
        pad=15,
        color="#111111",
    )
    plt.xlabel(
        "Valor SHAP (Impacto Aditivo en la Puntuación de Ranking)", fontsize=10.5, labelpad=10
    )

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"[OK] SHAP Summary Plot guardado en: {out_file.name}")
    return out_file


def identify_customer_archetypes(
    features_df: pl.DataFrame,
) -> dict[str, tuple[pl.DataFrame, ArchetypeProfile]]:
    """Identifica 3 clientes arquetípicos representativos en la matriz de validación.

    Arquetipos:
    1. Arquetipo A: Cliente Fiel / Comprador de Básicos (alto volumen, recompras activas).
    2. Arquetipo B: Explorador Joven / Tendencias (edad < 25, digital, candidate sources R3/R7).
    3. Arquetipo C: Cliente Esporádico / Cold-Start Ligero (1 compra previa, candidate sources R4).

    Parameters
    ----------
    features_df : pl.DataFrame
        Matriz tabular de candidatos con todas las features procesadas.

    Returns
    -------
    dict[str, tuple[pl.DataFrame, ArchetypeProfile]]
        Mapeo {archetype_id: (row_df, ArchetypeProfile)}.
    """
    # Arquetipo A: Alta fidelidad histórica y reposición recurrente
    arch_a_candidates = features_df.filter(
        (pl.col("u_total_transactions") >= 8)
        & (pl.col("uxa_repurchase_count") >= 1)
        & (pl.col("uxa_bought_dept_before") == 1)
    )
    assert arch_a_candidates.height > 0, "No se encontraron candidatos para Arquetipo A"
    row_a = arch_a_candidates.sort("uxa_repurchase_count", descending=True).head(1)
    profile_a = ArchetypeProfile(
        archetype_id="A",
        customer_idx=int(row_a["customer_idx"][0]),
        article_id=int(row_a["article_id"][0]),
        persona_label="Arquetipo A: Cliente Fiel (Básicos & Recompras)",
        business_segment="Comprador Recurrente de Alto Valor",
        description=(
            "Usuario de alta fidelidad histórica con múltiples transacciones previas. "
            "La recomendación está impulsada por la frecuencia de recompra (uxa_repurchase_count) "
            "y la afinidad departamental (uxa_dept_affinity)."
        ),
    )

    # Arquetipo B: Joven, digital y buscador de novedades
    arch_b_candidates = features_df.filter(
        (pl.col("u_age") < 25)
        & (pl.col("u_online_ratio") >= 0.7)
        & ((pl.col("is_R3") == 1) | (pl.col("is_R7") == 1))
        & (pl.col("uxa_repurchase_count") == 0)
    )
    assert arch_b_candidates.height > 0, "No se encontraron candidatos para Arquetipo B"
    row_b = arch_b_candidates.sort(["n_sources", "a_sales_count"], descending=[True, True]).head(1)
    profile_b = ArchetypeProfile(
        archetype_id="B",
        customer_idx=int(row_b["customer_idx"][0]),
        article_id=int(row_b["article_id"][0]),
        persona_label="Arquetipo B: Explorador Joven (Tendencias & Digital)",
        business_segment="Generación Z Digital / Descubrimiento",
        description=(
            "Usuario joven (<25 años) con canal exclusivamente digital. Ante la ausencia de compras "
            "previas del artículo, el ranking prioriza popularidad generacional (is_R3) y novedades de moda (is_R7)."
        ),
    )

    # Arquetipo C: Esporádico / Cold-Start Ligero
    arch_c_candidates = features_df.filter(
        (pl.col("u_total_transactions") == 1)
        & (pl.col("uxa_repurchase_count") == 0)
        & (pl.col("is_R4") == 1)
        & (pl.col("customer_idx") != profile_b.customer_idx)
    )
    assert arch_c_candidates.height > 0, "No se encontraron candidatos para Arquetipo C"
    # Seleccionar artículo de canal físico o masivo distinto al de B
    row_c = (
        arch_c_candidates.filter(pl.col("article_id") != profile_b.article_id)
        .sort(["a_sales_count", "n_sources"], descending=[True, True])
        .head(1)
    )
    if row_c.height == 0:
        row_c = arch_c_candidates.sort(
            ["a_sales_count", "n_sources"], descending=[True, True]
        ).head(1)

    profile_c = ArchetypeProfile(
        archetype_id="C",
        customer_idx=int(row_c["customer_idx"][0]),
        article_id=int(row_c["article_id"][0]),
        persona_label="Arquetipo C: Cliente Esporádico (Cold-Start Ligero)",
        business_segment="Comprador Ocasional de Canal Físico",
        description=(
            "Cliente con una única compra histórica registrada. La recomendación se sustenta "
            "en la demanda del canal preferido (is_R4), el volumen global de ventas (a_sales_count) "
            "y la regularización taxonómica del catálogo."
        ),
    )

    return {
        "A": (row_a, profile_a),
        "B": (row_b, profile_b),
        "C": (row_c, profile_c),
    }


def plot_shap_waterfall_personas(
    ranker_model: LGBMRankerModel,
    archetypes: dict[str, tuple[pl.DataFrame, ArchetypeProfile]],
    feature_names: list[str],
    output_path: Path | str = FIGURES_DIR / "fig_12_shap_waterfall_personas.png",
    max_display: int = 8,
) -> Path:
    r"""Genera la figura consolidada con 3 Waterfall Plots para los arquetipos de usuario.

    Ilustra empíricamente la propiedad aditiva:
        f(x) = E[f(x)] + \sum_{i=1}^M \phi_i(x)

    Parameters
    ----------
    ranker_model : LGBMRankerModel
        Modelo LGBMRanker con booster entrenado.
    archetypes : dict[str, tuple[pl.DataFrame, ArchetypeProfile]]
        Mapeo de arquetipos generado por identify_customer_archetypes.
    feature_names : list[str]
        Nombres de las variables de modelado.
    output_path : Path | str, optional
        Ruta del archivo PNG final.
    max_display : int, optional
        Número de variables a desglosar en cada cascada (por defecto 8).

    Returns
    -------
    Path
        Ruta final de la figura generada.
    """
    assert ranker_model.booster is not None, "El modelo ranker debe contener un booster válido"
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(ranker_model.booster)

    fig, axes = plt.subplots(3, 1, figsize=(11, 15))

    keys = ["A", "B", "C"]
    for i, k in enumerate(keys):
        row_df, profile = archetypes[k]
        x_eval = row_df.select(feature_names).to_numpy()
        exp = explainer(x_eval)
        exp.feature_names = feature_names

        plt.sca(axes[i])
        shap.plots.waterfall(exp[0], max_display=max_display, show=False)

        # Título y subtítulo con conclusiones accionables
        title_text = (
            f"{profile.persona_label} | Cust #{profile.customer_idx}, Art #{profile.article_id}\n"
            f"Segmento: {profile.business_segment}"
        )
        axes[i].set_title(
            title_text, fontsize=11, fontweight="bold", pad=12, loc="left", color="#111111"
        )

    fig.suptitle(
        r"Descomposición Aditiva Local TreeSHAP: $f(x) = E[f(x)] + \sum_{i=1}^M \phi_i$ en 3 Arquetipos de Cliente",
        fontsize=13,
        fontweight="bold",
        y=0.995,
        color="#0b2545",
    )

    plt.subplots_adjust(hspace=0.45)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"[OK] Waterfall Personas Plot guardado en: {out_file.name}")
    return out_file


# Funciones Wrapper de Compatibilidad


def explain_global_shap(
    ranker_model: LGBMRankerModel,
    X_sample: pl.DataFrame | np.ndarray,
    feature_names: list[str],
    output_path: Path | str = FIGURES_DIR / "fig_11_shap_summary_global.png",
    max_display: int = 15,
) -> shap.Explanation:
    """Wrapper de compatibilidad para cálculo global y generación de gráfico."""
    shap_values = compute_shap_values(ranker_model, X_sample, feature_names)
    plot_shap_summary(shap_values, X_sample, feature_names, output_path, max_display=max_display)
    return shap_values


def explain_local_customer(
    ranker_model: LGBMRankerModel,
    customer_features: pl.DataFrame,
    feature_names: list[str],
    persona_label: str,
    output_path: Path | str,
    max_display: int = 8,
) -> None:
    """Wrapper de compatibilidad para generar un Waterfall Plot individual."""
    assert ranker_model.booster is not None, "El modelo debe contener un booster válido"
    explainer = shap.TreeExplainer(ranker_model.booster)
    X_eval = customer_features.select(feature_names).to_numpy()

    shap_values = explainer(X_eval)
    shap_values.feature_names = feature_names

    plt.figure(figsize=(9, 5), dpi=300)
    shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
    plt.title(
        f"Explicación Local SHAP: Arquetipo: {persona_label}",
        fontsize=11,
        pad=10,
        fontweight="bold",
    )
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[OK] Gráfico local ({persona_label}) guardado en: {out_file.name}")
