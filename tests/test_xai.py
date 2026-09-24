from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, FIGURES_DIR, MODELS_DIR
from src.modeling.ranker import LGBMRankerModel
from src.xai.shap_analysis import (
    compute_shap_values,
    get_top_shap_features,
    identify_customer_archetypes,
)


def test_shap_computation_and_additive_property():
    model_path = MODELS_DIR / "lgbm_ranker.txt"
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"
    assert model_path.exists(), f"Modelo no encontrado: {model_path}"
    assert feat_path.exists(), f"Features no encontradas: {feat_path}"

    ranker = LGBMRankerModel.load(model_path)
    features_df = pl.read_parquet(feat_path)
    feature_names = [
        c
        for c in features_df.columns
        if c not in {"customer_idx", "article_id", "target", "source"}
    ]

    sample = features_df.head(20)
    X_eval = sample.select(feature_names).to_numpy()
    shap_exp = compute_shap_values(ranker, X_eval, feature_names)

    assert shap_exp.values.shape == (20, len(feature_names))
    assert shap_exp.base_values.shape == (20,)
    assert shap_exp.feature_names == feature_names

    raw_preds = ranker.predict(X_eval)
    for i in range(len(X_eval)):
        reconstructed = shap_exp.base_values[i] + np.sum(shap_exp.values[i])
        np.testing.assert_allclose(raw_preds[i], reconstructed, rtol=1e-5, atol=1e-5)


def test_top_shap_features_structure():
    model_path = MODELS_DIR / "lgbm_ranker.txt"
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"

    ranker = LGBMRankerModel.load(model_path)
    features_df = pl.read_parquet(feat_path)
    feature_names = [
        c
        for c in features_df.columns
        if c not in {"customer_idx", "article_id", "target", "source"}
    ]

    sample = features_df.head(50)
    X_eval = sample.select(feature_names).to_numpy()
    shap_exp = compute_shap_values(ranker, X_eval, feature_names)

    top_df = get_top_shap_features(shap_exp, feature_names, top_n=10)
    assert top_df.height == 10
    assert set(top_df.columns) == {"rank", "feature", "mean_abs_shap", "relative_pct"}
    vals = top_df["mean_abs_shap"].to_list()
    assert vals == sorted(vals, reverse=True)


def test_archetypes_identification():
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"
    features_df = pl.read_parquet(feat_path)

    archetypes = identify_customer_archetypes(features_df)
    assert set(archetypes.keys()) == {"A", "B", "C"}

    for k, (row_df, profile) in archetypes.items():
        assert row_df.height == 1
        assert profile.archetype_id == k
        assert profile.customer_idx > 0
        assert profile.article_id > 0
        assert len(profile.persona_label) > 0


def test_figures_artifacts_integrity():
    fig11 = FIGURES_DIR / "fig_cap10_01_shap_summary_global.png"
    fig12 = FIGURES_DIR / "fig_cap10_02_shap_waterfall_personas.png"

    for fig_path in [fig11, fig12]:
        assert fig_path.exists(), f"Figura {fig_path.name} no encontrada"
        size_kb = fig_path.stat().st_size / 1024.0
        assert size_kb > 50.0, f"Figura {fig_path.name} es liviana: {size_kb:.1f} KB"

        with Image.open(fig_path) as img:
            dpi = img.info.get("dpi", (0, 0))
            assert round(dpi[0]) >= 295, f"DPI insuficiente en {fig_path.name}: {dpi}"


def run_all_tests():
    print("=" * 75)
    print("  EJECUTANDO TESTS DE INTERPRETABILIDAD XAI (TEST_XAI.PY)")
    print("=" * 75)
    test_shap_computation_and_additive_property()
    print("[PASS] 1. Propiedad aditiva exacta f(x) = E[f(x)] + sum(phi) y cálculo TreeSHAP.")
    test_top_shap_features_structure()
    print("[PASS] 2. Estructura y ordenamiento de get_top_shap_features.")
    test_archetypes_identification()
    print("[PASS] 3. Segmentación e identificación de los 3 arquetipos de clientes.")
    test_figures_artifacts_integrity()
    print("[PASS] 4. Integridad de figuras oficiales (> 50 KB).")
    print("=" * 75)
    print("  [OK] TODOS LOS TESTS DE INTERPRETABILIDAD XAI SUPERADOS CON ÉXITO")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()
