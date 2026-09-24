"""Suite de Pruebas Unitarias para el Modelo LGBMRanker.

Valida:
1. Instanciación, entrenamiento y early stopping de LGBMRankerModel.
2. Generación y orden de predicciones de ranking.
3. Extracción de importancia de características (Gain y Split) con suma porcentual 100%.
4. Serialización (.txt nativo) y deserialización sin pérdida de feature_names.
5. Inferencia Top-12 y cálculo de MAP@12.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, LGBM_SAMPLE_PARAMS, MODELS_DIR  # noqa: E402
from src.evaluation.metrics import map_at_k  # noqa: E402
from src.modeling.ranker import LGBMRankerModel  # noqa: E402
from src.utils.validation import (  # noqa: E402
    build_ground_truth_dict,
    create_rolling_temporal_split,
    prepare_ranker_split,
    split_transactions_temporal,
)


def test_lgbmranker_fit_predict_and_importance():
    """Prueba de ajuste, inferencia y extracción de importancia."""
    tx = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    feat = pl.read_parquet(DATA_PROCESSED_DIR / "features_matrix.parquet")

    split = split_transactions_temporal(tx, val_days=7)
    train_split = prepare_ranker_split(
        feat, split.train_df, negative_ratio=5, drop_empty_queries=True
    )
    val_split = prepare_ranker_split(
        feat, split.val_df, negative_ratio=None, drop_empty_queries=True
    )

    ranker = LGBMRankerModel(params=LGBM_SAMPLE_PARAMS)
    ranker.fit(
        X=train_split.X,
        y=train_split.y,
        groups=train_split.groups,
        feature_names=train_split.feature_names,
        eval_set=[(val_split.X, val_split.y)],
        eval_group=[val_split.groups],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    scores = ranker.predict(val_split.X)
    assert len(scores) == val_split.X.height
    assert not np.isnan(scores).any()

    imp_df = ranker.get_feature_importance()
    assert imp_df.height == len(train_split.feature_names)
    assert set(imp_df.columns) == {"feature", "gain_importance", "gain_pct", "split_importance"}
    # La suma de porcentajes debe ser aproximadamente 100%
    assert abs(imp_df["gain_pct"].sum() - 100.0) < 1e-2

    # Serialización y carga
    test_model_path = MODELS_DIR / "test_ranker_tmp.txt"
    ranker.save(test_model_path)
    assert test_model_path.exists()

    loaded = LGBMRankerModel.load(test_model_path)
    assert loaded.feature_names == ranker.feature_names
    scores_loaded = loaded.predict(val_split.X)
    np.testing.assert_allclose(scores, scores_loaded, rtol=1e-5)

    # Limpieza
    if test_model_path.exists():
        test_model_path.unlink()
    meta_path = test_model_path.with_name(f"{test_model_path.stem}_meta.json")
    if meta_path.exists():
        meta_path.unlink()


def test_top12_evaluation_workflow():
    """Valida la canalización de inferencia Top-12 y métrica MAP@12."""
    tx = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")

    split = split_transactions_temporal(tx, val_days=7)
    val_tx = split.val_df

    actuals = build_ground_truth_dict(val_tx, active_only=True)
    assert len(actuals) > 0

    # Simular predicciones
    cust_list = list(actuals.keys())[:10]
    preds = {c: actuals[c][:12] for c in cust_list}

    # Para clientes con predicción perfecta, MAP debe ser 1.0
    perfect_map = map_at_k({c: actuals[c] for c in cust_list}, preds, k=12)
    assert perfect_map == 1.0


def test_lgbmranker_fit_with_rolling_temporal_split():
    """Valida el ajuste del Ranker utilizando partición rolling desacoplada."""
    tx = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    feat = pl.read_parquet(DATA_PROCESSED_DIR / "features_matrix.parquet").head(5000)

    rolling = create_rolling_temporal_split(tx, val_days=7, train_target_days=7)

    train_split = prepare_ranker_split(
        feat, rolling.train_target_tx, negative_ratio=3, drop_empty_queries=True
    )
    val_split = prepare_ranker_split(
        feat, rolling.val_target_tx, negative_ratio=None, drop_empty_queries=True
    )

    if train_split.df.height > 0 and val_split.df.height > 0:
        ranker = LGBMRankerModel(params=LGBM_SAMPLE_PARAMS)
        ranker.fit(
            X=train_split.X,
            y=train_split.y,
            groups=train_split.groups,
            feature_names=train_split.feature_names,
            eval_set=[(val_split.X, val_split.y)],
            eval_group=[val_split.groups],
            early_stopping_rounds=5,
            verbose_eval=False,
        )
        preds = ranker.predict(val_split.X)
        assert len(preds) == val_split.X.height
        assert not np.isnan(preds).any()


def run_all_tests():
    print("=" * 70)
    print("  EJECUTANDO TESTS DE MODELO LGBMRANKER (TEST_RANKER.PY)")
    print("=" * 70)

    test_lgbmranker_fit_predict_and_importance()
    print("[PASS] 1. Fit, early stopping, predicción, feature importance y guardado.")

    test_top12_evaluation_workflow()
    print("[PASS] 2. Inferencia Top-12 y cálculo de MAP@12.")

    test_lgbmranker_fit_with_rolling_temporal_split()
    print("[PASS] 3. Ajuste del Ranker con partición rolling desacoplada.")

    print("=" * 70)
    print("  [OK] TODOS LOS TESTS DE MODELIZACIÓN SUPERADOS EXITOSAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
