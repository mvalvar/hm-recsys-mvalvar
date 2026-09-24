"""Suite de Pruebas Unitarias para el Esquema de Validación Temporal Estricto.

Valida:
1. Separación disjunta de fechas y garantía formal anti-leakage (max(train) < min(val)).
2. Construcción correcta de diccionarios de Ground Truth (activos vs. universo total).
3. Asignación vectorizada de etiquetas target (y=1 vs y=0) y porcentaje de positivos.
4. Coherencia matemática estricta de grupos (sum(groups) == df.height).
5. Determinismo del muestreo negativo estratificado por cliente.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, RANDOM_SEED  # noqa: E402
from src.utils.validation import (  # noqa: E402
    RollingTemporalSplit,
    assert_zero_feature_target_leakage,
    assert_zero_temporal_leakage,
    build_ground_truth_dict,
    create_rolling_temporal_split,
    prepare_ranker_split,
    split_transactions_temporal,
)


def load_test_data():
    """Carga los DataFrames optimizados de prueba desde data_processed/."""
    tx = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    feat = pl.read_parquet(DATA_PROCESSED_DIR / "features_matrix.parquet")
    return tx, feat


def test_temporal_split_and_leakage_immunity():
    """Valida la partición temporal estricta y la ausencia de look-ahead bias."""
    tx, _ = load_test_data()

    # Split con val_days=7 por defecto
    split_res = split_transactions_temporal(tx, val_days=7)
    train_df = split_res.train_df
    val_df = split_res.val_df

    assert train_df.height > 0
    assert val_df.height > 0
    assert train_df.height + val_df.height == tx.height

    # Fechas
    train_max = train_df.select(pl.col("t_dat").max()).item()
    val_min = val_df.select(pl.col("t_dat").min()).item()
    assert train_max < val_min, f"Violación temporal: {train_max} >= {val_min}"
    assert train_max == split_res.val_start_date - datetime.timedelta(days=1)
    assert val_min == split_res.val_start_date

    # Verificación de assert_zero_temporal_leakage
    assert assert_zero_temporal_leakage(train_df, val_df, split_res.val_start_date)

    # Comprobación defensiva: Crear una fuga deliberada debe levantar AssertionError
    leaked_train = pl.concat([train_df, val_df.head(5)])
    with pytest.raises(AssertionError, match=r"LOOK-AHEAD BIAS DETECTADO"):
        assert_zero_temporal_leakage(leaked_train, val_df, split_res.val_start_date)


def test_ground_truth_dict_construction():
    """Valida la extracción estructurada del Ground Truth local de prueba."""
    tx, _ = load_test_data()
    split_res = split_transactions_temporal(tx, val_days=7)
    val_df = split_res.val_df

    # Modo active_only=True
    gt_active = build_ground_truth_dict(val_df, active_only=True)
    val_users = val_df["customer_idx"].unique().to_list()
    assert len(gt_active) == len(val_users)
    for u in val_users[:20]:
        assert u in gt_active
        items = gt_active[u]
        assert len(items) > 0
        # Sin duplicados
        assert len(items) == len(set(items))

    # Modo active_only=False con clientes inactivos
    sample_universe = val_users[:10] + [9999998, 9999999]
    gt_all = build_ground_truth_dict(val_df, customer_indices=sample_universe, active_only=False)
    assert 9999998 in gt_all and gt_all[9999998] == []
    assert 9999999 in gt_all and gt_all[9999999] == []


def test_prepare_ranker_split_group_coherence_and_targets():
    """Valida la asignación de target, grupos de consulta y downsampling para LightGBM."""
    tx, feat = load_test_data()
    split_res = split_transactions_temporal(tx, val_days=7)
    val_df = split_res.val_df

    # Sin muestreo negativo (Modo Evaluación / Validación completa)
    eval_split = prepare_ranker_split(
        feature_matrix=feat,
        ground_truth_df=val_df,
        negative_ratio=None,
        drop_empty_queries=False,
    )
    assert eval_split.df.height == feat.height
    assert sum(eval_split.groups) == feat.height
    assert len(eval_split.groups) == feat["customer_idx"].n_unique()
    assert all(g > 0 for g in eval_split.groups)

    # Validar contigüidad estricta de customer_idx
    cust_series = eval_split.customer_ids.to_list()
    seen_customers = set()
    current_cust = None
    for c in cust_series:
        if c != current_cust:
            assert c not in seen_customers, (
                f"customer_idx {c} no es contiguo en el DataFrame de ranking!"
            )
            seen_customers.add(c)
            current_cust = c

    # Con muestreo negativo 1:5 (Modo Entrenamiento)
    train_split = prepare_ranker_split(
        feature_matrix=feat,
        ground_truth_df=val_df,
        negative_ratio=5,
        drop_empty_queries=False,
        seed=RANDOM_SEED,
    )
    assert train_split.df.height <= feat.height
    assert sum(train_split.groups) == train_split.df.height
    assert len(train_split.groups) == train_split.customer_ids.n_unique()
    assert all(g > 0 for g in train_split.groups)

    # Porcentaje de positivos
    pos_count = (train_split.y == 1).sum()
    pos_ratio = pos_count / train_split.df.height
    assert pos_ratio > 0.005, f"Ratio de positivos ({pos_ratio:.4f}) menor al umbral de 0.5%"

    # Determinismo con seed fijo
    train_split_2 = prepare_ranker_split(
        feature_matrix=feat,
        ground_truth_df=val_df,
        negative_ratio=5,
        drop_empty_queries=False,
        seed=RANDOM_SEED,
    )
    assert train_split.df.equals(train_split_2.df), (
        "El muestreo negativo no es determinista con la misma semilla!"
    )


def test_assert_zero_feature_target_leakage_cases():
    """Valida la auditoría formal contra fuga entre features y target por timestamps."""
    # Caso válido con fechas dinámicas
    base_date = datetime.date(2025, 6, 1)
    feat_tx = pl.DataFrame(
        {
            "customer_idx": [1, 2],
            "article_id": [10, 20],
            "t_dat": [base_date, base_date + datetime.timedelta(days=5)],
        }
    )
    target_tx = pl.DataFrame(
        {
            "customer_idx": [1, 3],
            "article_id": [10, 30],
            "t_dat": [
                base_date + datetime.timedelta(days=6),
                base_date + datetime.timedelta(days=12),
            ],
        }
    )
    assert assert_zero_feature_target_leakage(feat_tx, target_tx, step_name="test_valid") is True

    # Caso con contaminación: feat_max >= tgt_min
    leaked_feat = pl.concat(
        [
            feat_tx,
            pl.DataFrame(
                {
                    "customer_idx": [4],
                    "article_id": [40],
                    "t_dat": [base_date + datetime.timedelta(days=6)],  # Mismo día que min(target)
                }
            ),
        ]
    )
    with pytest.raises(AssertionError, match=r"LOOK-AHEAD LEAKAGE DETECTADO"):
        assert_zero_feature_target_leakage(leaked_feat, target_tx, step_name="test_leak")

    # Caso con target previo a features
    reversed_target = pl.DataFrame(
        {
            "customer_idx": [1],
            "article_id": [10],
            "t_dat": [base_date - datetime.timedelta(days=1)],
        }
    )
    with pytest.raises(AssertionError, match=r"LOOK-AHEAD LEAKAGE DETECTADO"):
        assert_zero_feature_target_leakage(feat_tx, reversed_target, step_name="test_reversed")

    # Validaciones defensivas de esquema y DataFrames vacíos
    empty_df = pl.DataFrame({"customer_idx": [], "article_id": [], "t_dat": []}).cast(
        {"t_dat": pl.Date}
    )
    with pytest.raises(ValueError, match=r"feature_tx no puede estar vacío"):
        assert_zero_feature_target_leakage(empty_df, target_tx)
    with pytest.raises(ValueError, match=r"target_tx no puede estar vacío"):
        assert_zero_feature_target_leakage(feat_tx, empty_df)

    no_tdat = pl.DataFrame({"customer_idx": [1], "article_id": [10]})
    with pytest.raises(ValueError, match=r"Columna 't_dat' obligatoria"):
        assert_zero_feature_target_leakage(no_tdat, target_tx)


def test_create_rolling_temporal_split_dynamic():
    """Valida la generación de ventanas deslizantes desacopladas sin fechas fijas."""
    # Crear transacciones sintéticas sobre un horizonte arbitrario no relacionado con 2020
    start_date = datetime.date(2024, 3, 1)
    n_days = 28
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_days)]
    synth_tx = pl.DataFrame(
        {
            "customer_idx": [i % 10 for i in range(len(dates) * 3)],
            "article_id": [100 + (i % 20) for i in range(len(dates) * 3)],
            "t_dat": [dates[i // 3] for i in range(len(dates) * 3)],
        }
    )

    val_days = 7
    train_target_days = 7
    split = create_rolling_temporal_split(
        synth_tx, val_days=val_days, train_target_days=train_target_days
    )
    assert isinstance(split, RollingTemporalSplit)

    max_synth = start_date + datetime.timedelta(days=n_days - 1)  # 2024-03-28

    # Verificación de fechas calculadas dinámicamente
    assert split.val_target_end == max_synth
    expected_val_target_start = max_synth - datetime.timedelta(days=val_days - 1)
    assert split.val_target_start == expected_val_target_start
    assert split.val_features_max == expected_val_target_start - datetime.timedelta(days=1)

    assert split.train_target_end == split.val_features_max
    expected_train_target_start = split.train_target_end - datetime.timedelta(
        days=train_target_days - 1
    )
    assert split.train_target_start == expected_train_target_start
    assert split.train_features_max == expected_train_target_start - datetime.timedelta(days=1)

    # Verificación de particiones de DataFrames
    assert split.val_target_tx.height > 0
    assert split.val_features_tx.height > 0
    assert split.train_target_tx.height > 0
    assert split.train_features_tx.height > 0

    # Separación temporal absoluta (cero solapamiento entre features y su target respectivo)
    val_feat_max = split.val_features_tx.select(pl.col("t_dat").max()).item()
    val_tgt_min = split.val_target_tx.select(pl.col("t_dat").min()).item()
    assert val_feat_max < val_tgt_min
    assert val_feat_max == split.val_features_max
    assert val_tgt_min == split.val_target_start

    train_feat_max = split.train_features_tx.select(pl.col("t_dat").max()).item()
    train_tgt_min = split.train_target_tx.select(pl.col("t_dat").min()).item()
    assert train_feat_max < train_tgt_min
    assert train_feat_max == split.train_features_max
    assert train_tgt_min == split.train_target_start

    # Desacoplamiento entre train target y val target
    train_tgt_max = split.train_target_tx.select(pl.col("t_dat").max()).item()
    assert train_tgt_max < val_tgt_min

    # Soporte de history_days
    split_hist = create_rolling_temporal_split(
        synth_tx,
        val_days=val_days,
        train_target_days=train_target_days,
        history_days=7,
    )
    val_feat_min = split_hist.val_features_tx.select(pl.col("t_dat").min()).item()
    expected_val_feat_min = split.val_features_max - datetime.timedelta(days=6)
    assert val_feat_min == expected_val_feat_min


def test_create_rolling_temporal_split_on_processed_data():
    """Valida la partición rolling sobre transacciones reales de data_processed/."""
    tx, _ = load_test_data()
    split = create_rolling_temporal_split(tx, val_days=7, train_target_days=7)

    assert split.train_features_tx.height > 0
    assert split.train_target_tx.height > 0
    assert split.val_features_tx.height > 0
    assert split.val_target_tx.height > 0

    # Total de transacciones sumadas debe coincidir con la partición temporal
    assert assert_zero_feature_target_leakage(split.train_features_tx, split.train_target_tx)
    assert assert_zero_feature_target_leakage(split.val_features_tx, split.val_target_tx)


def run_all_tests():
    print("=" * 70)
    print("  EJECUTANDO TESTS DE VALIDACIÓN TEMPORAL (TEST_VALIDATION.PY)")
    print("=" * 70)

    test_temporal_split_and_leakage_immunity()
    print("[PASS] 1. Partición temporal estricta y prueba defensiva anti-leakage.")

    test_ground_truth_dict_construction()
    print("[PASS] 2. Construcción del diccionario Ground Truth (activos e inactivos).")

    test_prepare_ranker_split_group_coherence_and_targets()
    print("[PASS] 3. Asignación de target, coherencia matemática de grupos y downsampling.")

    test_assert_zero_feature_target_leakage_cases()
    print("[PASS] 4. Auditoría matemática formal contra fuga features-target por timestamps.")

    test_create_rolling_temporal_split_dynamic()
    print("[PASS] 5. Partición desacoplada rolling-window dinámica y agnóstica de fechas fijas.")

    test_create_rolling_temporal_split_on_processed_data()
    print("[PASS] 6. Partición desacoplada rolling-window sobre transacciones procesadas.")

    print("=" * 70)
    print("  [OK] TODOS LOS TESTS DE VALIDACIÓN TEMPORAL SUPERADOS EXITOSAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
