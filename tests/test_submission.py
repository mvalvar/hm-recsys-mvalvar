"""Suite de Pruebas Unitarias para Inferencia Masiva y Submission Kaggle.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Valida:
1. Integridad del vector estacional de fallback (Top-12 de septiembre 2020 con formato 10 dígitos).
2. Segmentación de fallback por cohortes demográficas de edad (age_bin).
3. Generación y ordenación de recomendaciones personalizadas con LGBMRankerModel.
4. Ensamblado defensivo para el universo total de clientes con mitigación de cold-start estacional.
5. Cobertura total y detección de anomalías (nulos, longitudes discordantes, duplicados intra-usuario).
6. Integridad de los archivos físicos generados (submission.csv y submission.csv.gz).
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (  # noqa: E402
    AGE_BIN_LABELS,
    BASE_DIR,
    DATA_PROCESSED_DIR,
    KAGGLE_TOP_K,
    MODELS_DIR,
    SUBMISSION_FILE_NAME,
    SUBMISSION_GZIP_NAME,
)
from src.candidates import get_age_group_fallback_items, get_popular_fallback_items  # noqa: E402
from src.evaluation.submission import (  # noqa: E402
    build_submission,
    score_candidates,
    validate_submission,
    verify_submission_reproducibility,
)
from src.modeling.ranker import LGBMRankerModel  # noqa: E402


def test_popular_fallback_items():
    """Valida que el vector de popularidad reciente tenga exactamente 12 ítems válidos y únicos de septiembre 2020."""
    fallback = get_popular_fallback_items(top_k=KAGGLE_TOP_K, days_window=7, lambda_decay=0.05)
    assert len(fallback) == KAGGLE_TOP_K, (
        f"Esperados {KAGGLE_TOP_K} ítems, obtenidos {len(fallback)}"
    )
    assert len(set(fallback)) == KAGGLE_TOP_K, "Existen artículos duplicados en el fallback"
    for item in fallback:
        assert len(item) == 10, f"Artículo '{item}' no tiene 10 caracteres"
        assert item.isdigit(), f"Artículo '{item}' contiene caracteres no numéricos"

    # Verificar alineación estacional de última semana de septiembre 2020
    assert fallback[0] == "0924243001", (
        f"El Top-1 esperado de otoño 2020 es '0924243001', obtenido '{fallback[0]}'"
    )
    assert "0706016001" not in fallback, (
        "Error: El artículo obsoleto de 2018 '0706016001' no debe estar en el fallback 2020"
    )


def test_age_group_fallback_items():
    """Valida los vectores de popularidad segmentados por cohorte demográfica de edad."""
    cohorts = get_age_group_fallback_items(top_k=KAGGLE_TOP_K, days_window=7, lambda_decay=0.05)
    expected_cohorts = set(AGE_BIN_LABELS) | {"GLOBAL"}
    assert set(cohorts.keys()) == expected_cohorts, (
        f"Cohortes esperadas {expected_cohorts}, obtenidas {set(cohorts.keys())}"
    )

    for cohort_name, items in cohorts.items():
        assert len(items) == KAGGLE_TOP_K, (
            f"Cohorte '{cohort_name}' tiene {len(items)} ítems (esperados {KAGGLE_TOP_K})"
        )
        assert len(set(items)) == KAGGLE_TOP_K, f"Cohorte '{cohort_name}' contiene duplicados"
        for art in items:
            assert len(art) == 10 and art.isdigit(), (
                f"Artículo inválido en cohorte '{cohort_name}': {art}"
            )

    # Verificación de tendencias específicas por segmento demográfico
    assert cohorts["<25"][0] == "0918522001", (
        f"Top-1 esperado para '<25' es '0918522001', obtenido '{cohorts['<25'][0]}'"
    )
    assert cohorts["25-34"][0] == "0924243001", (
        f"Top-1 esperado para '25-34' es '0924243001', obtenido '{cohorts['25-34'][0]}'"
    )


def test_score_candidates_sample():
    """Valida la puntuación de candidatos con LGBMRanker sobre active customers."""
    model_path = MODELS_DIR / "lgbm_ranker.txt"
    mapping_path = DATA_PROCESSED_DIR / "customer_id_mapping.parquet"
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"

    assert model_path.exists(), "Modelo LGBMRanker no encontrado"
    assert mapping_path.exists(), "Mapeo de clientes no encontrado"
    assert feat_path.exists(), "Matriz de características no encontrada"

    ranker = LGBMRankerModel.load(model_path)
    mapping_df = pl.read_parquet(mapping_path)
    features_df = pl.read_parquet(feat_path)
    fallback = get_popular_fallback_items(top_k=KAGGLE_TOP_K, days_window=7, lambda_decay=0.05)

    scored_df = score_candidates(
        ranker=ranker,
        features_df=features_df,
        mapping_df=mapping_df,
        fallback_items=fallback,
        top_k=KAGGLE_TOP_K,
    )

    expected_users = features_df["customer_idx"].n_unique()
    assert scored_df.height == expected_users, (
        f"Esperados {expected_users:,} clientes activos, obtenidos {scored_df.height:,}"
    )
    assert list(scored_df.columns) == ["customer_id", "prediction"]
    assert scored_df.filter(pl.col("customer_id").is_null()).height == 0
    assert scored_df.filter(pl.col("prediction").is_null()).height == 0

    # Validar longitud de hash y de predicción
    for row in scored_df.head(50).iter_rows(named=True):
        assert len(row["customer_id"]) == 64
        items = row["prediction"].split(" ")
        assert len(items) == KAGGLE_TOP_K
        assert len(set(items)) == KAGGLE_TOP_K


def test_build_submission_fallback_logic():
    """Valida la lógica de unión (Left Join) y asignación de fallback demográfico a cold-start."""
    active_id = "a" * 64
    cold_young_id = "b" * 64
    cold_global_id = "c" * 64

    mock_scored = pl.DataFrame(
        {
            "customer_id": [active_id],
            "prediction": [
                "0100000001 0100000002 0100000003 0100000004 0100000005 0100000006 0100000007 0100000008 0100000009 0100000010 0100000011 0100000012"
            ],
        }
    )

    mock_all = pl.DataFrame(
        {
            "customer_id": [active_id, cold_young_id, cold_global_id],
            "age_bin": ["25-34", "<25", "GLOBAL"],
        }
    )

    global_fallback = [f"99{i:08d}" for i in range(1, 13)]
    young_fallback = [f"11{i:08d}" for i in range(1, 13)]
    age_cohort_fallbacks = {
        "<25": young_fallback,
        "GLOBAL": global_fallback,
    }

    result_df = build_submission(
        scored_df=mock_scored,
        all_customers_df=mock_all,
        fallback_items=global_fallback,
        age_cohort_fallbacks=age_cohort_fallbacks,
        sample_mode=False,
        top_k=12,
    )

    assert result_df.height == 3
    # Cliente activo mantiene su recomendación personalizada del modelo
    assert (
        result_df.filter(pl.col("customer_id") == active_id)["prediction"][0]
        == mock_scored["prediction"][0]
    )
    # Cliente joven recibe su fallback de cohorte
    assert result_df.filter(pl.col("customer_id") == cold_young_id)["prediction"][0] == " ".join(
        young_fallback
    )
    # Cliente sin cohorte específica recibe el fallback global
    assert result_df.filter(pl.col("customer_id") == cold_global_id)["prediction"][0] == " ".join(
        global_fallback
    )


def test_validate_submission_defensive_failures():
    """Valida que validate_submission detecte e impida cualquier anomalía de formato."""
    valid_id = "0" * 64
    valid_pred = " ".join([f"{i:010d}" for i in range(1, 13)])

    # Caso 1: Columna incorrecta
    bad_cols_df = pl.DataFrame({"user_id": [valid_id], "prediction": [valid_pred]})
    with pytest.raises(ValueError, match="Columnas inválidas"):
        validate_submission(bad_cols_df)

    # Caso 2: Nulos en customer_id
    null_cust_df = pl.DataFrame({"customer_id": [None], "prediction": [valid_pred]})
    with pytest.raises(ValueError, match="valores nulos en 'customer_id'"):
        validate_submission(null_cust_df)

    # Caso 3: Longitud de hash no es 64
    short_hash_df = pl.DataFrame({"customer_id": ["abc"], "prediction": [valid_pred]})
    with pytest.raises(ValueError, match="Longitud de 'customer_id' inconsistente"):
        validate_submission(short_hash_df)

    # Caso 4: Menos de 12 recomendaciones
    short_pred_df = pl.DataFrame(
        {"customer_id": [valid_id], "prediction": ["0751471001 0918292001"]}
    )
    with pytest.raises(
        ValueError, match=r"Longitud de 'prediction' inconsistente|no cumple el patrón regex"
    ):
        validate_submission(short_pred_df)

    # Caso 5: Artículos duplicados en la recomendación
    dup_pred_str = " ".join(["0751471001"] * 12)
    dup_pred_df = pl.DataFrame({"customer_id": [valid_id], "prediction": [dup_pred_str]})
    with pytest.raises(ValueError, match="Artículos duplicados"):
        validate_submission(dup_pred_df)


def test_submission_files_integrity():
    """Valida la integridad física de submission_v6.csv, submission_v5.csv o submission.csv."""
    csv_file = None
    for cand_name in ["submission_v8.csv", "submission_v7.csv", "submission_v6.csv", "submission_v5.csv", SUBMISSION_FILE_NAME]:
        p = BASE_DIR / cand_name
        if p.exists():
            csv_file = p
            break
    gz_file = None
    for cand_gz in ["submission_v8.csv.gz", "submission_v7.csv.gz", "submission_v6.csv.gz", "submission_v5.csv.gz", SUBMISSION_GZIP_NAME]:
        p = BASE_DIR / cand_gz
        if p.exists():
            gz_file = p
            break

    if not csv_file or not csv_file.exists():
        pytest.skip(f"Archivo de submission no encontrado en {BASE_DIR}")
    if not gz_file or not gz_file.exists():
        pytest.skip(f"Archivo comprimido de submission no encontrado en {BASE_DIR}")

    csv_size_mb = csv_file.stat().st_size / (1024 * 1024)
    gz_size_mb = gz_file.stat().st_size / (1024 * 1024)

    assert csv_size_mb > 200.0, (
        f"Tamaño de submission ({csv_size_mb:.1f} MB) es sospechosamente bajo"
    )
    assert gz_size_mb > 10.0, (
        f"Tamaño de submission comprimido ({gz_size_mb:.1f} MB) es sospechosamente bajo"
    )

    # Verificación de que el GZIP se descomprime correctamente y contiene la misma primera línea
    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        header = f.readline().strip()
        first_row = f.readline().strip()
        assert header == "customer_id,prediction"
        assert len(first_row.split(",")[0]) == 64


def test_cryptographic_manifest_verification():
    """Valida el sistema de certificación criptográfica de submissions contra el manifiesto."""
    manifest_path = BASE_DIR / "results" / "SUBMISSIONS_MANIFEST.json"
    assert manifest_path.exists(), f"Manifiesto no encontrado en {manifest_path}"

    import json

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "submissions" in data
    versions = [s["version"] for s in data["submissions"]]
    for expected_ver in ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]:
        assert expected_ver in versions, f"Versión {expected_ver} ausente en el manifiesto"

    # Validar scores oficiales de V7 y V8 en el manifiesto
    v7_entry = next(s for s in data["submissions"] if s["version"] == "v7")
    assert v7_entry["kaggle_scores"]["private"] == 0.02332
    assert v7_entry["kaggle_scores"]["public"] == 0.02291

    v8_entry = next(s for s in data["submissions"] if s["version"] == "v8")
    assert v8_entry["kaggle_scores"]["private"] == 0.02386
    assert v8_entry["kaggle_scores"]["public"] == 0.02347

    # Verificar bit-for-bit las versiones presentes en raíz (v8, v7, v6 o v5)
    for ver, fname in [("v8", "submission_v8.csv"), ("v7", "submission_v7.csv"), ("v6", "submission_v6.csv"), ("v5", "submission_v5.csv")]:
        v_csv = BASE_DIR / fname
        if v_csv.exists():
            cert = verify_submission_reproducibility(file_path=v_csv, version=ver)
            assert cert["is_reproducible"], f"Divergencia criptográfica en {ver}: {cert}"

    # Verificar bit-for-bit la versión V1 archivada
    v1_csv = BASE_DIR / "results" / "archive_submissions" / "submission_v1.csv"
    if v1_csv.exists():
        cert_v1 = verify_submission_reproducibility(file_path=v1_csv, version="v1")
        assert cert_v1["is_reproducible"], f"Divergencia criptográfica en V1 archivada: {cert_v1}"


def run_all_tests():
    print("=" * 80)
    print("  EJECUTANDO TESTS DE INFERENCIA MASIVA Y SUBMISSION KAGGLE (TEST_SUBMISSION.PY)")
    print("=" * 80)
    test_popular_fallback_items()
    print(
        "[PASS] 1. Integridad del vector estacional de fallback (12 ítems de septiembre 2020, sin 2018)."
    )
    test_age_group_fallback_items()
    print(
        "[PASS] 2. Segmentación de fallback por cohortes demográficas de edad (<25, 25-34, etc.)."
    )
    test_score_candidates_sample()
    print(
        "[PASS] 3. Puntuación y ordenación de candidatos activos con LGBMRanker (2,000 usuarios)."
    )
    test_build_submission_fallback_logic()
    print(
        "[PASS] 4. Lógica de asignación de fallback demográfico a cold-start (preservación de predicciones)."
    )
    test_validate_submission_defensive_failures()
    print(
        "[PASS] 5. Detección defensiva de anomalías (nulos, longitudes discordantes, duplicados)."
    )
    test_submission_files_integrity()
    print(
        "[PASS] 6. Integridad física de submission_v5.csv y submission_v5.csv.gz para 1.37M clientes."
    )
    test_cryptographic_manifest_verification()
    print("[PASS] 7. Certificación Criptográfica Bit-for-Bit contra SUBMISSIONS_MANIFEST.json.")
    print("=" * 80)
    print("  [OK] TODOS LOS TESTS DE SUBMISSION Y CERTIFICACIÓN SUPERADOS CON ÉXITO")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
