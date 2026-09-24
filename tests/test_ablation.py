"""Suite de Pruebas Unitarias para el Framework de Ablación (A1 a A6).

Verifica:
1. Exactitud de las ejecuciones aisladas ceteris paribus para A1 a A6.
2. Formato estructurado de los objetos AblationResult (MAP@12, Recall@12, Delta, RAM).
3. Preservación del límite presupuestario de memoria (< 2.0 GB RSS).
4. Generación correcta de la visualización de la Frontera de Pareto.
5. Persistencia en disco de los resultados consolidados (CSV).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from src.evaluation.ablation import (  # noqa: E402
    AblationResult,
    load_ablation_base_data,
    run_ablation_experiment,
    run_experiment_a1,
    run_experiment_a2,
    run_experiment_a3,
    run_experiment_a4,
    run_experiment_a5,
    run_experiment_a6,
)


def test_ablation_a1():
    """Valida la ejecución del Experimento A1 sobre horizontes reducidos [3, 5]."""
    tx_df, cust_df, art_df = load_ablation_base_data(use_sample=True, cohort_limit=500)
    results = run_experiment_a1(
        tx_df=tx_df,
        customers_df=cust_df,
        articles_df=art_df,
        windows=[3, 5],
        use_sample=True,
    )
    assert len(results) == 2, f"Se esperaban 2 resultados, obtenidos {len(results)}"
    for r in results:
        assert r.experiment_id == "A1"
        assert r.map12 >= 0.0, f"MAP@12 inválido: {r.map12}"
        assert r.recall12 >= 0.0, f"Recall@12 inválido: {r.recall12}"
        assert r.memory_mb < 2000.0, f"Violación de presupuesto RAM: {r.memory_mb} MB"
        assert r.n_candidates > 0
    print("[PASS] Experimento A1 validado exitosamente.")


def test_ablation_a2():
    """Valida la ejecución del Experimento A2 (Leave-One-Out R1..R8)."""
    tx_df, cust_df, art_df = load_ablation_base_data(use_sample=True, cohort_limit=500)
    results = run_experiment_a2(
        tx_df=tx_df,
        customers_df=cust_df,
        articles_df=art_df,
        use_sample=True,
    )
    assert len(results) == 9, (
        f"Se esperaban 9 variantes (Full + 8 fuentes), obtenidos {len(results)}"
    )
    full_res = results[0]
    assert full_res.variant == "Full (R1..R8)"
    assert full_res.delta_vs_baseline == 0.0

    for r in results:
        assert r.experiment_id == "A2"
        assert r.recall_ceiling > 0.0, f"Recall ceiling inválido: {r.recall_ceiling}"
        assert r.memory_mb < 2000.0, f"Violación de presupuesto RAM: {r.memory_mb} MB"
    print("[PASS] Experimento A2 validado exitosamente.")


def test_ablation_a3():
    """Valida la ejecución del Experimento A3 (Candidate Source Flags)."""
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"
    tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
    assert feat_path.exists() and tx_path.exists(), "Archivos base no encontrados"

    feat_df = pl.read_parquet(feat_path, n_rows=50000)
    tx_df = pl.read_parquet(tx_path).filter(pl.col("customer_idx") < 741)

    results = run_experiment_a3(
        features_df=feat_df,
        tx_df=tx_df,
        use_sample=True,
        cohort_limit=500,
    )
    assert len(results) == 2, f"Se esperaban 2 variantes, obtenidas {len(results)}"
    assert results[0].variant.startswith("Con Source Flags")
    assert results[1].variant.startswith("Sin Source Flags")
    assert results[0].map12 >= results[1].map12, (
        "El modelo con flags debe igualar o superar al modelo sin flags"
    )
    for r in results:
        assert r.memory_mb < 2000.0, f"Violación de presupuesto RAM: {r.memory_mb} MB"
    print("[PASS] Experimento A3 validado exitosamente.")


def test_ablation_a4():
    """Valida la ejecución del Experimento A4 (Negative Downsampling 1:3 y 1:5 en muestra)."""
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"
    tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
    assert feat_path.exists() and tx_path.exists(), "Archivos base no encontrados"

    feat_df = pl.read_parquet(feat_path, n_rows=50000)
    tx_df = pl.read_parquet(tx_path).filter(pl.col("customer_idx") < 741)

    results = run_experiment_a4(
        features_df=feat_df,
        tx_df=tx_df,
        ratios=[3, 5],
        use_sample=True,
        cohort_limit=500,
    )
    assert len(results) == 2, f"Se esperaban 2 variantes, obtenidas {len(results)}"
    for r in results:
        assert r.experiment_id == "A4"
        assert r.map12 >= 0.0
        assert r.memory_mb < 2000.0, f"Violación de presupuesto RAM: {r.memory_mb} MB"
    print("[PASS] Experimento A4 validado exitosamente.")


def test_ablation_a5():
    """Valida la ejecución del Experimento A5 (LambdaRank vs BCE)."""
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"
    tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
    assert feat_path.exists() and tx_path.exists(), "Archivos base no encontrados"

    feat_df = pl.read_parquet(feat_path, n_rows=50000)
    tx_df = pl.read_parquet(tx_path).filter(pl.col("customer_idx") < 741)

    results = run_experiment_a5(
        features_df=feat_df,
        tx_df=tx_df,
        use_sample=True,
        cohort_limit=500,
    )
    assert len(results) == 2, f"Se esperaban 2 variantes, obtenidas {len(results)}"
    assert results[0].variant == "LambdaRank (Listwise)"
    assert results[1].variant == "Binary Cross-Entropy (Pointwise)"
    for r in results:
        assert r.experiment_id == "A5"
        assert r.map12 >= 0.0
        assert r.memory_mb < 2000.0, f"Violación de presupuesto RAM: {r.memory_mb} MB"
    print("[PASS] Experimento A5 validado exitosamente.")


def test_ablation_a6():
    """Valida la ejecución del Experimento A6 (Frontera de Pareto dinámica y sintética)."""
    # Sin historial y allow_synthetic_baseline=False debe lanzar ValueError
    with pytest.raises(ValueError, match="ablation_history"):
        run_experiment_a6(ablation_history=[], use_sample=True, allow_synthetic_baseline=False)

    # Con allow_synthetic_baseline=True genera catálogo teórico de referencia
    results = run_experiment_a6(ablation_history=[], use_sample=True, allow_synthetic_baseline=True)
    assert len(results) == 6, (
        f"Se esperaban 6 configuraciones operacionales, obtenidas {len(results)}"
    )

    pareto_points = [r for r in results if r.details.get("is_pareto")]
    assert len(pareto_points) >= 1, "Debe haber al menos un punto en la frontera de Pareto"

    # Con historial empírico real deriva la frontera dinámicamente
    mock_history = [
        AblationResult(
            experiment_id="A1",
            variant="Ventana 3w",
            map12=0.0195,
            recall12=0.040,
            delta_vs_baseline=0.0,
            delta_pct=0.0,
            n_candidates=1000,
            recall_ceiling=0.08,
            memory_mb=350.0,
            duration_sec=0.15,
            description="Historial A1",
        ),
        AblationResult(
            experiment_id="A1",
            variant="Ventana 5w",
            map12=0.0210,
            recall12=0.048,
            delta_vs_baseline=0.0015,
            delta_pct=7.69,
            n_candidates=1000,
            recall_ceiling=0.08,
            memory_mb=450.0,
            duration_sec=0.22,
            description="Historial A1",
        ),
        AblationResult(
            experiment_id="A5",
            variant="Binary Pointwise",
            map12=0.0180,
            recall12=0.039,
            delta_vs_baseline=-0.003,
            delta_pct=-14.2,
            n_candidates=1000,
            recall_ceiling=0.08,
            memory_mb=460.0,
            duration_sec=0.25,
            description="Historial A5",
        ),
    ]
    dynamic_results = run_experiment_a6(ablation_history=mock_history, use_sample=True)
    assert len(dynamic_results) == 3
    dyn_pareto = [r for r in dynamic_results if r.details.get("is_pareto")]
    assert len(dyn_pareto) >= 1, (
        "Debe existir al menos un punto Pareto óptimo en resultados dinámicos"
    )

    fig_path = FIGURES_DIR / "fig_cap09_03_pareto_frontier_ram_map12.png"
    assert fig_path.exists(), f"El gráfico de Pareto no se generó en {fig_path}"
    print("[PASS] Experimento A6 y visualización de Pareto validados exitosamente.")


def test_ablation_dispatcher_and_persistence():
    """Valida el dispatcher general y la persistencia de ablation_results.csv."""
    results = run_ablation_experiment(
        experiment_id="A3",
        use_sample=True,
        output_dir=TABLES_DIR,
    )
    assert len(results) == 2
    out_csv = TABLES_DIR / "ablation_results.csv"
    assert out_csv.exists(), f"No se encontró el archivo exportado en {out_csv}"

    df = pl.read_csv(out_csv)
    assert df.height >= 2
    assert "map12" in df.columns
    assert "delta_vs_baseline" in df.columns
    print("[PASS] Dispatcher y persistencia CSV validados exitosamente.")


if __name__ == "__main__":
    print("=" * 70)
    print("  EJECUTANDO TESTS DEL FRAMEWORK DE ABLACIÓN (TEST_ABLATION.PY)")
    print("=" * 70)
    test_ablation_a1()
    test_ablation_a2()
    test_ablation_a3()
    test_ablation_a4()
    test_ablation_a5()
    test_ablation_a6()
    test_ablation_dispatcher_and_persistence()
    print("=" * 70)
    print("  [OK] TODOS LOS TESTS DE ABLACIÓN (A1 A A6) SUPERADOS EXITOSAMENTE")
    print("=" * 70)
