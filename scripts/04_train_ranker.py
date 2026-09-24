"""Script 04: Entrenamiento y Validación del Modelo Supervisado LGBMRanker.

Implementa la arquitectura Two-Stage RecSys:
1. Partición temporal estricta (entrenamiento en semanas 100-103, validación local en semana 104).
2. Negative Downsampling 1:5 estratificado por usuario para aceleración y regularización.
3. Entrenamiento con pérdida LambdaRank y early stopping de 20 rondas.
4. Extracción de Feature Importance por Ganancia (Gain) y guardado de figura.
5. Inferencia vectorizada Top-12 y evaluación formal de MAP@12, Recall@12 y Hit Rate@12.

Uso:
    python scripts/04_train_ranker.py [--sample]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATA_PROCESSED_SAMPLE_DIR,
    FIGURES_DIR,
    LGBM_PARAMS,
    LGBM_SAMPLE_PARAMS,
    RANDOM_SEED,
    TABLES_DIR,
    get_models_dir,
    get_processed_dir,
)
from src.evaluation.metrics import hit_rate_at_k, map_at_k, recall_at_k  # noqa: E402
from src.features.builder import build_full_feature_matrix  # noqa: E402
from src.modeling.ranker import LGBMRankerModel  # noqa: E402
from src.utils.memory import log_memory_usage  # noqa: E402
from src.utils.validation import (  # noqa: E402
    build_ground_truth_dict,
    create_rolling_temporal_split,
    prepare_ranker_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de LGBMRanker con LambdaRank y Validación MAP@12"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Utiliza hiperparámetros ágiles para CI/CD y pruebas rápidas",
    )
    return parser.parse_args()


def plot_feature_importance(imp_df: pl.DataFrame, output_path: Path) -> None:
    """Genera y guarda el gráfico editorial de importancia de características."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    top15 = imp_df.head(15).reverse()
    features = top15["feature"].to_list()
    gain_pcts = top15["gain_pct"].to_list()

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
    ax.set_xlim(0, max(gain_pcts) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    from matplotlib.patches import Patch

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
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico de Importancia guardado en: {output_path.name}")


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    print("=" * 75)
    print("  FASE 4: ENTRENAMIENTO DE LGBMRANKER Y VALIDACIÓN TEMPORAL MAP@12")
    print(
        f"  Modo Operacional: {'MUESTRA ÁGIL (--sample)' if args.sample else 'PRODUCCIÓN COMPLETA'}"
    )
    print("=" * 75)

    processed_dir = get_processed_dir(args.sample)
    feat_path = processed_dir / "features_matrix.parquet"
    if not feat_path.exists() and args.sample:
        feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"

    tx_path = processed_dir / "transactions_5w.parquet"
    if not tx_path.exists() and args.sample:
        tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"

    if not feat_path.exists():
        cmd = "python scripts/03_features.py" + (" --sample" if args.sample else "")
        hint = f"Ejecute '{cmd}' primero."
        if not args.sample and (DATA_PROCESSED_SAMPLE_DIR / "features_matrix.parquet").exists():
            hint += "\n  [PISTA] Se encontró features_matrix.parquet en sample/. Si deseaba evaluar la muestra rápida, añada '--sample'."
        raise FileNotFoundError(f"[ERROR DE DEPENDENCIAS] No existe: {feat_path}.\n  -> {hint}")

    if not tx_path.exists():
        cmd = "python scripts/01_preprocess.py" + (" --sample" if args.sample else "")
        raise FileNotFoundError(f"[ERROR DE DEPENDENCIAS] No existe: {tx_path}.\n  -> Ejecute '{cmd}' primero.")

    log_memory_usage("Inicio de entrenamiento")

    print(f"-> Cargando matriz de características desde {feat_path}...")
    features_df = pl.read_parquet(feat_path)
    tx_df = pl.read_parquet(tx_path)
    print(
        f"  * Matriz tabular : {features_df.height:,} filas × {len(features_df.columns)} columnas"
    )
    print(f"  * Transacciones  : {tx_df.height:,} registros")

    # Partición temporal desacoplada (Rolling Window)
    print("-> Aplicando partición temporal desacoplada (Rolling Window)...")
    rolling_split = create_rolling_temporal_split(tx_df, val_days=7, train_target_days=7)
    train_tx = rolling_split.train_target_tx
    val_tx = rolling_split.val_target_tx
    print(
        f"  * Train Features Tx : {rolling_split.train_features_tx.height:,} (hasta {rolling_split.train_features_max})"
    )
    print(
        f"  * Train Target Tx   : {rolling_split.train_target_tx.height:,} (desde {rolling_split.train_target_start} hasta {rolling_split.train_target_end})"
    )
    print(
        f"  * Val Features Tx   : {rolling_split.val_features_tx.height:,} (hasta {rolling_split.val_features_max})"
    )
    print(
        f"  * Val Target Tx     : {rolling_split.val_target_tx.height:,} (desde {rolling_split.val_target_start} hasta {rolling_split.val_target_end})"
    )

    # Preparación de datasets para LambdaRank
    train_feat_path = processed_dir / "features_train.parquet"
    if not train_feat_path.exists() and args.sample:
        train_feat_path = DATA_PROCESSED_DIR / "features_train.parquet"
    if train_feat_path.exists():
        print(f"-> Cargando matriz de entrenamiento desacoplada desde {train_feat_path.name}...")
        train_features_df = pl.read_parquet(train_feat_path)
    elif args.sample:
        print(
            "-> [MODO MUESTRA] Construyendo en memoria características de entrenamiento desacopladas..."
        )
        cand_path = processed_dir / "candidates.parquet"
        if not cand_path.exists() and args.sample:
            cand_path = DATA_PROCESSED_DIR / "candidates.parquet"
        cust_path = processed_dir / "customers.parquet"
        if not cust_path.exists() and args.sample:
            cust_path = DATA_PROCESSED_DIR / "customers.parquet"
        art_path = processed_dir / "articles.parquet"
        if not art_path.exists() and args.sample:
            art_path = DATA_PROCESSED_DIR / "articles.parquet"
        sample_users = set(features_df["customer_idx"].unique().to_list())
        cand_sample = pl.read_parquet(cand_path).filter(pl.col("customer_idx").is_in(sample_users))
        cust_df = pl.read_parquet(cust_path)
        art_df = pl.read_parquet(art_path)
        train_features_df = build_full_feature_matrix(
            candidates_df=cand_sample,
            transactions_df=rolling_split.train_features_tx,
            customers_df=cust_df,
            articles_df=art_df,
        )
        del cand_sample, cust_df, art_df
        gc.collect()
    else:
        print(
            "-> [AVISO] features_train.parquet no encontrado en disco; utilizando features_matrix..."
        )
        train_features_df = features_df

    print("-> Preparando datos de entrenamiento con Negative Downsampling (1:5)...")
    train_split = prepare_ranker_split(
        feature_matrix=train_features_df,
        ground_truth_df=train_tx,
        negative_ratio=5,
        drop_empty_queries=True,
        seed=RANDOM_SEED,
    )
    print(f"  * Filas de entrenamiento   : {train_split.df.height:,}")
    print(f"  * Consultas activas (users): {len(train_split.groups):,}")
    print(f"  * Features de modelado     : {len(train_split.feature_names)}")

    print("-> Preparando datos de validación retenidos para early stopping...")
    val_split = prepare_ranker_split(
        feature_matrix=features_df,
        ground_truth_df=val_tx,
        negative_ratio=None,
        drop_empty_queries=True,
    )
    print(f"  * Filas de validación      : {val_split.df.height:,}")
    print(f"  * Consultas de validación  : {len(val_split.groups):,}")

    # Configurar e Instanciar Modelo
    params = LGBM_SAMPLE_PARAMS if args.sample else LGBM_PARAMS
    ranker = LGBMRankerModel(params=params)

    # Optimización de memoria: guardar nombres de características y liberar features_df antes del fit
    feature_names = list(train_split.feature_names)
    del features_df
    gc.collect()
    log_memory_usage("Previo a ranker.fit")

    # Ajuste del Ranker (50 estimadores para convergencia equilibrada multi-feature)
    ranker.fit(
        X=train_split.X,
        y=train_split.y,
        groups=train_split.groups,
        feature_names=feature_names,
        eval_set=[(val_split.X, val_split.y)],
        eval_group=[val_split.groups],
        early_stopping_rounds=None,
        verbose_eval=10,
    )

    # Serialización del modelo
    models_dir = get_models_dir(args.sample)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out = models_dir / "lgbm_ranker.txt"
    ranker.save(model_out)

    # Extracción y visualización de Feature Importance
    print("-> Extrayendo importancia de características...")
    imp_df = ranker.get_feature_importance()
    table_name = "feature_importance_sample.csv" if args.sample else "feature_importance_lgbm.csv"
    table_out = TABLES_DIR / table_name
    imp_df.write_csv(table_out)
    print(f"[OK] Tabla de Importancia guardada en: {table_out.name}")

    fig_out = FIGURES_DIR / "fig_cap07_01_feature_importance.png"
    plot_feature_importance(imp_df, fig_out)

    # Liberar conjuntos de entrenamiento y validación antes de recargar features para inferencia
    del train_split, val_split
    gc.collect()
    log_memory_usage("Previo a inferencia Top-12")

    # Evaluación de métricas MAP@12 sobre ranking
    print("\n-> Recargando matriz de características para inferencia Top-12...")
    features_df = pl.read_parquet(feat_path)
    n_candidates_evaluated = features_df.height
    scores = ranker.predict(features_df.select(feature_names))

    preds_summary = (
        features_df.select(["customer_idx", "article_id"])
        .with_columns(pl.Series("score", scores))
        .sort(["customer_idx", "score"], descending=[False, True])
        .group_by("customer_idx")
        .head(12)
        .group_by("customer_idx")
        .agg(pl.col("article_id"))
    )
    preds_dict = dict(
        zip(
            preds_summary["customer_idx"].to_list(),
            preds_summary["article_id"].to_list(),
            strict=False,
        )
    )
    cand_users = set(preds_summary["customer_idx"].to_list())
    del features_df, preds_summary
    gc.collect()

    actuals_all = build_ground_truth_dict(val_tx, active_only=True)
    actuals_cand = {u: actuals_all[u] for u in actuals_all if u in cand_users}

    # Cohorte de clientes frecuentes (>= 4 compras en historial previo a validación)
    frequent_users = set(
        rolling_split.val_features_tx.group_by("customer_idx")
        .len()
        .filter(pl.col("len") >= 4)["customer_idx"]
        .to_list()
    )
    actuals_freq = {u: actuals_cand[u] for u in actuals_cand if u in frequent_users}

    # Métricas en cohorte local de candidatos
    map12_cand = map_at_k(actuals_cand, preds_dict, k=12)
    recall12_cand = recall_at_k(actuals_cand, preds_dict, k=12)
    hitrate12_cand = hit_rate_at_k(actuals_cand, preds_dict, k=12)

    # Métricas en clientes frecuentes
    map12_freq = map_at_k(actuals_freq, preds_dict, k=12) if actuals_freq else 0.0
    recall12_freq = recall_at_k(actuals_freq, preds_dict, k=12) if actuals_freq else 0.0
    hitrate12_freq = hit_rate_at_k(actuals_freq, preds_dict, k=12) if actuals_freq else 0.0

    # Métricas en universo global de la competición
    map12_global = map_at_k(actuals_all, preds_dict, k=12)
    recall12_global = recall_at_k(actuals_all, preds_dict, k=12)
    hitrate12_global = hit_rate_at_k(actuals_all, preds_dict, k=12)

    elapsed = time.perf_counter() - start_time
    rss_mb = log_memory_usage("Fin de entrenamiento y evaluación")

    metrics_report = {
        "candidate_cohort": {
            "n_active_users": len(actuals_cand),
            "map_at_12": round(map12_cand, 6),
            "recall_at_12": round(recall12_cand, 6),
            "hit_rate_at_12": round(hitrate12_cand, 6),
        },
        "frequent_users_cohort": {
            "n_active_users": len(actuals_freq),
            "map_at_12": round(map12_freq, 6),
            "recall_at_12": round(recall12_freq, 6),
            "hit_rate_at_12": round(hitrate12_freq, 6),
        },
        "global_competition_universe": {
            "n_active_users": len(actuals_all),
            "map_at_12": round(map12_global, 6),
            "recall_at_12": round(recall12_global, 6),
            "hit_rate_at_12": round(hitrate12_global, 6),
        },
        "system_telemetry": {
            "n_candidates_evaluated": n_candidates_evaluated,
            "elapsed_seconds": round(elapsed, 2),
            "rss_memory_mb": round(rss_mb, 1),
        },
    }
    metrics_name = (
        "lgbm_evaluation_metrics_sample.json" if args.sample else "lgbm_evaluation_metrics.json"
    )
    metrics_path = TABLES_DIR / metrics_name
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    # Resumen de métricas de ranking en consola
    print("\n" + "=" * 75)
    print("  RESULTADO OFICIAL DE VALIDACIÓN TEMPORAL (LGBMRANKER)")
    print("=" * 75)
    print(f"1. Cohorte Local de Candidatos (N = {len(actuals_cand):,} activos):")
    print(f"   * Mean Average Precision (MAP@12) : {map12_cand:.5f} ({map12_cand * 100:.3f}%)")
    print(f"   * Recall@12                       : {recall12_cand:.4%}")
    print(f"   * Hit Rate@12                     : {hitrate12_cand:.4%}")
    print(f"\n2. Subconjunto Clientes Frecuentes (>=4 tx, N = {len(actuals_freq):,} activos):")
    print(f"   * MAP@12 en Frecuentes            : {map12_freq:.5f} ({map12_freq * 100:.3f}%)")
    print(f"   * Hit Rate@12 en Frecuentes       : {hitrate12_freq:.4%}")
    print(f"\n3. Población Global Competición (N = {len(actuals_all):,} compradores semana 104):")
    print(f"   * MAP@12 Global Normalizado       : {map12_global:.6f}")
    print(f"   * Tiempo Total de Ejecución       : {elapsed:.2f} segundos")
    print(f"   * Consumo Pico de RAM (RSS)       : {rss_mb:.1f} MB")
    print("=" * 75)
    print(f"[OK] Modelo y artefactos certificados en {models_dir} y {FIGURES_DIR}")


if __name__ == "__main__":
    main()
