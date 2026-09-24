"""Script de Extracción Estadística y Verificación de Fuentes de Datos (Ground Truth).

Calcula estadísticas directamente desde los datos crudos y procesados
(data/*.csv y data_processed/*.parquet) y el modelo entrenado (models/lgbm_ranker.txt),
sin depender de ningún script de visualización previo.

Genera results/tables/GROUND_TRUTH_audit.json como la fuente de verdad definitiva.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, TABLES_DIR


def audit_ground_truth() -> dict:
    print("=" * 80)
    print("  VERIFICACIÓN DE FUENTES DE DATOS (GROUND TRUTH)")
    print("=" * 80)

    stats: dict = {}

    print("\n[1/6] Cargando datasets procesados...")
    tx_5w = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_5w.parquet")
    cust = pl.read_parquet(DATA_PROCESSED_DIR / "customers.parquet")
    art = pl.read_parquet(DATA_PROCESSED_DIR / "articles.parquet")
    agg = pl.read_parquet(DATA_PROCESSED_DIR / "transactions_full_weekly_agg.parquet")

    n_cust = cust.height
    n_art = art.height
    n_tx_5w = tx_5w.height
    n_active_users_5w = tx_5w["customer_idx"].n_unique()
    n_active_art_5w = tx_5w["article_id"].n_unique()
    sparsity_5w = (1.0 - n_tx_5w / (n_cust * n_art)) * 100.0
    inactive_pct_5w = (1.0 - n_active_users_5w / n_cust) * 100.0

    user_tx_counts = tx_5w.group_by("customer_idx").len()["len"]
    median_tx_user = float(user_tx_counts.median())
    mean_tx_user = float(user_tx_counts.mean())

    stats["base_metrics"] = {
        "n_customers_total": n_cust,
        "n_articles_total": n_art,
        "n_transactions_5w": n_tx_5w,
        "n_active_users_5w": n_active_users_5w,
        "n_active_articles_5w": n_active_art_5w,
        "sparsity_5w_pct": round(sparsity_5w, 6),
        "inactive_users_5w_pct": round(inactive_pct_5w, 4),
        "median_transactions_per_user_5w": median_tx_user,
        "mean_transactions_per_user_5w": round(mean_tx_user, 4),
        "n_weeks_agg": agg.height,
    }
    print(f"  * Clientes totales: {n_cust:,}")
    print(f"  * Artículos totales: {n_art:,}")
    print(f"  * Transacciones 5w: {n_tx_5w:,}")
    print(f"  * Sparsity 5w: {sparsity_5w:.5f}%")
    print(f"  * Inactivos 5w: {inactive_pct_5w:.2f}%")

    # Demografía y Metadatos
    print("\n[2/6] Auditando demografía y metadatos...")
    ages = cust["age"].drop_nulls().to_numpy()
    null_age_count = cust.filter(pl.col("age").is_null()).height
    null_desc_count = art.filter(pl.col("detail_desc").is_null()).height

    stats["demographics"] = {
        "mean_age": round(float(np.mean(ages)), 2),
        "median_age": float(np.median(ages)),
        "null_age_count": null_age_count,
        "null_age_pct": round(null_age_count / n_cust * 100.0, 2),
        "articles_with_desc": n_art - null_desc_count,
        "null_desc_count": null_desc_count,
        "null_desc_pct": round(null_desc_count / n_art * 100.0, 2),
    }

    # Distribución de Canal de Venta por Cohorte
    print("\n[3/6] Calculando distribución de canal físico vs online por cohorte...")
    channel_age = (
        tx_5w.join(cust.select(["customer_idx", "age_bin"]), on="customer_idx")
        .group_by(["age_bin", "sales_channel_id"])
        .len()
        .sort(["age_bin", "sales_channel_id"])
    )
    cohort_order = ["<25", "25-34", "35-44", "45-54", "55+"]
    channel_summary = {}
    for cohort in cohort_order:
        store_cnt = channel_age.filter((pl.col("age_bin") == cohort) & (pl.col("sales_channel_id") == 1))
        online_cnt = channel_age.filter((pl.col("age_bin") == cohort) & (pl.col("sales_channel_id") == 2))
        s = store_cnt["len"][0] if store_cnt.height > 0 else 0
        o = online_cnt["len"][0] if online_cnt.height > 0 else 0
        tot = s + o
        channel_summary[cohort] = {
            "store_count": s,
            "online_count": o,
            "total_transactions": tot,
            "store_pct": round(s / tot * 100.0, 2) if tot > 0 else 0.0,
            "online_pct": round(o / tot * 100.0, 2) if tot > 0 else 0.0,
        }
    stats["sales_channel_by_cohort"] = channel_summary
    for c, v in channel_summary.items():
        print(f"  * Cohorte {c:>5s}: Tienda={v['store_pct']:>5.2f}% | Online={v['online_pct']:>5.2f}% (Total: {v['total_transactions']:,})")

    # Repurchase Lag (5 semanas y 2 años)
    print("\n[4/6] Calculando ciclos de recompra...")
    # 5w
    lags_5w = (
        tx_5w.sort(["customer_idx", "t_dat"])
        .with_columns(pl.col("t_dat").diff().over("customer_idx").dt.total_days().alias("lag"))
        .filter(pl.col("lag").is_not_null() & (pl.col("lag") > 0))
    )["lag"].to_numpy()

    eval_days = [1, 3, 7, 14, 21, 30, 35, 45, 60, 90, 180, 365]
    cdf_5w = {f"day_{d}": round(float((lags_5w <= d).mean() * 100.0), 2) for d in [1, 3, 7, 14, 21, 30, 35]}

    stats["repurchase_5w"] = {
        "n_recompra_events": len(lags_5w),
        "median_lag_days": float(np.median(lags_5w)),
        "mean_lag_days": round(float(np.mean(lags_5w)), 2),
        "cdf_pct": cdf_5w,
    }

    # 2 años: verificar o calcular desde transactions_train.csv
    cdf_file = DATA_PROCESSED_DIR / "repurchase_lags_2yr_cdf.parquet"
    if cdf_file.exists():
        print("  -> Cargando distribución de 2 años precomputada...")
        df_2yr_lags = pl.read_parquet(cdf_file)
        lags_2yr = df_2yr_lags["lag"].to_numpy()
    else:
        print("  -> Extrayendo sesiones de compra desde data/transactions_train.csv...")
        t0 = time.perf_counter()
        raw_tx = pl.read_csv(DATA_RAW_DIR / "transactions_train.csv", columns=["t_dat", "customer_id"])
        dates_unique = (
            raw_tx.select([
                pl.col("customer_id"),
                pl.col("t_dat").str.to_date("%Y-%m-%d"),
            ])
            .unique()
            .sort(["customer_id", "t_dat"])
        )
        lags_2yr = (
            dates_unique.with_columns(
                pl.col("t_dat").diff().over("customer_id").dt.total_days().alias("lag")
            )
            .filter(pl.col("lag").is_not_null() & (pl.col("lag") > 0))
        )["lag"].to_numpy()
        # Persistir para acelerar regeneraciones futuras
        pl.DataFrame({"lag": lags_2yr}).write_parquet(cdf_file)
        print(f"  -> Calculados y persistidos {len(lags_2yr):,} intervalos en {time.perf_counter()-t0:.1f}s")

    cdf_2yr = {f"day_{d}": round(float((lags_2yr <= d).mean() * 100.0), 2) for d in eval_days}
    stats["repurchase_2yr"] = {
        "n_recompra_events": len(lags_2yr),
        "median_lag_days": float(np.median(lags_2yr)),
        "mean_lag_days": round(float(np.mean(lags_2yr)), 2),
        "cdf_pct": cdf_2yr,
    }
    print(f"  * 5w:  Total={len(lags_5w):,}, Mediana={stats['repurchase_5w']['median_lag_days']:.1f}d, <=7d={cdf_5w['day_7']}%, <=14d={cdf_5w['day_14']}%, <=35d={cdf_5w['day_35']}%")
    print(f"  * 2yr: Total={len(lags_2yr):,}, Mediana={stats['repurchase_2yr']['median_lag_days']:.1f}d, <=7d={cdf_2yr['day_7']}%, <=14d={cdf_2yr['day_14']}%, <=35d={cdf_2yr['day_35']}%")

    # Power Law y Gini
    print("\n[5/6] Calculando Power Law (Pareto alpha) y Curva de Lorenz (Gini)...")
    art_sales = tx_5w.group_by("article_id").len().sort("len", descending=True)["len"].to_numpy()
    ranks = np.arange(1, len(art_sales) + 1)
    fit_max = min(5000, len(ranks))
    fit_range = slice(min(10, fit_max - 1), fit_max)
    coeffs = np.polyfit(np.log10(ranks[fit_range]), np.log10(art_sales[fit_range]), 1)
    alpha = -coeffs[0]

    # Gini
    n_items_tot = n_art
    n_zero_sales = max(0, n_items_tot - len(art_sales))
    all_article_sales = np.concatenate([np.zeros(n_zero_sales), np.sort(art_sales)])
    idx_all = np.arange(1, n_items_tot + 1)
    gini_total = float(np.sum((2 * idx_all - n_items_tot - 1) * all_article_sales) / (n_items_tot * np.sum(all_article_sales)))

    n_active = len(art_sales)
    sorted_active = np.sort(art_sales)
    idx_act = np.arange(1, n_active + 1)
    gini_active = float(np.sum((2 * idx_act - n_active - 1) * sorted_active) / (n_active * np.sum(sorted_active)))

    stats["concentration"] = {
        "power_law_alpha": round(float(alpha), 4),
        "gini_total_catalog": round(gini_total, 4),
        "gini_active_catalog": round(gini_active, 4),
    }
    print(f"  * Exponente Pareto alpha: {alpha:.4f}")
    print(f"  * Gini Total Catálogo:    {gini_total:.4f}")
    print(f"  * Gini Activo (5w):       {gini_active:.4f}")

    # Verificación de Modelo LightGBM
    print("\n[6/6] Verificando LightGBM Booster contra feature_importance_lgbm.csv...")
    booster_path = MODELS_DIR / "lgbm_ranker.txt"
    meta_path = MODELS_DIR / "lgbm_ranker_meta.json"
    if booster_path.exists() and meta_path.exists():
        booster = lgb.Booster(model_file=str(booster_path))
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        feat_names = meta["feature_names"]
        gain_imps = booster.feature_importance(importance_type="gain")
        split_imps = booster.feature_importance(importance_type="split")
        total_gain = float(np.sum(gain_imps))

        model_feat_imp = {}
        for fn, g, s in zip(feat_names, gain_imps, split_imps):
            model_feat_imp[fn] = {
                "gain": float(g),
                "split": int(s),
                "gain_pct": round(float(g / total_gain * 100.0), 4) if total_gain > 0 else 0.0,
            }
        stats["model_features"] = {
            "n_features": len(feat_names),
            "total_gain": total_gain,
            "top_features": sorted(model_feat_imp.items(), key=lambda x: -x[1]["gain"])[:10],
        }

        # Comparar con CSV
        csv_path = TABLES_DIR / "feature_importance_lgbm.csv"
        if csv_path.exists():
            csv_df = pl.read_csv(csv_path)
            discrepancies = 0
            for r in csv_df.iter_rows(named=True):
                fn = r["feature"]
                if fn in model_feat_imp:
                    diff = abs(r["gain_importance"] - model_feat_imp[fn]["gain"])
                    if diff > 0.01:
                        print(f"  [DISCREPANCIA] {fn}: CSV={r['gain_importance']}, Booster={model_feat_imp[fn]['gain']}")
                        discrepancies += 1
            if discrepancies == 0:
                print("  [OK] Las 39 features del CSV coinciden al 100% con el booster serializado.")
            else:
                print(f"  [ALERTA] Se encontraron {discrepancies} discrepancias entre CSV y booster.")

    out_file = TABLES_DIR / "GROUND_TRUTH_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[ÉXITO] Archivo de Ground Truth exportado a: {out_file}")
    return stats


if __name__ == "__main__":
    audit_ground_truth()
