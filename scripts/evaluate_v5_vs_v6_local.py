"""Evaluación Empírica Comparativa Local: V5 vs V6 sobre Semana de Test W104.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Objetivo:
Validar de forma experimental y con rigor estadístico el salto cualitativo del motor
de recomendación V6 (Waterfall Híbrido Causal) frente a la línea base V5 (Waterfall Estratificado)
sobre el conjunto de retención temporal estricto (Semana 104: 2020-09-16 a 2020-09-22),
evaluando el Ground Truth real de los 68.984 clientes activos sin contaminación temporal.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, TABLES_DIR  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    catalog_coverage,
    hit_rate_at_k,
    map_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


def run_local_evaluation() -> dict[str, Any]:
    r"""Ejecuta la comparación local fuera de línea entre V5 y V6 sobre la semana W104.

    Garantiza:
    1. Cero fugas temporales: el histórico de características y minería causal se restringe
       estrictamente a transacciones con fecha anterior a 2020-09-16.
    2. Evaluación oficial MAP@12, Recall@12, Hit Rate@12 y Catalog Coverage sobre los
       68.984 usuarios con compras en la semana de prueba.
    3. Comparación entre V5 canónico, V6 estricto (slots 1-3, 4-7, 8-12) y V6 optimizado.

    Returns
    -------
    dict[str, Any]
        Métricas cuantitativas comparativas y telemetría de ejecución.
    """
    start_total = time.perf_counter()
    con = duckdb.connect()

    parquet_5w = DATA_PROCESSED_DIR / "transactions_5w.parquet"
    parquet_10w = DATA_PROCESSED_DIR / "transactions_10w.parquet"
    parquet_cust = DATA_PROCESSED_DIR / "customers.parquet"
    parquet_art = DATA_PROCESSED_DIR / "articles.parquet"

    assert parquet_5w.exists(), f"Falta archivo: {parquet_5w}"
    assert parquet_10w.exists(), f"Falta archivo: {parquet_10w}"
    assert parquet_cust.exists(), f"Falta archivo: {parquet_cust}"
    assert parquet_art.exists(), f"Falta archivo: {parquet_art}"

    total_catalog_items = con.execute(f"SELECT count(distinct article_id) FROM '{parquet_art}'").fetchone()[0]

    logger.info("=" * 80)
    logger.info("  EVALUACIÓN LOCAL RIGUROSA DE RANKING: V5 vs V6 (SEMANA RETENIDA W104)")
    logger.info("=" * 80)

    # Ground Truth de la semana de validación W104
    logger.info("-> Extrayendo Ground Truth de la semana 104 (2020-09-16 al 2020-09-22)...")
    gt_rows = con.execute(f"""
        SELECT customer_idx, list(distinct article_id)
        FROM '{parquet_5w}'
        WHERE t_dat >= '2020-09-16' AND t_dat <= '2020-09-22'
        GROUP BY customer_idx
    """).fetchall()
    ground_truth = {r[0]: r[1] for r in gt_rows}
    val_customers = list(ground_truth.keys())
    n_eval = len(val_customers)
    logger.info(f"   * Clientes evaluados con compras efectivas en W104: {n_eval:,}")

    # Historial de compras recientes previo al corte (2020-08-18 a 2020-09-15)
    logger.info("-> Extrayendo historial de compras previo al corte (ventana de recencia)...")
    hist_rows = con.execute(f"""
        SELECT customer_idx, list(article_id ORDER BY last_d DESC, cnt DESC, article_id ASC)
        FROM (
            SELECT customer_idx, article_id, max(t_dat) as last_d, count(*) as cnt
            FROM '{parquet_5w}'
            WHERE t_dat >= '2020-08-18' AND t_dat <= '2020-09-15'
            GROUP BY customer_idx, article_id
        )
        GROUP BY customer_idx
    """).fetchall()
    cust_hist = {r[0]: r[1] for r in hist_rows}
    logger.info(f"   * Clientes de validación con historial previo reciente: {len(cust_hist):,}")

    # Superventas contemporáneas de la semana inmediatamente anterior (W103: 2020-09-09 a 2020-09-15)
    logger.info("-> Calculando superventas de otoño contemporáneas (W103)...")
    global_bs = [
        r[0]
        for r in con.execute(f"""
        SELECT article_id, count(*) as n
        FROM '{parquet_5w}'
        WHERE t_dat >= '2020-09-09' AND t_dat <= '2020-09-15'
        GROUP BY article_id
        ORDER BY n DESC, article_id ASC
        LIMIT 12
    """).fetchall()
    ]

    age_rows = con.execute(f"""
        WITH tx AS (
            SELECT t.article_id, c.age_bin
            FROM '{parquet_5w}' t
            JOIN '{parquet_cust}' c ON t.customer_idx = c.customer_idx
            WHERE t.t_dat >= '2020-09-09' AND t.t_dat <= '2020-09-15'
        )
        SELECT age_bin, article_id, count(*) as n
        FROM tx
        GROUP BY age_bin, article_id
        QUALIFY row_number() OVER (PARTITION BY age_bin ORDER BY n DESC, article_id ASC) <= 12
        ORDER BY age_bin ASC, n DESC, article_id ASC
    """).fetchall()

    age_map: dict[str, list[int]] = {"GLOBAL": list(global_bs)}
    for ab, art, _ in age_rows:
        age_map.setdefault(ab, []).append(art)

    for ab in ["<25", "25-34", "35-44", "45-54", "55+"]:
        items = age_map.get(ab, [])
        seen = set(items)
        for b in global_bs:
            if b not in seen:
                items.append(b)
                seen.add(b)
                if len(items) == 12:
                    break
        age_map[ab] = items[:12]

    cust_age = dict(
        con.execute(
            f"SELECT customer_idx, coalesce(age_bin, 'GLOBAL') FROM '{parquet_cust}'"
        ).fetchall()
    )

    # Minería causal de co-ocurrencias en cesta P(B|A) antes de W104
    logger.info("-> Minando patrones de co-ocurrencia transaccional en cesta P(B|A)...")
    t_cooccur = time.perf_counter()
    cooccur_rows = con.execute(f"""
        WITH recent_tx AS (
            SELECT customer_idx, t_dat, article_id
            FROM '{parquet_5w}'
            WHERE t_dat >= '2020-08-18' AND t_dat <= '2020-09-15'
        ),
        pairs AS (
            SELECT b1.article_id as art_a, b2.article_id as art_b, count(*) as pair_count
            FROM recent_tx b1
            JOIN recent_tx b2 ON b1.customer_idx = b2.customer_idx AND b1.t_dat = b2.t_dat AND b1.article_id != b2.article_id
            GROUP BY b1.article_id, b2.article_id
            HAVING count(*) >= 2
        )
        SELECT art_a, art_b, pair_count
        FROM pairs
        QUALIFY row_number() OVER (PARTITION BY art_a ORDER BY pair_count DESC, art_b ASC) <= 5
    """).fetchall()

    cooccur_map: dict[int, list[int]] = {}
    for a, b, _ in cooccur_rows:
        cooccur_map.setdefault(a, []).append(b)
    logger.info(
        f"   * Matriz causal P(B|A) minada en {time.perf_counter() - t_cooccur:.2f} s: "
        f"{len(cooccur_map):,} artículos con complementos de cesta asociados"
    )

        # Simulación de predicción V5 (Waterfall Estratificado Base)
    logger.info("\n-> Generando predicciones para V5 (Waterfall Estratificado Canónico)...")
    preds_v5: dict[int, list[int]] = {}
    for u in val_customers:
        ab = cust_age.get(u, "GLOBAL")
        fb = age_map.get(ab, global_bs)
        items: list[int] = []
        seen = set()
        if u in cust_hist:
            for it in cust_hist[u]:
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                    if len(items) == 12:
                        break
        for it in fb:
            if it not in seen:
                items.append(it)
                seen.add(it)
                if len(items) == 12:
                    break
        preds_v5[u] = items

    map_v5 = map_at_k(ground_truth, preds_v5, k=12)
    rec_v5 = recall_at_k(ground_truth, preds_v5, k=12)
    hit_v5 = hit_rate_at_k(ground_truth, preds_v5, k=12)
    cov_v5 = catalog_coverage(preds_v5, total_catalog_size=total_catalog_items, k=12)

    # Simulación de predicción V6 (Waterfall Truncado: kp <= 3, kc <= 4)
    logger.info("-> Generando predicciones para V6 (Waterfall Truncado Defectuoso: kp<=3)...")
    preds_v6: dict[int, list[int]] = {}
    n_cooccur_active_v6 = 0
    for u in val_customers:
        ab = cust_age.get(u, "GLOBAL")
        fb = age_map.get(ab, global_bs)
        items = []
        seen = set()
        if u in cust_hist:
            # Recompras truncadas a 3 (defecto de V6)
            for it in cust_hist[u]:
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                    if len(items) == 3:
                        break
            # Complementos en slots 4-7
            co_count = 0
            for b_it in cust_hist[u]:
                if b_it in cooccur_map:
                    for c_it in cooccur_map[b_it]:
                        if c_it not in seen:
                            items.append(c_it)
                            seen.add(c_it)
                            co_count += 1
                            if co_count == 4 or len(items) >= 7:
                                break
                if co_count == 4 or len(items) >= 7:
                    break
            if co_count > 0:
                n_cooccur_active_v6 += 1

        for it in fb:
            if it not in seen:
                items.append(it)
                seen.add(it)
                if len(items) == 12:
                    break
        preds_v6[u] = items

    map_v6 = map_at_k(ground_truth, preds_v6, k=12)
    rec_v6 = recall_at_k(ground_truth, preds_v6, k=12)
    hit_v6 = hit_rate_at_k(ground_truth, preds_v6, k=12)
    cov_v6 = catalog_coverage(preds_v6, total_catalog_size=total_catalog_items, k=12)

    # Simulación de predicción V7 (Waterfall Híbrido Causal: kp<=12, kc<=6)
    logger.info("-> Generando predicciones para V7 (Waterfall Híbrido Causal No Destructivo)...")
    preds_v7: dict[int, list[int]] = {}
    n_cooccur_active_v7 = 0
    for u in val_customers:
        ab = cust_age.get(u, "GLOBAL")
        fb = age_map.get(ab, global_bs)
        items = []
        seen = set()
        if u in cust_hist:
            # Paso 1: Recompras personales prioritarias NO DESTRUCTIVAS (hasta 12)
            for it in cust_hist[u]:
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                    if len(items) == 12:
                        break
            # Paso 2: Complementos causales P(B|A) ÚNICAMENTE en slots libres
            if len(items) < 12:
                co_count = 0
                for b_it in cust_hist[u]:
                    if b_it in cooccur_map:
                        for c_it in cooccur_map[b_it]:
                            if c_it not in seen:
                                items.append(c_it)
                                seen.add(c_it)
                                co_count += 1
                                if co_count == 6 or len(items) >= 12:
                                    break
                    if co_count == 6 or len(items) >= 12:
                        break
                if co_count > 0:
                    n_cooccur_active_v7 += 1

        # Paso 3: Superventas de otoño si aún restan slots
        for it in fb:
            if it not in seen:
                items.append(it)
                seen.add(it)
                if len(items) == 12:
                    break
        preds_v7[u] = items

    map_v7 = map_at_k(ground_truth, preds_v7, k=12)
    rec_v7 = recall_at_k(ground_truth, preds_v7, k=12)
    hit_v7 = hit_rate_at_k(ground_truth, preds_v7, k=12)
    cov_v7 = catalog_coverage(preds_v7, total_catalog_size=total_catalog_items, k=12)

    # Componentes e Inferencia V8 (Afinidad Global Ponderada y Suavizado Multi-Semana)
    logger.info("\n-> Preparando componentes de V8 (28d personal, afinidad global ponderada, bestsellers multi-semana)...")

    # A. Bestsellers multi-semana con decaimiento óptimo gamma = 0.12
    global_bs_v8 = [
        int(r[0])
        for r in con.execute(f"""
            WITH tx AS (
                SELECT article_id,
                       datediff('day', cast(t_dat as date), date '2020-09-15') as days_ago
                FROM '{parquet_5w}'
                WHERE t_dat >= '2020-08-26' AND t_dat <= '2020-09-15'
            )
            SELECT article_id, SUM(POWER(0.12, days_ago / 7.0)) as pop_score
            FROM tx
            GROUP BY article_id
            ORDER BY pop_score DESC, article_id ASC
            LIMIT 12
        """).fetchall()
    ]
    age_rows_v8 = con.execute(f"""
        WITH tx AS (
            SELECT t.article_id, c.age,
                   datediff('day', cast(t.t_dat as date), date '2020-09-15') as days_ago
            FROM '{parquet_5w}' t
            JOIN '{parquet_cust}' c ON t.customer_idx = c.customer_idx
            WHERE t.t_dat >= '2020-08-26' AND t.t_dat <= '2020-09-15'
        ),
        binned AS (
            SELECT article_id,
                   CASE
                       WHEN age < 25 THEN '<25'
                       WHEN age <= 34 THEN '25-34'
                       WHEN age <= 44 THEN '35-44'
                       WHEN age <= 54 THEN '45-54'
                       WHEN age IS NOT NULL THEN '55+'
                       ELSE 'GLOBAL'
                   END as age_bin,
                   days_ago
            FROM tx
        )
        SELECT age_bin, article_id, SUM(POWER(0.12, days_ago / 7.0)) as pop_score
        FROM binned
        GROUP BY age_bin, article_id
        QUALIFY row_number() OVER (PARTITION BY age_bin ORDER BY pop_score DESC, article_id ASC) <= 12
        ORDER BY age_bin ASC, pop_score DESC, article_id ASC
    """).fetchall()
    age_map_v8: dict[str, list[int]] = {"GLOBAL": list(global_bs_v8)}
    for ab, art, _ in age_rows_v8:
        age_map_v8.setdefault(ab, []).append(int(art))
    for ab in ['<25', '25-34', '35-44', '45-54', '55+']:
        items_ab = age_map_v8.get(ab, [])
        seen_ab = set(items_ab)
        for b in global_bs_v8:
            if b not in seen_ab:
                items_ab.append(b)
                seen_ab.add(b)
                if len(items_ab) == 12:
                    break
        age_map_v8[ab] = items_ab[:12]

    # B. Matriz de Co-ocurrencia P(B|A) ponderada por recencia transaccional (soporte >= 3)
    cooccur_rows_v8 = con.execute(f"""
        WITH recent_tx AS (
            SELECT customer_idx, t_dat, article_id,
                   datediff('day', cast(t_dat as date), date '2020-09-15') as days_ago
            FROM '{parquet_5w}'
            WHERE t_dat >= '2020-08-12' AND t_dat <= '2020-09-15'
        ),
        pairs AS (
            SELECT b1.article_id as art_a, b2.article_id as art_b,
                   SUM(POWER(0.85, (b1.days_ago + b2.days_ago) / 14.0)) as pair_weight,
                   count(*) as pair_count
            FROM recent_tx b1
            JOIN recent_tx b2 ON b1.customer_idx = b2.customer_idx AND b1.t_dat = b2.t_dat AND b1.article_id != b2.article_id
            GROUP BY b1.article_id, b2.article_id
            HAVING count(*) >= 3
        )
        SELECT art_a, art_b, pair_weight
        FROM pairs
        QUALIFY row_number() OVER (PARTITION BY art_a ORDER BY pair_weight DESC, art_b ASC) <= 8
    """).fetchall()
    cooccur_dict_v8: dict[int, list[tuple[int, float]]] = {}
    for a, b, pw in cooccur_rows_v8:
        cooccur_dict_v8.setdefault(int(a), []).append((int(b), float(pw)))

    # C. Compras personales en la ventana óptima de 28 días con days_ago
    p_rows_v8 = con.execute(f"""
        SELECT customer_idx,
               list(article_id ORDER BY last_d DESC, cnt DESC, article_id ASC),
               list(days_ago ORDER BY last_d DESC, cnt DESC, article_id ASC)
        FROM (
            SELECT customer_idx, article_id,
                   MAX(t_dat) as last_d, count(*) as cnt,
                   datediff('day', cast(MAX(t_dat) as date), date '2020-09-15') as days_ago
            FROM '{parquet_5w}'
            WHERE t_dat >= '2020-08-19' AND t_dat <= '2020-09-15'
            GROUP BY customer_idx, article_id
        )
        GROUP BY customer_idx
    """).fetchall()
    p_v8 = {r[0]: r[1] for r in p_rows_v8}
    p_days_v8 = {r[0]: r[2] for r in p_rows_v8}

    # D. Inferencia V8
    logger.info("-> Generando predicciones para V8 (Afinidad Global Ponderada y Suavizado Multi-Semana)...")
    preds_v8: dict[int, list[int]] = {}
    n_cooccur_active_v8 = 0
    for u in val_customers:
        ab = cust_age.get(u, "GLOBAL")
        fb_v8 = age_map_v8.get(ab, global_bs_v8)
        items = []
        seen = set()

        if u in p_v8:
            base_items = p_v8[u]
            base_days = p_days_v8[u]
            # Paso 1: Personal no destructivo (hasta 12)
            for it in base_items:
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                    if len(items) == 12:
                        break

            # Paso 2: Afinidad Global Ponderada P(B|A) en huecos libres
            if len(items) < 12:
                comp_scores: dict[int, float] = {}
                for b_it, d_ago in zip(base_items[:5], base_days[:5], strict=False):
                    w_base = 0.80 ** (d_ago / 7.0)
                    if b_it in cooccur_dict_v8:
                        for c_it, pw in cooccur_dict_v8[b_it]:
                            if c_it not in seen:
                                comp_scores[c_it] = comp_scores.get(c_it, 0.0) + w_base * pw
                if comp_scores:
                    sorted_comps = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)
                    co_count = 0
                    for c_it, _ in sorted_comps[:6]:
                        if c_it not in seen:
                            items.append(c_it)
                            seen.add(c_it)
                            co_count += 1
                            if len(items) >= 12:
                                break
                    if co_count > 0:
                        n_cooccur_active_v8 += 1

        # Paso 3: Superventas multi-semana por cohorte
        if len(items) < 12:
            for it in fb_v8:
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                    if len(items) == 12:
                        break
        # Paso 4: Relleno global
        if len(items) < 12:
            for it in global_bs_v8:
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                    if len(items) == 12:
                        break

        preds_v8[u] = items

    map_v8 = map_at_k(ground_truth, preds_v8, k=12)
    rec_v8 = recall_at_k(ground_truth, preds_v8, k=12)
    hit_v8 = hit_rate_at_k(ground_truth, preds_v8, k=12)
    cov_v8 = catalog_coverage(preds_v8, total_catalog_size=total_catalog_items, k=12)

    delta_v8_vs_v7 = map_v8 - map_v7
    delta_v8_vs_v7_pct = (delta_v8_vs_v7 / map_v7) * 100.0

    delta_v8_vs_v5 = map_v8 - map_v5
    delta_v8_vs_v5_pct = (delta_v8_vs_v5 / map_v5) * 100.0

    duration = time.perf_counter() - start_total

    # Reporte comparativo de 4 vías
    logger.info("\n" + "=" * 105)
    logger.info("  RESULTADOS COMPARATIVOS OFICIALES DE VALIDACIÓN LOCAL (W104: V5 vs V6 vs V7 vs V8)")
    logger.info("=" * 105)
    logger.info(f"{'Métrica':<18} | {'V5 (Canónico)':<15} | {'V6 (Truncado)':<15} | {'V7 (No Destruct)':<17} | {'V8 (Afinidad Global)':<22} | {'Delta V8 vs V7':<15}")
    logger.info("-" * 105)
    logger.info(f"{'MAP@12':<18} | {map_v5:<15.5f} | {map_v6:<15.5f} | {map_v7:<17.5f} | {map_v8:<22.5f} | {delta_v8_vs_v7_pct:+.2f}% ({delta_v8_vs_v5_pct:+.2f}% vs V5)")
    logger.info(f"{'Recall@12':<18} | {rec_v5:<15.5f} | {rec_v6:<15.5f} | {rec_v7:<17.5f} | {rec_v8:<22.5f} | {((rec_v8-rec_v7)/rec_v7)*100:+.2f}%")
    logger.info(f"{'Hit Rate@12':<18} | {hit_v5:<15.5f} | {hit_v6:<15.5f} | {hit_v7:<17.5f} | {hit_v8:<22.5f} | {((hit_v8-hit_v7)/hit_v7)*100:+.2f}%")
    logger.info(f"{'Catalog Coverage':<18} | {cov_v5:<15.5f} | {cov_v6:<15.5f} | {cov_v7:<17.5f} | {cov_v8:<22.5f} | {((cov_v8-cov_v7)/cov_v7)*100:+.2f}%")
    logger.info("-" * 105)
    logger.info(f"Clientes evaluados en W104 : {n_eval:,}")
    logger.info(f"Tiempo total de ejecución  : {duration:.2f} segundos")
    logger.info("=" * 105 + "\n")

    results = {
        "evaluation_cohort": "W104 (2020-09-16 a 2020-09-22)",
        "customers_evaluated": n_eval,
        "active_customers_with_history": len(cust_hist),
        "v5_metrics": {
            "map12": round(map_v5, 5),
            "recall12": round(rec_v5, 5),
            "hit_rate12": round(hit_v5, 5),
            "catalog_coverage": round(cov_v5, 5),
        },
        "v6_metrics": {
            "map12": round(map_v6, 5),
            "recall12": round(rec_v6, 5),
            "hit_rate12": round(hit_v6, 5),
            "catalog_coverage": round(cov_v6, 5),
        },
        "v7_metrics": {
            "map12": round(map_v7, 5),
            "recall12": round(rec_v7, 5),
            "hit_rate12": round(hit_v7, 5),
            "catalog_coverage": round(cov_v7, 5),
        },
        "v8_metrics": {
            "map12": round(map_v8, 5),
            "recall12": round(rec_v8, 5),
            "hit_rate12": round(hit_v8, 5),
            "catalog_coverage": round(cov_v8, 5),
        },
        "delta_v8_vs_v7": {
            "map12_absolute": round(delta_v8_vs_v7, 5),
            "map12_percent": round(delta_v8_vs_v7_pct, 2),
            "recall12_absolute": round(rec_v8 - rec_v7, 5),
            "hit_rate12_absolute": round(hit_v8 - hit_v7, 5),
        },
        "delta_v8_vs_v5": {
            "map12_absolute": round(delta_v8_vs_v5, 5),
            "map12_percent": round(delta_v8_vs_v5_pct, 2),
            "recall12_absolute": round(rec_v8 - rec_v5, 5),
            "hit_rate12_absolute": round(hit_v8 - hit_v5, 5),
        },
        "execution_duration_sec": round(duration, 2),
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_file_4way = TABLES_DIR / "eval_v5_v6_v7_v8_local.json"
    with open(out_file_4way, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[GUARDADO] Resultados comparativos de 4 vías exportados a: {out_file_4way}")

    out_file_v7 = TABLES_DIR / "eval_v5_v6_v7_local.json"
    with open(out_file_v7, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_local_evaluation()
