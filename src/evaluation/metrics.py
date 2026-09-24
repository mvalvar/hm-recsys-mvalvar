"""Módulo de evaluación de métricas de ranking para sistemas de recomendación.

Implementa la métrica oficial Mean Average Precision at 12 (MAP@12) de la competición
Kaggle H&M Personalized Fashion Recommendations, junto a métricas clave de diagnóstico:
- Recall@K: Medición del techo teórico (*recall ceiling*) del generador de candidatos.
- Hit Rate@K: Proporción de usuarios con al menos un acierto en la lista de corte.
- Catalog Coverage: Diversidad agregada de recomendaciones sobre el catálogo activo.
- Interfaz nativa Polars para evaluación directa sobre DataFrames estructurados.

Marco Matemático Oficial:
=========================
1. Average Precision at K (AP@k):
   $$AP@k = \\frac{1}{\\min(m, k)} \\sum_{i=1}^{k} P(i) \\times \\text{rel}(i)$$
   donde:
   - $m$: Número de artículos reales adquiridos por el usuario en el periodo de test ($|\\text{actual}|$).
   - $k$: Profundidad de corte de la lista recomendada ($k=12$).
   - $P(i)$: Precisión acumulada en la posición $i$, definida como:
     $$P(i) = \\frac{\\text{número de aciertos en } [1, i]}{i}$$
   - $\\text{rel}(i) \\in \\{0, 1\\}$: Indicador de relevancia (1 si el artículo predicho en rango $i$ es relevante, 0 en caso contrario).
   - Deduplicación obligatoria: Si una lista contiene predicciones repetidas, solo se evalúa la primera aparición.

2. Mean Average Precision at K (MAP@k):
   $$MAP@k = \\frac{1}{|U|} \\sum_{u=1}^{|U|} AP@k(u)$$
   donde $|U|$ representa la cardinalidad del conjunto de usuarios con compras en el periodo de evaluación.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def ap_at_k(actual: Sequence[int], predicted: Sequence[int], k: int = 12) -> float:
    r"""Calcula el Average Precision at K (AP@k) para una recomendación individual.

    $$AP@k = \frac{1}{\min(m, k)} \sum_{i=1}^{k} P(i) \times \text{rel}(i)$$

    Parameters
    ----------
    actual : Sequence[int]
        Artículos efectivamente comprados por el usuario en la semana de prueba (Ground Truth).
    predicted : Sequence[int]
        Artículos recomendados ordenados descendentemente por relevancia prevista.
    k : int, optional
        Profundidad máxima de corte (por defecto 12).

    Returns
    -------
    float
        Valor AP@k acotado en el intervalo $[0.0, 1.0]$. Retorna 0.0 si actual o predicted están vacíos.
    """
    if len(actual) == 0 or len(predicted) == 0 or k <= 0:
        return 0.0

    actual_set = set(actual)
    if not actual_set:
        return 0.0

    # Truncar estrictamente a las primeras k posiciones (sin compactar duplicados)
    preds = predicted[:k]
    score: float = 0.0
    num_hits: int = 0
    seen_hits: set[int] = set()

    for i, p in enumerate(preds):
        if p in actual_set and p not in seen_hits:
            seen_hits.add(p)
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(len(actual), k)


def map_at_k(
    actuals: dict[int, Sequence[int]],
    predictions: dict[int, Sequence[int]],
    k: int = 12,
) -> float:
    r"""Calcula la media de Average Precision (MAP@k) sobre la totalidad de usuarios evaluados.

    $$MAP@k = \frac{1}{|U|} \sum_{u=1}^{|U|} AP@k(u)$$

    Parameters
    ----------
    actuals : dict[int, Sequence[int]]
        Diccionario `{customer_idx: [artículos_reales]}`.
    predictions : dict[int, Sequence[int]]
        Diccionario `{customer_idx: [artículos_predichos]}`.
    k : int, optional
        Profundidad de corte de la lista (por defecto 12).

    Returns
    -------
    float
        Métrica global MAP@k.
    """
    if not actuals:
        return 0.0

    scores = [
        ap_at_k(actual_items, predictions.get(user_id, ()), k=k)
        for user_id, actual_items in actuals.items()
    ]
    return float(sum(scores) / len(scores))


def recall_at_k(
    actuals: dict[int, Sequence[int]],
    predictions: dict[int, Sequence[int]],
    k: int,
) -> float:
    r"""Calcula la cobertura de ítems relevantes capturados dentro del top-k (Recall@K).

    $$\text{Recall@k} = \frac{1}{|U|} \sum_{u \in U} \frac{|\text{actual}_u \cap \text{predicted}_{u, :k}|}{|\text{actual}_u|}$$

    Parameters
    ----------
    actuals : dict[int, Sequence[int]]
        Diccionario `{customer_idx: [artículos_reales]}`.
    predictions : dict[int, Sequence[int]]
        Diccionario `{customer_idx: [artículos_predichos]}`.
    k : int
        Corte del conjunto (ej. k=80 para techo teórico de candidatos o k=12 para recomendador final).

    Returns
    -------
    float
        Valor Recall@k en $[0.0, 1.0]$.
    """
    if not actuals or k <= 0:
        return 0.0

    recalls: list[float] = []
    for user_id, actual_items in actuals.items():
        if not actual_items:
            continue
        actual_set = set(actual_items)
        user_preds = predictions.get(user_id, ())
        preds_k = set(user_preds[:k])
        hits = len(actual_set.intersection(preds_k))
        recalls.append(hits / len(actual_set))

    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def hit_rate_at_k(
    actuals: dict[int, Sequence[int]],
    predictions: dict[int, Sequence[int]],
    k: int = 12,
) -> float:
    r"""Calcula la tasa de acierto binaria por usuario (Hit Rate@K).

    $$\text{HitRate@k} = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}(|\text{actual}_u \cap \text{predicted}_{u, :k}| > 0)$$

    Parameters
    ----------
    actuals : dict[int, Sequence[int]]
        Diccionario `{customer_idx: [artículos_reales]}`.
    predictions : dict[int, Sequence[int]]
        Diccionario `{customer_idx: [artículos_predichos]}`.
    k : int, optional
        Corte superior de la lista (por defecto 12).

    Returns
    -------
    float
        Porcentaje de usuarios con al menos un artículo relevante en su top-k.
    """
    if not actuals or k <= 0:
        return 0.0

    hits = 0
    for user_id, actual_items in actuals.items():
        if not actual_items:
            continue
        actual_set = set(actual_items)
        user_preds = predictions.get(user_id, ())
        preds_k = set(user_preds[:k])
        if bool(actual_set.intersection(preds_k)):
            hits += 1

    return float(hits / len(actuals))


def catalog_coverage(
    predictions: dict[int, Sequence[int]],
    total_catalog_size: int,
    k: int = 12,
) -> float:
    r"""Mide la proporción de artículos únicos recomendados respecto al total del catálogo.

    $$\text{Coverage@k} = \frac{|\bigcup_{u \in U} \text{predicted}_{u, :k}|}{\text{total\_catalog\_size}}$$

    Parameters
    ----------
    predictions : dict[int, Sequence[int]]
        Diccionario con las recomendaciones por usuario.
    total_catalog_size : int
        Número total de artículos distintos en el catálogo comercial.
    k : int, optional
        Límite de corte por usuario (por defecto 12).

    Returns
    -------
    float
        Cobertura del catálogo en $[0.0, 1.0]$.
    """
    if not predictions or total_catalog_size <= 0:
        return 0.0

    unique_recommended: set[int] = set()
    for user_preds in predictions.values():
        unique_recommended.update(user_preds[:k])

    return float(len(unique_recommended) / total_catalog_size)


def evaluate_ranking_df(
    df_eval: pl.DataFrame,
    user_col: str = "customer_idx",
    actual_col: str = "actual_articles",
    pred_col: str = "predicted_articles",
    k: int = 12,
    total_catalog_size: int | None = None,
) -> dict[str, float]:
    """Evalúa las métricas de ranking a partir de un Polars DataFrame.

    Parameters
    ----------
    df_eval : pl.DataFrame
        DataFrame conteniendo al menos la columna de identificador de usuario,
        la lista de artículos reales y la lista de artículos predichos.
    user_col : str, optional
        Nombre de la columna de usuario (default 'customer_idx').
    actual_col : str, optional
        Nombre de la columna con lista de enteros reales (default 'actual_articles').
    pred_col : str, optional
        Nombre de la columna con lista de enteros predichos (default 'predicted_articles').
    k : int, optional
        Profundidad de corte para la evaluación (default 12).
    total_catalog_size : int, optional
        Tamaño total del catálogo para medir coverage. Si es None, se deduce de los artículos observados.

    Returns
    -------
    dict[str, float]
        Diccionario con las métricas calculadas:
        {f'map@{k}', f'recall@{k}', f'hit_rate@{k}', f'coverage@{k}'}
    """
    assert isinstance(df_eval, pl.DataFrame), "df_eval debe ser un pl.DataFrame"
    assert user_col in df_eval.columns, f"Columna '{user_col}' no encontrada en df_eval"
    assert actual_col in df_eval.columns, f"Columna '{actual_col}' no encontrada en df_eval"
    assert pred_col in df_eval.columns, f"Columna '{pred_col}' no encontrada en df_eval"

    # Extracción eficiente a estructuras nativas de Python
    users = df_eval[user_col].to_list()
    actuals_list = df_eval[actual_col].to_list()
    preds_list = df_eval[pred_col].to_list()

    actuals: dict[int, list[int]] = {
        u: [int(x) for x in acts] if acts is not None else []
        for u, acts in zip(users, actuals_list, strict=False)
    }
    predictions: dict[int, list[int]] = {
        u: [int(x) for x in preds] if preds is not None else []
        for u, preds in zip(users, preds_list, strict=False)
    }

    if total_catalog_size is None:
        # Deducción de catálogo observado si no se especifica explícitamente
        catalog_set: set[int] = set()
        for acts in actuals.values():
            catalog_set.update(acts)
        for prs in predictions.values():
            catalog_set.update(prs)
        total_catalog_size = max(len(catalog_set), 1)

    return {
        f"map@{k}": map_at_k(actuals, predictions, k=k),
        f"recall@{k}": recall_at_k(actuals, predictions, k=k),
        f"hit_rate@{k}": hit_rate_at_k(actuals, predictions, k=k),
        f"coverage@{k}": catalog_coverage(predictions, total_catalog_size=total_catalog_size, k=k),
    }


# PRUEBAS BÁSICAS Y CASOS LÍMITE (EDGE CASES)
if __name__ == "__main__":
    print("=" * 70)
    print("  EJECUTANDO SUITE DE TESTS DEFENSIVOS: METRICS.PY")
    print("=" * 70)

    # Caso 1: Predicción perfecta (aciertos en primeras posiciones)
    # actual: [1, 2], pred: [1, 2, 3] -> P(1)=1/1, P(2)=2/2 -> AP = (1 + 1) / min(2, 12) = 1.0
    c1_score = ap_at_k([1, 2], [1, 2, 3], k=12)
    print(f"Test 1 [Predicción Perfecta]: AP@12 = {c1_score:.4f} (Esperado: 1.0000)")
    assert abs(c1_score - 1.0) < 1e-6, f"Fallo en Test 1: {c1_score}"

    # Caso 2: Acierto tardío
    # actual: [1], pred: [2, 1] -> P(1)=0, P(2)=1/2 -> AP = 0.5 / min(1, 12) = 0.5
    c2_score = ap_at_k([1], [2, 1], k=12)
    print(f"Test 2 [Acierto Tardío]: AP@12 = {c2_score:.4f} (Esperado: 0.5000)")
    assert abs(c2_score - 0.5) < 1e-6, f"Fallo en Test 2: {c2_score}"

    # Caso 3: Cero aciertos
    c3_score = ap_at_k([1, 2], [3, 4], k=12)
    print(f"Test 3 [Cero Aciertos]: AP@12 = {c3_score:.4f} (Esperado: 0.0000)")
    assert abs(c3_score - 0.0) < 1e-6, f"Fallo en Test 3: {c3_score}"

    # Caso 4: Manejo de duplicados en predicción y listas vacías
    # actual: [1], pred: [1, 1, 2] -> deduplicado a [1, 2] -> AP = 1.0
    c4_dup_score = ap_at_k([1], [1, 1, 2], k=12)
    print(f"Test 4a [Deduplicación]: AP@12 = {c4_dup_score:.4f} (Esperado: 1.0000)")
    assert abs(c4_dup_score - 1.0) < 1e-6, f"Fallo en Test 4a: {c4_dup_score}"

    c4_empty_act = ap_at_k([], [1, 2], k=12)
    c4_empty_pred = ap_at_k([1], [], k=12)
    print(
        f"Test 4b [Listas Vacías]: ActEmpty = {c4_empty_act:.4f}, PredEmpty = {c4_empty_pred:.4f}"
    )
    assert c4_empty_act == 0.0 and c4_empty_pred == 0.0, "Fallo en Test 4b"

    # Caso 5: Evaluación sobre DataFrame Polars
    df_toy = pl.DataFrame(
        {
            "customer_idx": [101, 102, 103],
            "actual_articles": [[1, 2], [3], [4, 5]],
            "predicted_articles": [[1, 2, 99], [99, 3], [88, 77]],
        }
    )
    metrics_res = evaluate_ranking_df(df_toy, k=12, total_catalog_size=100)
    print("\nTest 5 [Wrapper Polars]:")
    for k_metric, val in metrics_res.items():
        print(f"  * {k_metric}: {val:.4f}")

    # U101: AP=1.0, Rec=1.0, Hit=1
    # U102: AP=0.5, Rec=1.0, Hit=1
    # U103: AP=0.0, Rec=0.0, Hit=0
    # MAP: (1.0 + 0.5 + 0.0) / 3 = 0.5000
    # Recall: (1.0 + 1.0 + 0.0) / 3 = 0.6667
    # HitRate: 2 / 3 = 0.6667
    # Unique predicted: {1, 2, 99, 3, 88, 77} = 6 items -> Coverage = 6 / 100 = 0.0600
    assert abs(metrics_res["map@12"] - 0.5) < 1e-4, "Fallo en MAP@12"
    assert abs(metrics_res["recall@12"] - (2.0 / 3.0)) < 1e-4, "Fallo en Recall@12"
    assert abs(metrics_res["hit_rate@12"] - (2.0 / 3.0)) < 1e-4, "Fallo en HitRate@12"
    assert abs(metrics_res["coverage@12"] - 0.06) < 1e-4, "Fallo en Coverage@12"

    print("\n" + "=" * 70)
    print("[OK] Todos los tests unitarios y aserciones pasaron exitosamente.")
    print("=" * 70)
