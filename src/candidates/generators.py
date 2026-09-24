"""Módulo de Generación de Candidatos (Etapa de Recall).

Implementa las 8 heurísticas complementarias para maximizar la cobertura del catálogo
y servir hasta 100 candidatos de alta relevancia por usuario (N_CANDIDATES_PER_USER = 100)
antes del re-ranking supervisado con LightGBM (LambdaRank):

- R1: Recompra histórica reciente (lealtad y reposición periódica)
- R2: Popularidad global con decaimiento temporal exponencial
- R3: Popularidad segmentada por cohorte de edad (mitigación de cold start)
- R4: Popularidad discriminada por canal de venta dominante (físico/online)
- R5: Filtrado colaborativo artículo-artículo basado en co-ocurrencias (Item-CF)
- R6: Afinidad por familias de producto y departamentos habituales del usuario
- R7: Artículos en tendencia y aceleración de demanda inter-semanal
- R8: Popularidad estacional por departamento favorito del usuario

Consideraciones de rendimiento:
- Todas las operaciones son vectorizadas en Polars (cero bucles nativos Python por usuario).
- Tipado estricto: customer_idx (Int32), article_id (Int32), rank_in_source (Int16).
- Consolidación deduplicada con generación de meta-features (n_sources, best_rank, is_R1 a is_R8).
"""

from __future__ import annotations

import datetime
import gc

import polars as pl

from config.settings import (
    AGE_BIN_LABELS,
    AGE_POPULAR_TOP_K,
    CHANNEL_POPULAR_TOP_K,
    DATA_PROCESSED_DIR,
    GLOBAL_POPULAR_TOP_K,
    ITEM_CF_TOP_K,
    LAMBDA_DECAY,
    N_CANDIDATES_PER_USER,
    PRODUCT_FAMILY_TOP_K,
    REPURCHASE_TOP_K,
    TRENDING_TOP_K,
    USER_DEPT_POPULAR_TOP_K,
)


def generate_repurchase(
    transactions_df: pl.DataFrame,
    top_k: int = REPURCHASE_TOP_K,
) -> pl.DataFrame:
    r"""R1: Recompra Histórica Reciente.

    Recupera los artículos adquiridos previamente por el cliente en la ventana activa,
    ordenados jerárquicamente por la fecha de compra más reciente y el número total de recompras.

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    if "customer_idx" not in transactions_df.columns:
        raise AssertionError("Columna 'customer_idx' requerida")
    if "article_id" not in transactions_df.columns:
        raise AssertionError("Columna 'article_id' requerida")
    if "t_dat" not in transactions_df.columns:
        raise AssertionError("Columna 't_dat' requerida")
    if transactions_df.height == 0:
        raise AssertionError("transactions_df no puede estar vacío")

    repurchase_df = (
        transactions_df.group_by(["customer_idx", "article_id"])
        .agg(
            pl.len().alias("purchase_count"),
            pl.col("t_dat").max().alias("last_purchase_date"),
        )
        .sort(["last_purchase_date", "purchase_count"], descending=[True, True])
        .group_by("customer_idx")
        .head(top_k)
        .with_columns(
            pl.lit("R1_repurchase").cast(pl.String).alias("source"),
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16)
            .over("customer_idx")
            .alias("rank_in_source"),
        )
        .select(
            [
                pl.col("customer_idx").cast(pl.Int32),
                pl.col("article_id").cast(pl.Int32),
                pl.col("source"),
                pl.col("rank_in_source"),
            ]
        )
    )
    return repurchase_df


def generate_global_popularity(
    transactions_df: pl.DataFrame,
    lambda_decay: float = LAMBDA_DECAY,
    top_k: int = GLOBAL_POPULAR_TOP_K,
) -> pl.DataFrame:
    r"""R2: Popularidad Global con Decaimiento Temporal Exponencial.

    Pondera el volumen de ventas según la distancia temporal a la fecha máxima de corte:
    $$w(t) = \exp(-\lambda \cdot \Delta t_{\text{semanas}})$$

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    assert "t_dat" in transactions_df.columns, "Columna 't_dat' requerida"
    assert "article_id" in transactions_df.columns, "Columna 'article_id' requerida"
    assert "customer_idx" in transactions_df.columns, "Columna 'customer_idx' requerida"
    assert transactions_df.height > 0, "transactions_df no puede estar vacío"

    max_date = transactions_df.select(pl.col("t_dat").max()).item()

    # Cálculo vectorizado del peso con decaimiento temporal
    popular_items = (
        transactions_df.with_columns(
            ((pl.lit(max_date) - pl.col("t_dat")).dt.total_days() / 7.0).alias("weeks_ago")
        )
        .with_columns((-lambda_decay * pl.col("weeks_ago")).exp().alias("weight"))
        .group_by("article_id")
        .agg(pl.col("weight").sum().alias("score"))
        .sort("score", descending=True)
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16).alias("rank_in_source"),
            pl.lit("R2_popular_global").cast(pl.String).alias("source"),
        )
        .select(
            [
                pl.col("article_id").cast(pl.Int32),
                pl.col("source"),
                pl.col("rank_in_source"),
            ]
        )
    )

    # Replicación eficiente para todos los clientes únicos
    customers = transactions_df.select(pl.col("customer_idx").cast(pl.Int32)).unique()
    return customers.join(popular_items, how="cross").select(
        ["customer_idx", "article_id", "source", "rank_in_source"]
    )


def generate_age_group_popularity(
    transactions_df: pl.DataFrame,
    customers_df: pl.DataFrame,
    top_k: int = AGE_POPULAR_TOP_K,
) -> pl.DataFrame:
    r"""R3: Popularidad Segmentada por Cohorte de Edad (age_bin).

    Identifica los artículos más vendidos dentro del grupo etario del cliente (<25, 25-34, 35-44, 45-54, 55+).
    Provee personalización inmediata para clientes nuevos o con historial reducido (Cold Start).

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    assert "age_bin" in customers_df.columns, "Columna 'age_bin' requerida en customers_df"
    assert "customer_idx" in customers_df.columns, (
        "Columna 'customer_idx' requerida en customers_df"
    )
    assert "customer_idx" in transactions_df.columns, (
        "Columna 'customer_idx' requerida en transactions_df"
    )

    tx_with_age = transactions_df.join(
        customers_df.select(["customer_idx", "age_bin"]), on="customer_idx", how="inner"
    )

    top_per_age = (
        tx_with_age.group_by(["age_bin", "article_id"])
        .len(name="count")
        .sort(["age_bin", "count"], descending=[False, True])
        .group_by("age_bin")
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16).over("age_bin").alias("rank_in_source"),
            pl.lit("R3_popular_age").cast(pl.String).alias("source"),
        )
    )

    return (
        customers_df.select(["customer_idx", "age_bin"])
        .join(top_per_age, on="age_bin", how="inner")
        .select(
            [
                pl.col("customer_idx").cast(pl.Int32),
                pl.col("article_id").cast(pl.Int32),
                pl.col("source"),
                pl.col("rank_in_source"),
            ]
        )
    )


def generate_channel_popularity(
    transactions_df: pl.DataFrame,
    top_k: int = CHANNEL_POPULAR_TOP_K,
) -> pl.DataFrame:
    r"""R4: Popularidad por Canal Dominante (Físico vs. Digital).

    Determina el canal preferente de cada cliente (sales_channel_id: 1=Tienda física, 2=Online)
    y recomienda los productos líderes de dicho canal, garantizando alineamiento omnicanal.

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    assert "sales_channel_id" in transactions_df.columns, "Columna 'sales_channel_id' requerida"
    assert "customer_idx" in transactions_df.columns, "Columna 'customer_idx' requerida"

    user_preferred_channel = (
        transactions_df.group_by(["customer_idx", "sales_channel_id"])
        .len(name="ch_count")
        .sort(["customer_idx", "ch_count"], descending=[False, True])
        .group_by("customer_idx")
        .first()
        .select(["customer_idx", "sales_channel_id"])
    )

    top_per_channel = (
        transactions_df.group_by(["sales_channel_id", "article_id"])
        .len(name="count")
        .sort(["sales_channel_id", "count"], descending=[False, True])
        .group_by("sales_channel_id")
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16)
            .over("sales_channel_id")
            .alias("rank_in_source"),
            pl.lit("R4_popular_channel").cast(pl.String).alias("source"),
        )
    )

    return user_preferred_channel.join(top_per_channel, on="sales_channel_id", how="inner").select(
        [
            pl.col("customer_idx").cast(pl.Int32),
            pl.col("article_id").cast(pl.Int32),
            pl.col("source"),
            pl.col("rank_in_source"),
        ]
    )


def generate_item_cf(
    transactions_df: pl.DataFrame,
    window_days: int = 14,
    top_k: int = ITEM_CF_TOP_K,
) -> pl.DataFrame:
    r"""R5: Filtrado Colaborativo Ítem-Ítem (Co-ocurrencias / Co-compras).

    Calcula la frecuencia de compra conjunta de pares de artículos en una ventana reciente de 14 días.
    Para cada usuario, proyecta sus últimos artículos adquiridos hacia los ítems más frecuentemente co-comprados.

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    assert "t_dat" in transactions_df.columns, "Columna 't_dat' requerida"
    assert "customer_idx" in transactions_df.columns, "Columna 'customer_idx' requerida"
    assert "article_id" in transactions_df.columns, "Columna 'article_id' requerida"

    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    recent_tx = transactions_df.filter(
        pl.col("t_dat") >= (max_date - datetime.timedelta(days=window_days))
    )

    if recent_tx.height == 0:
        recent_tx = transactions_df

    # Para evitar explosión cuadrática en clientes con compras atípicas masivas,
    # acotamos el cálculo a un máximo de 10 transacciones recientes por cliente
    recent_bounded = (
        recent_tx.sort("t_dat", descending=True)
        .group_by("customer_idx")
        .head(10)
        .select(["customer_idx", "article_id"])
    )

    # Co-ocurrencias entre artículos adquiridos por el mismo cliente
    pairs = (
        recent_bounded.join(recent_bounded, on="customer_idx")
        .filter(pl.col("article_id") != pl.col("article_id_right"))
        .group_by(["article_id", "article_id_right"])
        .len(name="co_count")
        .sort(["article_id", "co_count"], descending=[False, True])
        .group_by("article_id")
        .head(5)
        .rename({"article_id_right": "recommended_article"})
    )

    # Últimos ítems adquiridos por cada usuario para disparar recomendaciones
    user_last_items = (
        recent_bounded.group_by("customer_idx").head(3).select(["customer_idx", "article_id"])
    )

    item_cf_candidates = (
        user_last_items.join(pairs, on="article_id", how="inner")
        .group_by(["customer_idx", "recommended_article"])
        .agg(pl.col("co_count").sum().alias("score"))
        .sort(["customer_idx", "score"], descending=[False, True])
        .group_by("customer_idx")
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16)
            .over("customer_idx")
            .alias("rank_in_source"),
            pl.lit("R5_itemcf").cast(pl.String).alias("source"),
        )
        .rename({"recommended_article": "article_id"})
        .select(
            [
                pl.col("customer_idx").cast(pl.Int32),
                pl.col("article_id").cast(pl.Int32),
                pl.col("source"),
                pl.col("rank_in_source"),
            ]
        )
    )
    return item_cf_candidates


def generate_product_family(
    transactions_df: pl.DataFrame,
    articles_df: pl.DataFrame,
    top_k: int = PRODUCT_FAMILY_TOP_K,
) -> pl.DataFrame:
    r"""R6: Afinidad por Familias de Producto y Departamentos Habituales.

    Identifica los 2 departamentos de producto (department_no) más consumidos por cada cliente
    y recomienda los artículos líderes en ventas dentro de dichas familias comerciales.

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    assert "department_no" in articles_df.columns, (
        "Columna 'department_no' requerida en articles_df"
    )
    assert "article_id" in articles_df.columns, "Columna 'article_id' requerida en articles_df"

    tx_with_dept = transactions_df.join(
        articles_df.select(["article_id", "department_no"]), on="article_id", how="inner"
    )

    # Los 2 departamentos con mayor historial para cada usuario
    user_top_dept = (
        tx_with_dept.group_by(["customer_idx", "department_no"])
        .len(name="dept_count")
        .sort(["customer_idx", "dept_count"], descending=[False, True])
        .group_by("customer_idx")
        .head(2)
        .select(["customer_idx", "department_no"])
    )

    # Top artículos por departamento comercial
    top_per_dept = (
        tx_with_dept.group_by(["department_no", "article_id"])
        .len(name="count")
        .sort(["department_no", "count"], descending=[False, True])
        .group_by("department_no")
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16)
            .over("department_no")
            .alias("rank_in_source"),
            pl.lit("R6_product_family").cast(pl.String).alias("source"),
        )
    )

    return user_top_dept.join(top_per_dept, on="department_no", how="inner").select(
        [
            pl.col("customer_idx").cast(pl.Int32),
            pl.col("article_id").cast(pl.Int32),
            pl.col("source"),
            pl.col("rank_in_source"),
        ]
    )


def generate_trending_items(
    transactions_df: pl.DataFrame,
    top_k: int = TRENDING_TOP_K,
) -> pl.DataFrame:
    r"""R7: Artículos en Tendencia y Aceleración de Demanda.

    Calcula la tasa de crecimiento relativo de ventas entre la semana reciente ($V_t$) y la previa ($V_{t-1}$):
    $$\text{velocity} = \frac{V_{t} - V_{t-1}}{V_{t-1} + 1}$$

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    assert "t_dat" in transactions_df.columns, "Columna 't_dat' requerida"
    assert "article_id" in transactions_df.columns, "Columna 'article_id' requerida"

    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    t_minus_7 = max_date - datetime.timedelta(days=7)
    t_minus_14 = max_date - datetime.timedelta(days=14)

    week_recent = (
        transactions_df.filter((pl.col("t_dat") > t_minus_7) & (pl.col("t_dat") <= max_date))
        .group_by("article_id")
        .len(name="recent_sales")
    )
    week_prior = (
        transactions_df.filter((pl.col("t_dat") > t_minus_14) & (pl.col("t_dat") <= t_minus_7))
        .group_by("article_id")
        .len(name="prior_sales")
    )

    trending = (
        week_recent.join(week_prior, on="article_id", how="left")
        .with_columns(pl.col("prior_sales").fill_null(0))
        .with_columns(
            (
                (pl.col("recent_sales") - pl.col("prior_sales")) / (pl.col("prior_sales") + 1.0)
            ).alias("velocity")
        )
        .filter(pl.col("recent_sales") >= 3)  # Soporte mínimo para filtrar ruido estadístico
        .sort("velocity", descending=True)
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16).alias("rank_in_source"),
            pl.lit("R7_trending").cast(pl.String).alias("source"),
        )
        .select(
            [
                pl.col("article_id").cast(pl.Int32),
                pl.col("source"),
                pl.col("rank_in_source"),
            ]
        )
    )

    customers = transactions_df.select(pl.col("customer_idx").cast(pl.Int32)).unique()
    return customers.join(trending, how="cross").select(
        ["customer_idx", "article_id", "source", "rank_in_source"]
    )


def generate_user_dept_popularity(
    transactions_df: pl.DataFrame,
    articles_df: pl.DataFrame,
    top_k: int = USER_DEPT_POPULAR_TOP_K,
    recent_days: int = 7,
) -> pl.DataFrame:
    r"""R8: Popularidad Estacional por Departamento Favorito del Usuario.

    Identifica el departamento dominante de cada usuario (mayor volumen de compras en la ventana)
    y recomienda los artículos más vendidos en ese departamento durante los últimos `recent_days` días,
    con decaimiento exponencial temporal para alineación estacional otoñal.

    Schema de Salida:
    -----------------
    (customer_idx: Int32, article_id: Int32, source: Utf8, rank_in_source: Int16)
    """
    assert "customer_idx" in transactions_df.columns, "Columna 'customer_idx' requerida"
    assert "article_id" in transactions_df.columns, "Columna 'article_id' requerida"
    assert "t_dat" in transactions_df.columns, "Columna 't_dat' requerida"
    assert "department_no" in articles_df.columns, (
        "Columna 'department_no' requerida en articles_df"
    )
    assert transactions_df.height > 0, "transactions_df no puede estar vacío"

    # Identificar el departamento favorito (más comprado) de cada usuario
    tx_dept = transactions_df.select(["customer_idx", "article_id"]).join(
        articles_df.select(["article_id", "department_no"]),
        on="article_id",
        how="inner",
    )

    user_dept_counts = (
        tx_dept.group_by(["customer_idx", "department_no"])
        .len(name="dept_purchases")
        .sort(["customer_idx", "dept_purchases"], descending=[False, True])
        .group_by("customer_idx")
        .first()
        .select(["customer_idx", "department_no"])
    )

    # Artículos líderes en la última semana por departamento con decaimiento exponencial
    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    cutoff_date = max_date - datetime.timedelta(days=recent_days)

    recent_tx = transactions_df.filter(pl.col("t_dat") >= cutoff_date)
    if recent_tx.height == 0:
        recent_tx = transactions_df

    recent_art_sales = (
        recent_tx.select(["article_id", "t_dat"])
        .join(articles_df.select(["article_id", "department_no"]), on="article_id", how="inner")
        .with_columns(
            ((pl.lit(max_date) - pl.col("t_dat")).dt.total_days().cast(pl.Float32)).alias(
                "days_ago"
            )
        )
        .with_columns((-0.05 * pl.col("days_ago")).exp().alias("weight"))
        .group_by(["department_no", "article_id"])
        .agg(pl.col("weight").sum().alias("dept_art_sales"))
        .sort(["department_no", "dept_art_sales"], descending=[False, True])
        .group_by("department_no")
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int16)
            .over("department_no")
            .alias("rank_in_source")
        )
        .select(["department_no", "article_id", "rank_in_source"])
    )

    # Asignación vectorizada de candidatos a cada usuario según su departamento dominante
    dept_candidates = (
        user_dept_counts.join(recent_art_sales, on="department_no", how="inner")
        .with_columns(pl.lit("R8_user_dept_popularity").cast(pl.String).alias("source"))
        .select(
            [
                pl.col("customer_idx").cast(pl.Int32),
                pl.col("article_id").cast(pl.Int32),
                pl.col("source"),
                pl.col("rank_in_source").cast(pl.Int16),
            ]
        )
    )

    return dept_candidates


def consolidate_candidates(
    *candidate_dfs: pl.DataFrame,
    max_per_user: int = N_CANDIDATES_PER_USER,
    n_partitions: int = 4,
    partition_threshold: int = 1_500_000,
) -> pl.DataFrame:
    r"""Módulo de Consolidación y Generación de Meta-Features de Candidatos.

    Fusiona los DataFrames generados por las 8 heurísticas complementarias, eliminando duplicados
    y creando las meta-características binarias y ordinales que alimentarán al modelo LGBMRanker:
    - is_R1 a is_R8: Indicadores booleanos de las heurísticas de origen.
    - n_sources: Número de fuentes independientes que recomendaron el artículo (señal de consenso).
    - best_rank: Mejor posición ordinal obtenida entre todas las fuentes.

    Ordenación Final:
    -----------------
    Ordena descendentemente por `n_sources` y ascendentemente por `best_rank`, recortando
    al límite superior `max_per_user` (por defecto 100).
    Para datasets a escala masiva (> 1.5M pares), ejecuta particionamiento estricto por rangos
    de customer_idx con recolección de basura para mantener la memoria RSS estrictamente < 2.0 GB.
    """
    assert len(candidate_dfs) > 0, "Debe proveer al menos un DataFrame de candidatos"

    # Si se pasa una lista/tupla como único argumento posicional, desempaquetar limpiamente
    if len(candidate_dfs) == 1 and isinstance(candidate_dfs[0], (list, tuple)):
        candidate_dfs = tuple(candidate_dfs[0])

    # Filtrar dataframes no vacíos
    valid_dfs = [df for df in candidate_dfs if df is not None and df.height > 0]
    assert len(valid_dfs) > 0, "Todos los DataFrames de candidatos provistos están vacíos"

    total_rows = sum(df.height for df in valid_dfs)

    if total_rows > partition_threshold:
        # Procesamiento por bloques out-of-core para control de RAM
        source_name_to_id = {
            "R1_repurchase": 1,
            "R2_popular_global": 2,
            "R3_popular_age": 3,
            "R4_popular_channel": 4,
            "R5_itemcf": 5,
            "R6_product_family": 6,
            "R7_trending": 7,
            "R8_user_dept_popularity": 8,
        }
        tagged_dfs = []
        for df in valid_dfs:
            src_str = (
                df["source"][0] if "source" in df.columns and df.height > 0 else "R1_repurchase"
            )
            sid = source_name_to_id.get(src_str, 1)
            tagged_dfs.append(
                df.select(
                    [
                        pl.col("customer_idx").cast(pl.Int32),
                        pl.col("article_id").cast(pl.Int32),
                        pl.col("rank_in_source").cast(pl.Int16),
                        pl.lit(sid, dtype=pl.Int8).alias("source_id"),
                    ]
                )
            )
        all_candidates = pl.concat(tagged_dfs)
        del tagged_dfs
        gc.collect()

        max_c = all_candidates.select(pl.col("customer_idx").max()).item()
        step = (max_c + 1) // n_partitions + 1

        part_results = []
        for p in range(n_partitions):
            c_min = p * step
            c_max = (p + 1) * step if p < n_partitions - 1 else max_c + 1
            part_df = all_candidates.filter(
                (pl.col("customer_idx") >= c_min) & (pl.col("customer_idx") < c_max)
            )
            agg_part = (
                part_df.group_by(["customer_idx", "article_id"])
                .agg(
                    pl.col("rank_in_source").min().cast(pl.Int16).alias("best_rank"),
                    pl.len().cast(pl.Int8).alias("n_sources"),
                    (pl.col("source_id") == 1).any().alias("is_R1"),
                    (pl.col("source_id") == 2).any().alias("is_R2"),
                    (pl.col("source_id") == 3).any().alias("is_R3"),
                    (pl.col("source_id") == 4).any().alias("is_R4"),
                    (pl.col("source_id") == 5).any().alias("is_R5"),
                    (pl.col("source_id") == 6).any().alias("is_R6"),
                    (pl.col("source_id") == 7).any().alias("is_R7"),
                    (pl.col("source_id") == 8).any().alias("is_R8"),
                )
                .sort(["customer_idx", "n_sources", "best_rank"], descending=[False, True, False])
                .group_by("customer_idx", maintain_order=True)
                .head(max_per_user)
                .select(
                    [
                        pl.col("customer_idx").cast(pl.Int32),
                        pl.col("article_id").cast(pl.Int32),
                        pl.col("best_rank"),
                        pl.col("n_sources"),
                        pl.col("is_R1"),
                        pl.col("is_R2"),
                        pl.col("is_R3"),
                        pl.col("is_R4"),
                        pl.col("is_R5"),
                        pl.col("is_R6"),
                        pl.col("is_R7"),
                        pl.col("is_R8"),
                    ]
                )
            )
            del part_df
            gc.collect()
            part_results.append(agg_part)

        del all_candidates
        gc.collect()
        consolidated = pl.concat(part_results)
        del part_results
        gc.collect()
    else:
        # Modo muestra rápida estándar
        all_candidates = pl.concat(valid_dfs)
        consolidated = (
            all_candidates.group_by(["customer_idx", "article_id"])
            .agg(
                pl.col("rank_in_source").min().cast(pl.Int16).alias("best_rank"),
                pl.len().cast(pl.Int8).alias("n_sources"),
                (pl.col("source") == "R1_repurchase").any().alias("is_R1"),
                (pl.col("source") == "R2_popular_global").any().alias("is_R2"),
                (pl.col("source") == "R3_popular_age").any().alias("is_R3"),
                (pl.col("source") == "R4_popular_channel").any().alias("is_R4"),
                (pl.col("source") == "R5_itemcf").any().alias("is_R5"),
                (pl.col("source") == "R6_product_family").any().alias("is_R6"),
                (pl.col("source") == "R7_trending").any().alias("is_R7"),
                (pl.col("source") == "R8_user_dept_popularity").any().alias("is_R8"),
            )
            .sort(["customer_idx", "n_sources", "best_rank"], descending=[False, True, False])
            .group_by("customer_idx", maintain_order=True)
            .head(max_per_user)
            .select(
                [
                    pl.col("customer_idx").cast(pl.Int32),
                    pl.col("article_id").cast(pl.Int32),
                    pl.col("best_rank"),
                    pl.col("n_sources"),
                    pl.col("is_R1"),
                    pl.col("is_R2"),
                    pl.col("is_R3"),
                    pl.col("is_R4"),
                    pl.col("is_R5"),
                    pl.col("is_R6"),
                    pl.col("is_R7"),
                    pl.col("is_R8"),
                ]
            )
        )

    # Validación de columnas requeridas
    assert consolidated.height > 0, "Error crítico: El DataFrame consolidado está vacío"
    assert consolidated.filter(pl.col("customer_idx").is_null()).height == 0, (
        "Nulos en customer_idx"
    )
    assert consolidated.filter(pl.col("article_id").is_null()).height == 0, "Nulos en article_id"

    return consolidated


def get_popular_fallback_items(
    transactions_df: pl.DataFrame | None = None,
    top_k: int = 12,
    days_window: int = 7,
    lambda_decay: float = 0.05,
) -> list[str]:
    r"""Retorna dinámicamente los artículos más vendidos en la última semana disponible.

    Alineación estacional y temporal:
    - Extrae transacciones en la ventana activa de los últimos `days_window` días respecto a max(t_dat).
    - Aplica decaimiento exponencial temporal: $w(t) = \exp(-\lambda \cdot \Delta t_{\text{semanas}})$.
    - Prioriza prendas de moda en aceleración de demanda (otoño 2020) erradicando cualquier lista estática de 2018.
    - Formatea de manera estricta cada identificador a cadena canónica de 10 dígitos (str.zfill(10)).

    Parameters
    ----------
    transactions_df : pl.DataFrame | None, optional
        Transacciones desde donde calcular la popularidad. Si es None, carga
        automáticamente data_processed/transactions_5w.parquet.
    top_k : int, optional
        Número de artículos a retornar (por defecto 12).
    days_window : int, optional
        Ventana de días para evaluar la estacionalidad inmediata (por defecto 7).
    lambda_decay : float, optional
        Tasa de decaimiento exponencial temporal semanal (por defecto 0.05).

    Returns
    -------
    list[str]
        Lista de top_k cadenas de 10 dígitos sin duplicados.
    """
    if transactions_df is None:
        tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
        assert tx_path.exists(), f"Archivo de transacciones no encontrado en: {tx_path}"
        transactions_df = pl.read_parquet(tx_path)

    assert transactions_df.height > 0, "transactions_df no puede estar vacío"
    assert "t_dat" in transactions_df.columns, "Columna 't_dat' requerida"
    assert "article_id" in transactions_df.columns, "Columna 'article_id' requerida"

    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    cutoff_date = max_date - datetime.timedelta(days=days_window - 1)

    # Filtrar transacciones en la ventana reciente con decaimiento temporal
    tx_recent = transactions_df.filter(pl.col("t_dat") >= cutoff_date)
    if tx_recent.height == 0:
        tx_recent = transactions_df

    weighted_sales = (
        tx_recent.with_columns(
            ((pl.lit(max_date) - pl.col("t_dat")).dt.total_days() / 7.0).alias("weeks_ago")
        )
        .with_columns((-lambda_decay * pl.col("weeks_ago")).exp().alias("weight"))
        .group_by("article_id")
        .agg(pl.col("weight").sum().alias("score"))
        .sort("score", descending=True)
        .head(top_k)
    )

    top_items = [f"{int(a):010d}" for a in weighted_sales["article_id"]]

    # Si hay menos de top_k en la ventana reciente, completar desde el historial previo
    if len(top_items) < top_k and transactions_df.height > len(top_items):
        seen = set(top_items)
        backup_sales = (
            transactions_df.group_by("article_id")
            .len()
            .sort("len", descending=True)
            .head(top_k * 3)["article_id"]
            .to_list()
        )
        for a in backup_sales:
            cand = f"{int(a):010d}"
            if cand not in seen:
                top_items.append(cand)
                seen.add(cand)
                if len(top_items) == top_k:
                    break

    return top_items[:top_k]


def get_age_group_fallback_items(
    transactions_df: pl.DataFrame | None = None,
    customers_df: pl.DataFrame | None = None,
    top_k: int = 12,
    days_window: int = 7,
    lambda_decay: float = 0.05,
) -> dict[str, list[str]]:
    r"""Genera el diccionario de vectores fallback de última semana segmentado por cohorte de edad.

    Segmentación demográfica:
    - Extrae las ventas de la última semana particionadas por age_bin ('<25', '25-34', '35-44', '45-54', '55+').
    - Si un grupo tiene menos de top_k ventas, se complementa con el vector global de última semana.
    - Incluye la clave 'GLOBAL' como referencia canónica para clientes con edad no registrada.

    Parameters
    ----------
    transactions_df : pl.DataFrame | None, optional
        Transacciones recientes. Si es None, carga data_processed/transactions_5w.parquet.
    customers_df : pl.DataFrame | None, optional
        Metadatos de clientes con columna age_bin. Si es None, carga data_processed/customers.parquet.
    top_k : int, optional
        Número de recomendaciones por grupo (por defecto 12).
    days_window : int, optional
        Ventana de días (por defecto 7).
    lambda_decay : float, optional
        Decaimiento temporal (por defecto 0.05).

    Returns
    -------
    dict[str, list[str]]
        Mapeo de cada cohorte ('<25', '25-34', etc. y 'GLOBAL') a sus 12 artículos de 10 dígitos.
    """
    if transactions_df is None:
        tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
        assert tx_path.exists(), f"Archivo no encontrado: {tx_path}"
        transactions_df = pl.read_parquet(tx_path)

    # Vector global de la última semana
    global_top = get_popular_fallback_items(
        transactions_df=transactions_df,
        top_k=top_k,
        days_window=days_window,
        lambda_decay=lambda_decay,
    )

    fallback_dict: dict[str, list[str]] = {"GLOBAL": global_top}

    # Cargar clientes con age_bin
    if customers_df is None:
        cust_path = DATA_PROCESSED_DIR / "customers.parquet"
        if cust_path.exists():
            customers_df = pl.read_parquet(cust_path)

    if customers_df is None or "age_bin" not in customers_df.columns:
        for label in AGE_BIN_LABELS:
            fallback_dict[label] = list(global_top)
        return fallback_dict

    # Filtrar transacciones de última semana y cruzar con cohorte
    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    cutoff_date = max_date - datetime.timedelta(days=days_window - 1)

    tx_recent = transactions_df.filter(pl.col("t_dat") >= cutoff_date)
    if tx_recent.height == 0:
        tx_recent = transactions_df

    tx_with_age = (
        tx_recent.join(
            customers_df.select(["customer_idx", "age_bin"]),
            on="customer_idx",
            how="inner",
        )
        .with_columns(
            ((pl.lit(max_date) - pl.col("t_dat")).dt.total_days() / 7.0).alias("weeks_ago")
        )
        .with_columns((-lambda_decay * pl.col("weeks_ago")).exp().alias("weight"))
    )

    for label in AGE_BIN_LABELS:
        cohort_tx = tx_with_age.filter(pl.col("age_bin") == label)
        if cohort_tx.height > 0:
            top_cohort = (
                cohort_tx.group_by("article_id")
                .agg(pl.col("weight").sum().alias("score"))
                .sort("score", descending=True)
                .head(top_k)
            )
            cohort_items = [f"{int(a):010d}" for a in top_cohort["article_id"]]
        else:
            cohort_items = []

        # Complementar con vector global si faltan ítems
        if len(cohort_items) < top_k:
            seen = set(cohort_items)
            for item in global_top:
                if item not in seen:
                    cohort_items.append(item)
                    seen.add(item)
                    if len(cohort_items) == top_k:
                        break

        fallback_dict[label] = cohort_items[:top_k]

    return fallback_dict
