"""Constructor de Características (39 Features en 4 Namespaces) para Re-Ranking Supervisado.

Diseñado bajo arquitectura out-of-core de Polars:
- Tipado estricto en memoria: Int8, Int16, Int32, Float32 (cero Float64/Int64).
- 4 Namespaces matemáticamente disjuntos:
    Namespace 1: Características del Usuario (9 features)
    Namespace 2: Características del Artículo (9 features)
    Namespace 3: Interacción Usuario x Artículo (8 features)
    Namespace 4: Meta-Features de Origen de Candidatos (13 features)
- Inmunidad arquitectónica a data leakage temporal.
- Relleno determinista de nulos con valores centinela documentados.
"""

from __future__ import annotations

import datetime
import gc
import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from config.settings import LAMBDA_DECAY

logger = logging.getLogger(__name__)


def build_user_features(
    transactions_df: pl.DataFrame,
    customers_df: pl.DataFrame,
) -> pl.DataFrame:
    r"""Namespace 1: Características históricas del usuario (9 features).

    Extrae el comportamiento de compra, capacidad de gasto, sesgo de canal y demografía:
    1. u_total_transactions: Volumen acumulado de compras (Int32).
    2. u_unique_articles: Variedad de prendas distintas adquiridas (Int32).
    3. u_mean_price: Precio medio histórico de sus compras (Float32).
    4. u_std_price: Desviación estándar del gasto, dispersión de presupuesto (Float32, 0.0 si tx <= 1).
    5. u_online_ratio: Proporción de compras en canal online (Float32, 0.5 si es nuevo).
    6. u_age: Edad imputada/normalizada del cliente (Int16, mediana 32).
    7. u_club_status: Estado del club de miembros codificado ordinalmente (Int8: 1=ACTIVE, 2=PRE-CREATE, 3=LEFT CLUB, 0=otro).
    8. u_fashion_news: Frecuencia de noticias codificada ordinalmente (Int8: 2=Regularly, 1=Monthly, 0=NONE).
    9. u_last_purchase_days_ago: Días transcurridos desde la última compra general del cliente (Int16, centinela 999).
    """
    assert "customer_idx" in transactions_df.columns, "customer_idx requerido en transactions_df"
    assert "customer_idx" in customers_df.columns, "customer_idx requerido en customers_df"

    max_date = transactions_df.select(pl.col("t_dat").max()).item()

    # Estadísticos agregados de transacciones previas
    u_stats = transactions_df.group_by("customer_idx").agg(
        pl.len().cast(pl.Int32).alias("u_total_transactions"),
        pl.col("article_id").n_unique().cast(pl.Int32).alias("u_unique_articles"),
        pl.col("price").mean().cast(pl.Float32).alias("u_mean_price"),
        pl.col("price").std().fill_null(0.0).cast(pl.Float32).alias("u_std_price"),
        (pl.col("sales_channel_id") == 2).mean().cast(pl.Float32).alias("u_online_ratio"),
        (pl.lit(max_date) - pl.col("t_dat").max())
        .dt.total_days()
        .cast(pl.Int16)
        .alias("u_last_purchase_days_ago"),
    )

    # Codificación ordinal de variables sociodemográficas y de club
    user_features = (
        customers_df.select(
            [
                "customer_idx",
                "age",
                "club_member_status",
                "fashion_news_frequency",
            ]
        )
        .join(u_stats, on="customer_idx", how="left")
        .with_columns(
            pl.col("u_total_transactions").fill_null(0).cast(pl.Int32),
            pl.col("u_unique_articles").fill_null(0).cast(pl.Int32),
            pl.col("u_mean_price").fill_null(0.0).cast(pl.Float32),
            pl.col("u_std_price").fill_null(0.0).cast(pl.Float32),
            pl.col("u_online_ratio").fill_null(0.5).cast(pl.Float32),
            pl.col("u_last_purchase_days_ago").fill_null(999).cast(pl.Int16),
            pl.col("age").round().fill_null(32.0).cast(pl.Int16).alias("u_age"),
            pl.when(pl.col("club_member_status") == "ACTIVE")
            .then(pl.lit(1, dtype=pl.Int8))
            .when(pl.col("club_member_status") == "PRE-CREATE")
            .then(pl.lit(2, dtype=pl.Int8))
            .when(pl.col("club_member_status") == "LEFT CLUB")
            .then(pl.lit(3, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("u_club_status"),
            pl.when(pl.col("fashion_news_frequency") == "Regularly")
            .then(pl.lit(2, dtype=pl.Int8))
            .when(pl.col("fashion_news_frequency") == "Monthly")
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("u_fashion_news"),
        )
        .select(
            [
                "customer_idx",
                "u_total_transactions",
                "u_unique_articles",
                "u_mean_price",
                "u_std_price",
                "u_online_ratio",
                "u_age",
                "u_club_status",
                "u_fashion_news",
                "u_last_purchase_days_ago",
            ]
        )
    )
    return user_features


def build_article_features(
    transactions_df: pl.DataFrame,
    articles_df: pl.DataFrame,
    lambda_decay: float = LAMBDA_DECAY,
) -> pl.DataFrame:
    r"""Namespace 2: Características intrínsecas y de demanda del artículo (9 features).

    Captura la popularidad temporal, rango de precio y taxonomía del producto:
    1. a_sales_count: Volumen total histórico de ventas del artículo (Int32).
    2. a_unique_customers: Número de clientes distintos compradores (Int32).
    3. a_mean_price: Precio promedio histórico de venta (Float32).
    4. a_sales_decayed: Ventas ponderadas con decaimiento exp(-lambda * delta_t) (Float32).
    5. a_is_recent_introduction: Indicador binario de si el artículo debutó en las últimas 2 semanas (Int8).
    6. a_product_type_no: Código del tipo de prenda/producto (Int32).
    7. a_graphical_appearance_no: Patrón gráfico/estampado (Int32).
    8. a_colour_group_code: Código del grupo cromático (Int16).
    9. a_department_no: Código numérico del departamento comercial (Int32).
    """
    assert "article_id" in transactions_df.columns, "article_id requerido en transactions_df"
    assert "article_id" in articles_df.columns, "article_id requerido en articles_df"

    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    cutoff_recent = max_date - datetime.timedelta(days=14)
    global_mean_price = transactions_df.select(pl.col("price").mean()).item() or 0.0278

    # Cálculo con decaimiento temporal exponencial de ventas
    tx_with_decay = transactions_df.with_columns(
        ((pl.lit(max_date) - pl.col("t_dat")).dt.total_days() / 7.0).alias("weeks_ago")
    ).with_columns((-lambda_decay * pl.col("weeks_ago")).exp().alias("decay_weight"))

    a_stats = tx_with_decay.group_by("article_id").agg(
        pl.len().cast(pl.Int32).alias("a_sales_count"),
        pl.col("customer_idx").n_unique().cast(pl.Int32).alias("a_unique_customers"),
        pl.col("price").mean().cast(pl.Float32).alias("a_mean_price"),
        pl.col("decay_weight").sum().cast(pl.Float32).alias("a_sales_decayed"),
        (pl.col("t_dat").min() >= cutoff_recent).cast(pl.Int8).alias("a_is_recent_introduction"),
    )

    article_features = (
        articles_df.select(
            [
                "article_id",
                "product_type_no",
                "graphical_appearance_no",
                "colour_group_code",
                "department_no",
            ]
        )
        .join(a_stats, on="article_id", how="left")
        .with_columns(
            pl.col("a_sales_count").fill_null(0).cast(pl.Int32),
            pl.col("a_unique_customers").fill_null(0).cast(pl.Int32),
            pl.col("a_mean_price").fill_null(global_mean_price).cast(pl.Float32),
            pl.col("a_sales_decayed").fill_null(0.0).cast(pl.Float32),
            pl.col("a_is_recent_introduction").fill_null(0).cast(pl.Int8),
            pl.col("product_type_no").fill_null(-1).cast(pl.Int32).alias("a_product_type_no"),
            pl.col("graphical_appearance_no")
            .fill_null(-1)
            .cast(pl.Int32)
            .alias("a_graphical_appearance_no"),
            pl.col("colour_group_code").fill_null(-1).cast(pl.Int16).alias("a_colour_group_code"),
            pl.col("department_no").fill_null(-1).cast(pl.Int32).alias("a_department_no"),
        )
        .select(
            [
                "article_id",
                "a_sales_count",
                "a_unique_customers",
                "a_mean_price",
                "a_sales_decayed",
                "a_is_recent_introduction",
                "a_product_type_no",
                "a_graphical_appearance_no",
                "a_colour_group_code",
                "a_department_no",
            ]
        )
    )
    return article_features


def build_interaction_features(
    candidates_df: pl.DataFrame,
    transactions_df: pl.DataFrame,
    user_features: pl.DataFrame,
    article_features: pl.DataFrame,
) -> pl.DataFrame:
    r"""Namespace 3: Interacción Usuario x Artículo (8 features).

    Mide afinidad individual, frecuencia de recompra previa, elasticidad de precio y canal:
    1. uxa_repurchase_count: Veces previas que el usuario adquirió este artículo exacto (Int16, imputar 0).
    2. uxa_days_since_last_purchase: Días transcurridos desde la última compra de este ítem (Int16, centinela 999).
    3. uxa_dept_affinity: Compras previas del usuario en el mismo departamento de la prenda (Int16, imputar 0).
    4. uxa_price_diff: Diferencia de precio relativo: p_art - p_user (Float32).
    5. uxa_price_ratio: Ratio de elasticidad precio: p_art / (p_user + 10^-4) (Float32).
    6. uxa_bought_dept_before: Indicador binario de si ha comprado en este departamento (Int8: 0 o 1).
    7. uxa_channel_affinity: Coincidencia entre canal preferido del usuario y canal dominante del artículo (Float32: 1.0, 0.5 o 0.0).
    8. uxa_is_favorite_dept: Indicador binario de si la prenda pertenece al departamento dominante del usuario (Int8: 0 o 1).
    """
    assert "customer_idx" in candidates_df.columns, "customer_idx requerido en candidates_df"
    assert "article_id" in candidates_df.columns, "article_id requerido en candidates_df"

    max_date = transactions_df.select(pl.col("t_dat").max()).item()

    # Historial de compras usuario-artículo específico
    uxa_history = transactions_df.group_by(["customer_idx", "article_id"]).agg(
        pl.len().cast(pl.Int16).alias("uxa_repurchase_count"),
        (pl.lit(max_date) - pl.col("t_dat").max())
        .dt.total_days()
        .cast(pl.Int16)
        .alias("uxa_days_since_last_purchase"),
    )

    # Afinidad por departamento (compras del usuario por department_no)
    tx_dept = transactions_df.join(
        article_features.select(["article_id", "a_department_no"]), on="article_id", how="inner"
    )
    user_dept_counts = tx_dept.group_by(["customer_idx", "a_department_no"]).len(
        name="uxa_dept_affinity"
    )
    user_fav_dept = (
        tx_dept.group_by(["customer_idx", "a_department_no"])
        .len(name="fav_count")
        .sort(["customer_idx", "fav_count"], descending=[False, True])
        .group_by("customer_idx")
        .first()
        .select(
            [
                pl.col("customer_idx"),
                pl.col("a_department_no").alias("u_favorite_dept"),
            ]
        )
    )

    # Canal dominante de venta por artículo (1: Tienda, 2: Online)
    art_dominant_channel = (
        transactions_df.group_by(["article_id", "sales_channel_id"])
        .len(name="ch_count")
        .sort(["article_id", "ch_count"], descending=[False, True])
        .group_by("article_id")
        .first()
        .select(
            [
                pl.col("article_id"),
                pl.col("sales_channel_id").cast(pl.Int8).alias("a_dominant_channel"),
            ]
        )
    )

    # Ensamblado vectorizado de interacción sobre los pares candidatos
    base_pairs = candidates_df.select(["customer_idx", "article_id"])

    interaction_df = (
        base_pairs.join(uxa_history, on=["customer_idx", "article_id"], how="left")
        .with_columns(
            pl.col("uxa_repurchase_count").fill_null(0).cast(pl.Int16),
            pl.col("uxa_days_since_last_purchase").fill_null(999).cast(pl.Int16),
        )
        .join(
            user_features.select(["customer_idx", "u_mean_price", "u_online_ratio"]),
            on="customer_idx",
            how="left",
        )
        .join(
            article_features.select(["article_id", "a_mean_price", "a_department_no"]),
            on="article_id",
            how="left",
        )
        .join(
            user_dept_counts,
            on=["customer_idx", "a_department_no"],
            how="left",
        )
        .join(
            user_fav_dept,
            on="customer_idx",
            how="left",
        )
        .join(
            art_dominant_channel,
            on="article_id",
            how="left",
        )
        .with_columns(
            pl.col("uxa_dept_affinity").fill_null(0).cast(pl.Int16),
            (pl.col("a_mean_price") - pl.col("u_mean_price"))
            .cast(pl.Float32)
            .alias("uxa_price_diff"),
            (pl.col("a_mean_price") / (pl.col("u_mean_price") + 1e-4))
            .cast(pl.Float32)
            .alias("uxa_price_ratio"),
            pl.when(pl.col("uxa_dept_affinity").fill_null(0) > 0)
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("uxa_bought_dept_before"),
            pl.when(
                (pl.col("u_favorite_dept").is_not_null())
                & (pl.col("a_department_no") == pl.col("u_favorite_dept"))
            )
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("uxa_is_favorite_dept"),
            # Canal preferido del usuario: 2 si online_ratio >= 0.5, sino 1
            pl.when(
                (pl.when(pl.col("u_online_ratio") >= 0.5).then(2).otherwise(1))
                == pl.col("a_dominant_channel")
            )
            .then(pl.lit(1.0, dtype=pl.Float32))
            .when(pl.col("a_dominant_channel").is_null())
            .then(pl.lit(0.5, dtype=pl.Float32))
            .otherwise(pl.lit(0.0, dtype=pl.Float32))
            .alias("uxa_channel_affinity"),
        )
        .select(
            [
                "customer_idx",
                "article_id",
                "uxa_repurchase_count",
                "uxa_days_since_last_purchase",
                "uxa_dept_affinity",
                "uxa_price_diff",
                "uxa_price_ratio",
                "uxa_bought_dept_before",
                "uxa_channel_affinity",
                "uxa_is_favorite_dept",
            ]
        )
    )
    return interaction_df


def build_candidate_meta_features(
    candidates_df: pl.DataFrame,
) -> pl.DataFrame:
    r"""Namespace 4: Meta-Features de Origen de Candidatos (13 features).

    Encapsula el consenso multi-heurístico y la procedencia de cada par candidato:
    1..8. is_R1 a is_R8: Indicadores binarios de proposición por cada heurística (Int8).
    9. n_sources: Número total de heurísticas que propusieron este ítem (Int8, 1 a 8).
    10. best_rank: Posición más alta obtenida entre las heurísticas (Int16, 1 a 100).
    11. source_diversity_score: Puntuación ponderada según la fiabilidad de las fuentes (Float32).
    12. is_personal_candidate: Flag de heurística personalizada: R1, R5 o R8 (Int8).
    13. is_exploration_candidate: Flag de heurística de descubrimiento/tendencia: R6 o R7 (Int8).
    """
    assert "customer_idx" in candidates_df.columns, "customer_idx requerido en candidates_df"
    assert "article_id" in candidates_df.columns, "article_id requerido en candidates_df"

    # Pesos empíricos calibrados de fiabilidad por fuente
    w_R1 = 0.25  # Recompra (alta precisión)
    w_R2 = 0.10  # Popularidad global
    w_R3 = 0.10  # Popularidad por edad
    w_R4 = 0.10  # Popularidad por canal
    w_R5 = 0.15  # Item-CF co-compras
    w_R6 = 0.10  # Familias de producto
    w_R7 = 0.10  # Trending
    w_R8 = 0.10  # Popularidad por departamento favorito

    meta_df = (
        candidates_df.select(
            [
                pl.col("customer_idx"),
                pl.col("article_id"),
                pl.col("is_R1").fill_null(False).cast(pl.Int8).alias("is_R1"),
                pl.col("is_R2").fill_null(False).cast(pl.Int8).alias("is_R2"),
                pl.col("is_R3").fill_null(False).cast(pl.Int8).alias("is_R3"),
                pl.col("is_R4").fill_null(False).cast(pl.Int8).alias("is_R4"),
                pl.col("is_R5").fill_null(False).cast(pl.Int8).alias("is_R5"),
                pl.col("is_R6").fill_null(False).cast(pl.Int8).alias("is_R6"),
                pl.col("is_R7").fill_null(False).cast(pl.Int8).alias("is_R7"),
                pl.col("is_R8").fill_null(False).cast(pl.Int8).alias("is_R8"),
                pl.col("n_sources").fill_null(1).cast(pl.Int8).alias("n_sources"),
                pl.col("best_rank").fill_null(999).cast(pl.Int16).alias("best_rank"),
            ]
        )
        .with_columns(
            (
                w_R1 * pl.col("is_R1").cast(pl.Float32)
                + w_R2 * pl.col("is_R2").cast(pl.Float32)
                + w_R3 * pl.col("is_R3").cast(pl.Float32)
                + w_R4 * pl.col("is_R4").cast(pl.Float32)
                + w_R5 * pl.col("is_R5").cast(pl.Float32)
                + w_R6 * pl.col("is_R6").cast(pl.Float32)
                + w_R7 * pl.col("is_R7").cast(pl.Float32)
                + w_R8 * pl.col("is_R8").cast(pl.Float32)
            )
            .cast(pl.Float32)
            .alias("source_diversity_score"),
            ((pl.col("is_R1") == 1) | (pl.col("is_R5") == 1) | (pl.col("is_R8") == 1))
            .cast(pl.Int8)
            .alias("is_personal_candidate"),
            ((pl.col("is_R6") == 1) | (pl.col("is_R7") == 1))
            .cast(pl.Int8)
            .alias("is_exploration_candidate"),
        )
        .select(
            [
                "customer_idx",
                "article_id",
                "is_R1",
                "is_R2",
                "is_R3",
                "is_R4",
                "is_R5",
                "is_R6",
                "is_R7",
                "is_R8",
                "n_sources",
                "best_rank",
                "source_diversity_score",
                "is_personal_candidate",
                "is_exploration_candidate",
            ]
        )
    )
    return meta_df


def _process_feature_chunk(
    cand_chunk: pl.DataFrame,
    user_feats: pl.DataFrame,
    art_feats: pl.DataFrame,
    uxa_history: pl.DataFrame,
    user_dept_counts: pl.DataFrame,
    user_fav_dept: pl.DataFrame,
    art_dominant_channel: pl.DataFrame,
    expected_cols: list[str],
) -> pl.DataFrame:
    """Procesa un lote o partición de candidatos vectorizadamente asegurando tipos y cero nulos."""
    # Base y Joins de entidades
    base = (
        cand_chunk.select(
            [
                pl.col("customer_idx").cast(pl.Int32),
                pl.col("article_id").cast(pl.Int32),
            ]
        )
        .join(user_feats, on="customer_idx", how="left")
        .join(art_feats, on="article_id", how="left")
    )

    # Joins y cómputo de interacción (NS3)
    base = (
        base.join(uxa_history, on=["customer_idx", "article_id"], how="left")
        .join(user_dept_counts, on=["customer_idx", "a_department_no"], how="left")
        .join(user_fav_dept, on="customer_idx", how="left")
        .join(art_dominant_channel, on="article_id", how="left")
        .with_columns(
            pl.col("uxa_repurchase_count").fill_null(0).cast(pl.Int16),
            pl.col("uxa_days_since_last_purchase").fill_null(999).cast(pl.Int16),
            pl.col("uxa_dept_affinity").fill_null(0).cast(pl.Int16),
            (pl.col("a_mean_price") - pl.col("u_mean_price"))
            .cast(pl.Float32)
            .alias("uxa_price_diff"),
            (pl.col("a_mean_price") / (pl.col("u_mean_price") + 1e-4))
            .cast(pl.Float32)
            .alias("uxa_price_ratio"),
            pl.when(pl.col("uxa_dept_affinity").fill_null(0) > 0)
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("uxa_bought_dept_before"),
            pl.when(
                (pl.col("u_favorite_dept").is_not_null())
                & (pl.col("a_department_no") == pl.col("u_favorite_dept"))
            )
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("uxa_is_favorite_dept"),
            pl.when(
                (pl.when(pl.col("u_online_ratio") >= 0.5).then(2).otherwise(1))
                == pl.col("a_dominant_channel")
            )
            .then(pl.lit(1.0, dtype=pl.Float32))
            .when(pl.col("a_dominant_channel").is_null())
            .then(pl.lit(0.5, dtype=pl.Float32))
            .otherwise(pl.lit(0.0, dtype=pl.Float32))
            .alias("uxa_channel_affinity"),
        )
    )

    # Cómputo de meta-features de candidatos (NS4)
    w_R1, w_R2, w_R3, w_R4, w_R5, w_R6, w_R7, w_R8 = 0.25, 0.10, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10
    meta = cand_chunk.select(
        [
            pl.col("customer_idx").cast(pl.Int32),
            pl.col("article_id").cast(pl.Int32),
            pl.col("is_R1").fill_null(False).cast(pl.Int8).alias("is_R1"),
            pl.col("is_R2").fill_null(False).cast(pl.Int8).alias("is_R2"),
            pl.col("is_R3").fill_null(False).cast(pl.Int8).alias("is_R3"),
            pl.col("is_R4").fill_null(False).cast(pl.Int8).alias("is_R4"),
            pl.col("is_R5").fill_null(False).cast(pl.Int8).alias("is_R5"),
            pl.col("is_R6").fill_null(False).cast(pl.Int8).alias("is_R6"),
            pl.col("is_R7").fill_null(False).cast(pl.Int8).alias("is_R7"),
            pl.col("is_R8").fill_null(False).cast(pl.Int8).alias("is_R8"),
            pl.col("n_sources").fill_null(1).cast(pl.Int8).alias("n_sources"),
            pl.col("best_rank").fill_null(999).cast(pl.Int16).alias("best_rank"),
        ]
    ).with_columns(
        (
            w_R1 * pl.col("is_R1").cast(pl.Float32)
            + w_R2 * pl.col("is_R2").cast(pl.Float32)
            + w_R3 * pl.col("is_R3").cast(pl.Float32)
            + w_R4 * pl.col("is_R4").cast(pl.Float32)
            + w_R5 * pl.col("is_R5").cast(pl.Float32)
            + w_R6 * pl.col("is_R6").cast(pl.Float32)
            + w_R7 * pl.col("is_R7").cast(pl.Float32)
            + w_R8 * pl.col("is_R8").cast(pl.Float32)
        )
        .cast(pl.Float32)
        .alias("source_diversity_score"),
        ((pl.col("is_R1") == 1) | (pl.col("is_R5") == 1) | (pl.col("is_R8") == 1))
        .cast(pl.Int8)
        .alias("is_personal_candidate"),
        ((pl.col("is_R6") == 1) | (pl.col("is_R7") == 1))
        .cast(pl.Int8)
        .alias("is_exploration_candidate"),
    )

    chunk_matrix = base.join(meta, on=["customer_idx", "article_id"], how="left").select(
        expected_cols
    )
    return chunk_matrix


def build_full_feature_matrix(
    candidates_df: pl.DataFrame,
    transactions_df: pl.DataFrame,
    customers_df: pl.DataFrame,
    articles_df: pl.DataFrame,
    lambda_decay: float = LAMBDA_DECAY,
    out_path: Path | str | None = None,
    chunk_size: int = 2_500_000,
    streaming_threshold: int = 1_000_000,
) -> pl.DataFrame | None:
    r"""Orquesta la construcción y unión de las 39 características disjuntas en 4 namespaces.

    Schema Resultante (41 columnas: 2 claves + 39 características):
    --------------------------------------------------------------
    Claves:
      - customer_idx (Int32), article_id (Int32)
    Namespace 1: Usuario (9 features):
      - u_total_transactions (Int32), u_unique_articles (Int32), u_mean_price (Float32),
        u_std_price (Float32), u_online_ratio (Float32), u_age (Int16),
        u_club_status (Int8), u_fashion_news (Int8), u_last_purchase_days_ago (Int16)
    Namespace 2: Artículo (9 features):
      - a_sales_count (Int32), a_unique_customers (Int32), a_mean_price (Float32),
        a_sales_decayed (Float32), a_is_recent_introduction (Int8), a_product_type_no (Int32),
        a_graphical_appearance_no (Int32), a_colour_group_code (Int16), a_department_no (Int32)
    Namespace 3: Interacción Usuario x Artículo (8 features):
      - uxa_repurchase_count (Int16), uxa_days_since_last_purchase (Int16),
        uxa_dept_affinity (Int16), uxa_price_diff (Float32), uxa_price_ratio (Float32),
        uxa_bought_dept_before (Int8), uxa_channel_affinity (Float32), uxa_is_favorite_dept (Int8)
    Namespace 4: Meta-Features de Candidatos (13 features):
      - is_R1 a is_R8 (Int8 x 8), n_sources (Int8), best_rank (Int16),
        source_diversity_score (Float32), is_personal_candidate (Int8), is_exploration_candidate (Int8)
    """
    assert candidates_df.height > 0, "candidates_df no puede estar vacío"
    assert transactions_df.height > 0, "transactions_df no puede estar vacío"

    expected_cols = [
        "customer_idx",
        "article_id",
        # NS1
        "u_total_transactions",
        "u_unique_articles",
        "u_mean_price",
        "u_std_price",
        "u_online_ratio",
        "u_age",
        "u_club_status",
        "u_fashion_news",
        "u_last_purchase_days_ago",
        # NS2
        "a_sales_count",
        "a_unique_customers",
        "a_mean_price",
        "a_sales_decayed",
        "a_is_recent_introduction",
        "a_product_type_no",
        "a_graphical_appearance_no",
        "a_colour_group_code",
        "a_department_no",
        # NS3
        "uxa_repurchase_count",
        "uxa_days_since_last_purchase",
        "uxa_dept_affinity",
        "uxa_price_diff",
        "uxa_price_ratio",
        "uxa_bought_dept_before",
        "uxa_channel_affinity",
        "uxa_is_favorite_dept",
        # NS4
        "is_R1",
        "is_R2",
        "is_R3",
        "is_R4",
        "is_R5",
        "is_R6",
        "is_R7",
        "is_R8",
        "n_sources",
        "best_rank",
        "source_diversity_score",
        "is_personal_candidate",
        "is_exploration_candidate",
    ]

    logger.info("  -> [NS1] Construyendo características del usuario (9 features)...")
    user_feats = build_user_features(transactions_df, customers_df)

    logger.info("  -> [NS2] Construyendo características del artículo (9 features)...")
    art_feats = build_article_features(transactions_df, articles_df, lambda_decay=lambda_decay)

    logger.info("  -> [NS3] Precalculando tablas de interacción usuario-artículo...")
    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    uxa_history = transactions_df.group_by(["customer_idx", "article_id"]).agg(
        pl.len().cast(pl.Int16).alias("uxa_repurchase_count"),
        (pl.lit(max_date) - pl.col("t_dat").max())
        .dt.total_days()
        .cast(pl.Int16)
        .alias("uxa_days_since_last_purchase"),
    )

    tx_dept = transactions_df.join(
        art_feats.select(["article_id", "a_department_no"]), on="article_id", how="inner"
    )
    user_dept_counts = tx_dept.group_by(["customer_idx", "a_department_no"]).len(
        name="uxa_dept_affinity"
    )
    user_fav_dept = (
        tx_dept.group_by(["customer_idx", "a_department_no"])
        .len(name="fav_count")
        .sort(["customer_idx", "fav_count"], descending=[False, True])
        .group_by("customer_idx")
        .first()
        .select(
            [
                pl.col("customer_idx"),
                pl.col("a_department_no").alias("u_favorite_dept"),
            ]
        )
    )
    del tx_dept

    art_dominant_channel = (
        transactions_df.group_by(["article_id", "sales_channel_id"])
        .len(name="ch_count")
        .sort(["article_id", "ch_count"], descending=[False, True])
        .group_by("article_id")
        .first()
        .select(
            [
                pl.col("article_id"),
                pl.col("sales_channel_id").cast(pl.Int8).alias("a_dominant_channel"),
            ]
        )
    )

    total_candidates = candidates_df.height

    if out_path is not None and total_candidates > streaming_threshold:
        # Modo Streaming out-of-core a disco con PyArrow ParquetWriter
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"  -> Ensamblando y transmitiendo a disco en lotes ({total_candidates:,} pares, chunk_size={chunk_size:,})..."
        )

        n_chunks = (total_candidates + chunk_size - 1) // chunk_size
        writer: pq.ParquetWriter | None = None

        for i in range(n_chunks):
            c_start = i * chunk_size
            chunk_slice = candidates_df.slice(c_start, chunk_size)
            chunk_matrix = _process_feature_chunk(
                cand_chunk=chunk_slice,
                user_feats=user_feats,
                art_feats=art_feats,
                uxa_history=uxa_history,
                user_dept_counts=user_dept_counts,
                user_fav_dept=user_fav_dept,
                art_dominant_channel=art_dominant_channel,
                expected_cols=expected_cols,
            )
            del chunk_slice

            arrow_table = chunk_matrix.to_arrow()
            del chunk_matrix

            if writer is None:
                writer = pq.ParquetWriter(out_path, schema=arrow_table.schema, compression="zstd")
            writer.write_table(arrow_table)
            del arrow_table
            gc.collect()

        if writer is not None:
            writer.close()

        del (
            user_feats,
            art_feats,
            uxa_history,
            user_dept_counts,
            user_fav_dept,
            art_dominant_channel,
        )
        gc.collect()

        logger.info(f"  [OK] Matriz de características transmitida a disco: {out_path.name}")
        return None

    elif total_candidates > streaming_threshold:
        # Procesa en lotes y concatena en memoria
        logger.info(f"  -> Ensamblando en lotes en memoria ({total_candidates:,} pares)...")
        n_chunks = (total_candidates + chunk_size - 1) // chunk_size
        chunks_res = []
        for i in range(n_chunks):
            c_start = i * chunk_size
            chunk_slice = candidates_df.slice(c_start, chunk_size)
            chunk_matrix = _process_feature_chunk(
                cand_chunk=chunk_slice,
                user_feats=user_feats,
                art_feats=art_feats,
                uxa_history=uxa_history,
                user_dept_counts=user_dept_counts,
                user_fav_dept=user_fav_dept,
                art_dominant_channel=art_dominant_channel,
                expected_cols=expected_cols,
            )
            del chunk_slice
            chunks_res.append(chunk_matrix)
            gc.collect()

        full_matrix = pl.concat(chunks_res)
        del chunks_res
        gc.collect()
        return full_matrix

    else:
        # Modo muestra estándar en una pasada
        logger.info("  -> Ensamblando matriz tabular de muestra...")
        full_matrix = _process_feature_chunk(
            cand_chunk=candidates_df,
            user_feats=user_feats,
            art_feats=art_feats,
            uxa_history=uxa_history,
            user_dept_counts=user_dept_counts,
            user_fav_dept=user_fav_dept,
            art_dominant_channel=art_dominant_channel,
            expected_cols=expected_cols,
        )

        assert full_matrix.height == candidates_df.height, "Inconsistencia de filas tras joins"
        assert len(full_matrix.columns) == len(expected_cols), (
            f"Columnas inesperadas: {len(full_matrix.columns)}"
        )
        logger.info(
            f"  [OK] Matriz construida: {full_matrix.height:,} filas × {len(full_matrix.columns)} columnas"
        )
        if out_path is not None:
            out_p = Path(out_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            full_matrix.write_parquet(out_p, compression="zstd")
            logger.info(f"  [OK] Matriz de muestra guardada en: {out_p.name}")
        return full_matrix
