"""Esquema de validación temporal estricto (Inmunidad arquitectónica a fugas de datos).

En sistemas de recomendación para retail de moda rápida, la aplicación de Random K-Fold
Cross-Validation constituye un error metodológico crítico: rompe la causalidad temporal
e introduce look-ahead bias, permitiendo que compras futuras influyan en la predicción
de compras pasadas.

Este módulo implementa la partición puramente temporal y la preparación de datos para
modelos Learning-to-Rank (LTR / LGBMRanker):
1. split_transactions_temporal: División temporal estricta de transacciones.
2. build_ground_truth_dict: Extracción vectorizada de compras reales para evaluación de ranking.
3. prepare_ranker_split: Asignación de target binario, cálculo de query_lengths (groups)
   y soporte de negative downsampling estratificado por usuario.
4. assert_zero_temporal_leakage: Protocolo de auditoría defensiva contra contaminación temporal.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import NamedTuple

import polars as pl

# Asegurar importación de config y módulos raíz en ejecución directa
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import DATA_PROCESSED_DIR, RANDOM_SEED, VAL_WINDOW_DAYS  # noqa: E402


class TemporalSplitResult(NamedTuple):
    """Contenedor inmutable de particiones temporales de transacciones."""

    train_df: pl.DataFrame
    val_df: pl.DataFrame
    val_start_date: datetime.date
    val_end_date: datetime.date


class RollingTemporalSplit(NamedTuple):
    """Particiones temporales desacopladas para entrenamiento y validación de ranking sin fugas.

    Garantiza causalidad temporal estricta en dos ventanas deslizantes relativas a max(t_dat):
    1. Ventana de Entrenamiento:
       - train_features_tx: transacciones históricas hasta train_features_max.
       - train_target_tx: transacciones objetivo de compra durante train_target_start..train_target_end.
    2. Ventana de Validación:
       - val_features_tx: transacciones históricas hasta val_features_max.
       - val_target_tx: transacciones objetivo de compra durante val_target_start..val_target_end.
    """

    train_features_tx: pl.DataFrame
    train_target_tx: pl.DataFrame
    val_features_tx: pl.DataFrame
    val_target_tx: pl.DataFrame
    train_features_max: datetime.date
    train_target_start: datetime.date
    train_target_end: datetime.date
    val_features_max: datetime.date
    val_target_start: datetime.date
    val_target_end: datetime.date


class RankerSplitData(NamedTuple):
    """Contenedor estructurado de datos preparados para entrenamiento o evaluación con LGBMRanker."""

    X: pl.DataFrame
    y: pl.Series
    groups: list[int]
    customer_ids: pl.Series
    article_ids: pl.Series
    feature_names: list[str]
    df: pl.DataFrame


def assert_zero_temporal_leakage(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    val_start_date: datetime.date | None = None,
) -> bool:
    r"""Auditoría defensiva formal contra contaminación temporal (Look-ahead bias).

    Verifica estrictamente que:
    $$\max(t_{\text{train}}) < \min(t_{\text{val}})$$
    y que ninguna transacción de entrenamiento posea fecha contemporánea o posterior
    al horizonte de validación.

    Parameters
    ----------
    train_df : pl.DataFrame
        Transacciones de entrenamiento con columna 't_dat'.
    val_df : pl.DataFrame
        Transacciones de validación / prueba con columna 't_dat'.
    val_start_date : datetime.date | None, optional
        Fecha umbral de inicio de la ventana de validación.

    Returns
    -------
    bool
        True si supera todas las verificaciones matemáticas.
    """
    if "t_dat" not in train_df.columns:
        raise ValueError("Columna 't_dat' obligatoria en train_df")
    if "t_dat" not in val_df.columns:
        raise ValueError("Columna 't_dat' obligatoria en val_df")
    if train_df.height == 0:
        raise ValueError("train_df no puede estar vacío")
    if val_df.height == 0:
        raise ValueError("val_df no puede estar vacío")

    train_max_date = train_df.select(pl.col("t_dat").max()).item()
    val_min_date = val_df.select(pl.col("t_dat").min()).item()

    if not isinstance(train_max_date, datetime.date):
        raise TypeError(f"train max debe ser Date, recibido {type(train_max_date)}")
    if not isinstance(val_min_date, datetime.date):
        raise TypeError(f"val min debe ser Date, recibido {type(val_min_date)}")

    # Separación temporal estricta
    if train_max_date >= val_min_date:
        raise AssertionError(
            f"[LOOK-AHEAD BIAS DETECTADO] train_max ({train_max_date}) >= val_min ({val_min_date}). "
            f"Contaminación de transacciones futuras en el histórico de entrenamiento."
        )

    # Comprobación contra fecha umbral fija si fue provista
    if val_start_date is not None:
        if train_max_date >= val_start_date:
            raise AssertionError(
                f"train_max ({train_max_date}) debe ser estrictamente menor que val_start_date ({val_start_date})"
            )
        if val_min_date < val_start_date:
            raise AssertionError(
                f"val_min ({val_min_date}) debe ser mayor o igual que val_start_date ({val_start_date})"
            )

    # Ausencia absoluta de registros filtrados
    leak_count = train_df.filter(pl.col("t_dat") >= val_min_date).height
    if leak_count > 0:
        raise AssertionError(
            f"Se encontraron {leak_count} transacciones con fechas futuras en train_df!"
        )

    return True


def split_transactions_temporal(
    transactions_df: pl.DataFrame,
    val_days: int = VAL_WINDOW_DAYS,
    history_days: int | None = None,
    val_start_date: datetime.date | str | None = None,
) -> TemporalSplitResult:
    r"""Divide las transacciones en partición de entrenamiento y validación temporal estricta.

    Parameters
    ----------
    transactions_df : pl.DataFrame
        DataFrame con columna 't_dat' de tipo pl.Date.
    val_days : int, optional
        Días asignados a la ventana de validación (por defecto 7 días = 1 semana).
    history_days : int | None, optional
        Días de historial previo retenidos para el entrenamiento (None para retener todo el histórico).
    val_start_date : datetime.date | str | None, optional
        Fecha exacta de corte (YYYY-MM-DD). Si se omite, se calcula restando val_days
        a la fecha máxima presente en transactions_df.

    Returns
    -------
    TemporalSplitResult
        Tupla inmutable con (train_df, val_df, val_start_date, val_end_date).
    """
    if "t_dat" not in transactions_df.columns:
        raise ValueError("La columna 't_dat' es obligatoria para partición temporal")
    if transactions_df.height == 0:
        raise ValueError("El DataFrame no puede estar vacío")

    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    if not isinstance(max_date, datetime.date):
        raise TypeError(f"t_dat debe ser pl.Date, recibido {type(max_date)}")

    # Resolver fecha de inicio de validación
    if val_start_date is not None:
        if isinstance(val_start_date, str):
            cutoff_date = datetime.date.fromisoformat(val_start_date)
        elif isinstance(val_start_date, datetime.date):
            cutoff_date = val_start_date
        else:
            raise TypeError(
                f"val_start_date debe ser date o str ISO, recibido {type(val_start_date)}"
            )
    else:
        cutoff_date = max_date - datetime.timedelta(days=val_days - 1)

    # Filtrar entrenamiento: estrictamente antes del cutoff_date
    train_df = transactions_df.filter(pl.col("t_dat") < cutoff_date)

    # Si se define una ventana histórica finita (ej. 28 o 35 días)
    if history_days is not None:
        train_start_date = cutoff_date - datetime.timedelta(days=history_days)
        train_df = train_df.filter(pl.col("t_dat") >= train_start_date)

    # Filtrar validación: desde cutoff_date hasta max_date
    val_df = transactions_df.filter(
        (pl.col("t_dat") >= cutoff_date) & (pl.col("t_dat") <= max_date)
    )

    # Validación de time-based split contra data leakage
    assert_zero_temporal_leakage(train_df, val_df, cutoff_date)

    return TemporalSplitResult(
        train_df=train_df,
        val_df=val_df,
        val_start_date=cutoff_date,
        val_end_date=max_date,
    )


def assert_zero_feature_target_leakage(
    feature_tx: pl.DataFrame,
    target_tx: pl.DataFrame,
    step_name: str = "split",
) -> bool:
    r"""Audita formalmente que ninguna transacción de la matriz de características pertenezca o sea posterior al target.

    Verifica estrictamente que:
    $$\max(t_{\text{features}}) < \min(t_{\text{target}})$$
    y que ninguna fila de feature_tx posea t_dat >= min(target_tx.t_dat).

    Parameters
    ----------
    feature_tx : pl.DataFrame
        Transacciones utilizadas para el cómputo de características explicativas.
    target_tx : pl.DataFrame
        Transacciones utilizadas para la etiqueta objetivo de compra (y=1).
    step_name : str, optional
        Identificador de la etapa para trazabilidad de errores (ej. 'train_rolling', 'val_rolling').

    Returns
    -------
    bool
        True si supera la auditoría matemática.

    Raises
    ------
    ValueError
        Si los DataFrames están vacíos o carecen de columna 't_dat'.
    TypeError
        Si 't_dat' no es de tipo temporal date.
    AssertionError
        Si se detecta cualquier contaminación temporal (funciona bajo python -O al ser explícito).
    """
    if "t_dat" not in feature_tx.columns:
        raise ValueError(f"[{step_name}] Columna 't_dat' obligatoria en feature_tx")
    if "t_dat" not in target_tx.columns:
        raise ValueError(f"[{step_name}] Columna 't_dat' obligatoria en target_tx")
    if feature_tx.height == 0:
        raise ValueError(f"[{step_name}] feature_tx no puede estar vacío")
    if target_tx.height == 0:
        raise ValueError(f"[{step_name}] target_tx no puede estar vacío")

    feat_max = feature_tx.select(pl.col("t_dat").max()).item()
    tgt_min = target_tx.select(pl.col("t_dat").min()).item()

    if not isinstance(feat_max, datetime.date):
        raise TypeError(f"[{step_name}] feature max t_dat debe ser Date, recibido {type(feat_max)}")
    if not isinstance(tgt_min, datetime.date):
        raise TypeError(f"[{step_name}] target min t_dat debe ser Date, recibido {type(tgt_min)}")

    if feat_max >= tgt_min:
        raise AssertionError(
            f"[{step_name}] [LOOK-AHEAD LEAKAGE DETECTADO] feat_max ({feat_max}) >= tgt_min ({tgt_min}). "
            f"Las características contienen información contemporánea o futura respecto al periodo objetivo."
        )

    leaks = feature_tx.filter(pl.col("t_dat") >= tgt_min).height
    if leaks > 0:
        raise AssertionError(
            f"[{step_name}] Se encontraron {leaks} transacciones contemporáneas o futuras en feature_tx!"
        )

    return True


def create_rolling_temporal_split(
    transactions_df: pl.DataFrame,
    val_days: int = VAL_WINDOW_DAYS,
    train_target_days: int = VAL_WINDOW_DAYS,
    history_days: int | None = None,
) -> RollingTemporalSplit:
    r"""Genera particiones temporales desacopladas en ventana deslizante (Rolling Window).

    Deriva todas las fechas de corte algebraicamente a partir de $\max(t_{\text{dat}})$ del
    DataFrame provisto y de los horizontes relativos val_days y train_target_days, garantizando
    independencia total de fechas de calendario hardcodeadas:

    1. Ventana de Validación:
       - val_target_end = max_date
       - val_target_start = max_date - timedelta(days=val_days - 1)
       - val_features_max = val_target_start - timedelta(days=1)
       - val_target_tx: transacciones con t_dat en [val_target_start, val_target_end]
       - val_features_tx: transacciones con t_dat <= val_features_max

    2. Ventana de Entrenamiento (semana previa desacoplada):
       - train_target_end = val_features_max
       - train_target_start = train_target_end - timedelta(days=train_target_days - 1)
       - train_features_max = train_target_start - timedelta(days=1)
       - train_target_tx: transacciones con t_dat en [train_target_start, train_target_end]
       - train_features_tx: transacciones con t_dat <= train_features_max

    Parameters
    ----------
    transactions_df : pl.DataFrame
        Transacciones con columna 't_dat' de tipo pl.Date.
    val_days : int, optional
        Días de la ventana objetivo de validación (por defecto 7).
    train_target_days : int, optional
        Días de la ventana objetivo de entrenamiento (por defecto 7).
    history_days : int | None, optional
        Días máximos de historial para features (None para retener todo el histórico hacia atrás).

    Returns
    -------
    RollingTemporalSplit
        Tupla inmutable con DataFrames y fechas calculadas dinámicamente.
    """
    if "t_dat" not in transactions_df.columns:
        raise ValueError("Columna 't_dat' obligatoria en transactions_df")
    if transactions_df.height == 0:
        raise ValueError("transactions_df no puede estar vacío")

    max_date = transactions_df.select(pl.col("t_dat").max()).item()
    if not isinstance(max_date, datetime.date):
        raise TypeError(f"t_dat debe ser Date, recibido {type(max_date)}")

    # Ventana de Validación
    val_target_end = max_date
    val_target_start = max_date - datetime.timedelta(days=val_days - 1)
    val_features_max = val_target_start - datetime.timedelta(days=1)

    val_target_tx = transactions_df.filter(
        (pl.col("t_dat") >= val_target_start) & (pl.col("t_dat") <= val_target_end)
    )
    val_features_tx = transactions_df.filter(pl.col("t_dat") <= val_features_max)
    if history_days is not None:
        val_feat_min = val_features_max - datetime.timedelta(days=history_days - 1)
        val_features_tx = val_features_tx.filter(pl.col("t_dat") >= val_feat_min)

    # Ventana de Entrenamiento (semana previa desacoplada)
    train_target_end = val_features_max
    train_target_start = train_target_end - datetime.timedelta(days=train_target_days - 1)
    train_features_max = train_target_start - datetime.timedelta(days=1)

    train_target_tx = transactions_df.filter(
        (pl.col("t_dat") >= train_target_start) & (pl.col("t_dat") <= train_target_end)
    )
    train_features_tx = transactions_df.filter(pl.col("t_dat") <= train_features_max)
    if history_days is not None:
        train_feat_min = train_features_max - datetime.timedelta(days=history_days - 1)
        train_features_tx = train_features_tx.filter(pl.col("t_dat") >= train_feat_min)

    # Validación de time-based split en train y val
    assert_zero_feature_target_leakage(
        train_features_tx, train_target_tx, step_name="train_rolling"
    )
    assert_zero_feature_target_leakage(val_features_tx, val_target_tx, step_name="val_rolling")

    return RollingTemporalSplit(
        train_features_tx=train_features_tx,
        train_target_tx=train_target_tx,
        val_features_tx=val_features_tx,
        val_target_tx=val_target_tx,
        train_features_max=train_features_max,
        train_target_start=train_target_start,
        train_target_end=train_target_end,
        val_features_max=val_features_max,
        val_target_start=val_target_start,
        val_target_end=val_target_end,
    )


def build_ground_truth_dict(
    val_df: pl.DataFrame,
    customer_indices: list[int] | pl.Series | None = None,
    active_only: bool = True,
) -> dict[int, list[int]]:
    """Extrae el Ground Truth local de compras reales por usuario para evaluación de ranking.

    Parameters
    ----------
    val_df : pl.DataFrame
        Transacciones de la semana de prueba conteniendo ['customer_idx', 'article_id'].
    customer_indices : list[int] | pl.Series | None, optional
        Universo total de usuarios a evaluar. Si se provee y active_only=False, los usuarios
        sin compras en la semana de prueba se indexan con lista vacía [].
    active_only : bool, optional
        Si es True (defecto), retorna únicamente los clientes que realizaron al menos una compra.

    Returns
    -------
    dict[int, list[int]]
        Mapeo {customer_idx: [article_id_1, article_id_2, ...]}.
    """
    assert "customer_idx" in val_df.columns, "customer_idx requerido en val_df"
    assert "article_id" in val_df.columns, "article_id requerido en val_df"

    # Agrupar compras reales sin duplicados por usuario
    gt_agg = val_df.group_by("customer_idx").agg(
        pl.col("article_id").unique().alias("purchased_articles")
    )

    cust_list = gt_agg["customer_idx"].to_list()
    art_list = gt_agg["purchased_articles"].to_list()
    gt_dict: dict[int, list[int]] = dict(zip(cust_list, art_list, strict=False))

    # Si se solicitan todos los clientes del pool incluyendo los inactivos
    if not active_only and customer_indices is not None:
        target_indices = (
            customer_indices.to_list()
            if isinstance(customer_indices, pl.Series)
            else list(customer_indices)
        )
        for c in target_indices:
            if c not in gt_dict:
                gt_dict[c] = []

    return gt_dict


def prepare_ranker_split(
    feature_matrix: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    negative_ratio: int | None = None,
    drop_empty_queries: bool = False,
    seed: int = RANDOM_SEED,
) -> RankerSplitData:
    r"""Asigna la etiqueta objetivo binaria y estructura el dataset para LGBMRanker.

    Etiqueta Target:
    $$y_{ui} = \begin{cases} 1 & \text{si el usuario } u \text{ compró el artículo } i \text{ en ground\_truth\_df} \\ 0 & \text{en caso contrario} \end{cases}$$

    Garantiza:
    1. Agrupamiento contiguo por customer_idx para compatibilidad estricta con LightGBM.
    2. Vector de longitudes de grupo ('groups') donde sum(groups) == len(X).
    3. Soporte opcional de Negative Downsampling estratificado por cliente.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz tabular de candidatos con columnas ['customer_idx', 'article_id'] + ~35 features.
    ground_truth_df : pl.DataFrame
        Transacciones de compra con ['customer_idx', 'article_id'].
    negative_ratio : int | None, optional
        Ratio de ejemplos negativos (y=0) retenidos por cada positivo (y=1).
        Si es None o <= 0, se conservan todos los pares negativos (recomendado para validación).
    drop_empty_queries : bool, optional
        Si es True, descarta usuarios que tengan 0 candidatos positivos en el pool (optimiza gradientes en train).
    seed : int, optional
        Semilla determinista para el muestreo estocástico de negativos.

    Returns
    -------
    RankerSplitData
        NamedTuple con (X, y, groups, customer_ids, article_ids, feature_names, df).
    """
    assert "customer_idx" in feature_matrix.columns, "customer_idx requerido en feature_matrix"
    assert "article_id" in feature_matrix.columns, "article_id requerido en feature_matrix"
    assert "customer_idx" in ground_truth_df.columns, "customer_idx requerido en ground_truth_df"
    assert "article_id" in ground_truth_df.columns, "article_id requerido en ground_truth_df"
    assert feature_matrix.height > 0, "feature_matrix no puede estar vacía"

    # Asignación vectorizada de target binario
    gt_pairs = (
        ground_truth_df.select(["customer_idx", "article_id"])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("target"))
    )

    labeled = feature_matrix.join(
        gt_pairs, on=["customer_idx", "article_id"], how="left"
    ).with_columns(pl.col("target").fill_null(0).cast(pl.Int8))

    # Filtrado de consultas sin positivos (opcional para entrenamiento listwise)
    if drop_empty_queries:
        pos_users = labeled.filter(pl.col("target") == 1).select("customer_idx").unique()
        labeled = labeled.join(pos_users, on="customer_idx", how="inner")
        assert labeled.height > 0, (
            "No quedaron consultas con positivos tras drop_empty_queries=True"
        )

    # Negative Downsampling estratificado por usuario (si fue solicitado)
    if negative_ratio is not None and negative_ratio > 0:
        positives = labeled.filter(pl.col("target") == 1)
        negatives = labeled.filter(pl.col("target") == 0)

        # Contar positivos por usuario para muestrear proporcionalmente (N_pos * ratio)
        n_pos_per_user = positives.group_by("customer_idx").len(name="n_pos")

        sampled_negatives = (
            negatives.join(n_pos_per_user, on="customer_idx", how="left")
            .with_columns(
                pl.col("n_pos").fill_null(1)
            )  # Fallback a 1 si el usuario no tiene positivos
            .with_columns(
                pl.int_range(0, pl.len()).shuffle(seed=seed).over("customer_idx").alias("neg_rank")
            )
            .filter(pl.col("neg_rank") < (pl.col("n_pos") * negative_ratio))
            .drop(["n_pos", "neg_rank"])
        )

        prepared_df = pl.concat([positives, sampled_negatives])
    else:
        prepared_df = labeled

    # Ordenamiento estricto contiguo por cliente para LightGBM
    # LightGBM asume que todas las filas de cada query son adyacentes
    prepared_df = prepared_df.sort("customer_idx")

    # Cálculo del vector de longitudes de grupo
    group_counts = prepared_df.group_by("customer_idx", maintain_order=True).len(name="group_len")
    groups: list[int] = group_counts["group_len"].to_list()

    # Validación de longitud y consistencia de grupos para LightGBM
    assert len(groups) > 0, "El vector de grupos no puede estar vacío"
    assert sum(groups) == prepared_df.height, (
        f"Inconsistencia de grupos en LightGBM: sum(groups)={sum(groups)} != df.height={prepared_df.height}"
    )
    assert len(groups) == group_counts.height, (
        f"Inconsistencia en número de grupos: {len(groups)} != {group_counts.height}"
    )
    assert all(g > 0 for g in groups), "Se detectaron grupos con longitud <= 0"

    # Desacoplamiento de matrices X, y y claves
    exclude_cols = {"customer_idx", "article_id", "target", "source"}
    feature_names = [c for c in prepared_df.columns if c not in exclude_cols]

    X = prepared_df.select(feature_names)
    y = prepared_df["target"]
    customer_ids = prepared_df["customer_idx"]
    article_ids = prepared_df["article_id"]

    return RankerSplitData(
        X=X,
        y=y,
        groups=groups,
        customer_ids=customer_ids,
        article_ids=article_ids,
        feature_names=feature_names,
        df=prepared_df,
    )


if __name__ == "__main__":
    print("=" * 75)
    print("  VERIFICACIÓN INDEPENDIENTE: SRC/UTILS/VALIDATION.PY")
    print("=" * 75)

    tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
    feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"

    if not tx_path.exists():
        print(f"[ERROR] No se encontró {tx_path}. Ejecute scripts previos primero.")
        exit(1)

    print("-> Cargando transacciones de prueba...")
    tx_df = pl.read_parquet(tx_path)

    # Prueba de split temporal
    print("-> Ejecutando partición temporal estricta (val_days=7)...")
    split_res = split_transactions_temporal(tx_df, val_days=7)
    print(f"  * Train rows        : {split_res.train_df.height:,}")
    print(f"  * Val rows          : {split_res.val_df.height:,}")
    print(f"  * Val Start Date    : {split_res.val_start_date}")
    print(f"  * Val End Date      : {split_res.val_end_date}")

    # Prueba de validación anti-leakage
    assert_zero_temporal_leakage(split_res.train_df, split_res.val_df, split_res.val_start_date)
    print("  [PASS] Auditoría anti-leakage superada: 0 transacciones contaminadas.")

    # Prueba de extracción de Ground Truth
    print("-> Extrayendo Ground Truth de la semana de prueba...")
    gt_dict = build_ground_truth_dict(split_res.val_df, active_only=True)
    print(f"  * Clientes activos en test  : {len(gt_dict):,}")
    sample_c = list(gt_dict.keys())[0]
    print(
        f"  * Ejemplo cliente {sample_c} : {len(gt_dict[sample_c])} artículos comprados -> {gt_dict[sample_c]}"
    )

    # Prueba de asignación de target y grupos para LGBMRanker
    if feat_path.exists():
        print("-> Probando prepare_ranker_split sobre features_matrix.parquet...")
        feat_df = pl.read_parquet(feat_path)

        # Negative downsampling 1:5 para entrenamiento de ranking
        ranker_train = prepare_ranker_split(
            feature_matrix=feat_df,
            ground_truth_df=split_res.val_df,
            negative_ratio=5,
            drop_empty_queries=False,
            seed=RANDOM_SEED,
        )
        print(f"  * Filas preparadas (Train 1:5): {ranker_train.df.height:,}")
        print(f"  * Total de Grupos (Queries)   : {len(ranker_train.groups):,}")
        print(
            f"  * Suma de grupos == Filas     : {sum(ranker_train.groups)} == {ranker_train.df.height} [OK]"
        )
        pos_count = (ranker_train.y == 1).sum()
        print(
            f"  * Positivos etiquetados (y=1) : {pos_count:,} ({pos_count / ranker_train.df.height * 100:.2f}%)"
        )

        # Validación completa sin muestreo (todos los negativos retenidos)
        ranker_val = prepare_ranker_split(
            feature_matrix=feat_df,
            ground_truth_df=split_res.val_df,
            negative_ratio=None,
            drop_empty_queries=False,
        )
        print(f"  * Filas preparadas (Val All)  : {ranker_val.df.height:,}")
        print(
            f"  * Suma de grupos == Filas     : {sum(ranker_val.groups)} == {ranker_val.df.height} [OK]"
        )

    print("=" * 75)
    print("  [OK] INFRAESTRUCTURA DE VALIDACIÓN TEMPORAL CERTIFICADA")
    print("=" * 75)
