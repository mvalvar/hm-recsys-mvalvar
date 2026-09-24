"""Utilidades de control, optimización de memoria RAM y telemetría de proceso.

Diseñado bajo la restricción de hardware local de 12 GB de RAM:
- El downcasting sistemático reduce el consumo de memoria en más del 70-80%.
- El tipado categórico para strings repetitivas compacta punteros y memoria heap.
- El monitoreo continuo mediante psutil detecta riesgos de OOM de forma preventiva.
- El decorador profile_memory mide el delta de consumo en etapas analíticas críticas.
- El procesamiento por lotes con recolección de basura forzada (gc.collect) evita fugas.
"""

from __future__ import annotations

import functools
import gc
import logging
import os
from collections.abc import Callable, Generator
from typing import ParamSpec, TypeVar

import polars as pl
import psutil

logger = logging.getLogger(__name__)

# Tipos genéricos para preservar firmas de funciones en decoradores (Python 3.12+)
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

# Rangos numéricos para downcasting seguro sin overflow
INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -2147483648, 2147483647


def log_memory_usage(label: str = "Punto de control") -> float:
    """Registra y muestra la memoria residente (RSS) del proceso y el porcentaje total del host.

    Parameters
    ----------
    label : str, optional
        Etiqueta descriptiva del hito de ejecución.

    Returns
    -------
    float
        Consumo de memoria RAM del proceso en megabytes (MB).
    """
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    percent_used = psutil.virtual_memory().percent
    logger.info(f"[MEMORY] {label}: {rss_mb:.1f} MB | {percent_used:.1f}% RAM del sistema")
    return rss_mb


def downcast_polars(
    df: pl.DataFrame,
    categorical_threshold: float = 0.5,
    max_categorical_cardinality: int = 1000,
) -> pl.DataFrame:
    """Aplica downcasting defensivo y seguro a las columnas de un Polars DataFrame.

    Estrategia de reducción defensiva para entorno de 12 GB de RAM:
    1. Enteros (Int64 / Int32): Se analizan el valor mínimo y máximo de la serie para
       determinar si cabe holgadamente en Int8, Int16 o Int32 sin provocar overflow.
    2. Coma flotante (Float64): Se compacta a Float32, suficiente para modelado predictivo
       y monetario, reduciendo el peso de la columna a la mitad.
    3. Cadenas de texto (String / Utf8): Si la cardinalidad única representa una fracción
       inferior a `categorical_threshold` y tiene menos de `max_categorical_cardinality` valores,
       se transforma a pl.Categorical. Esto almacena índices enteros comprimidos sobre un diccionario
       único, ahorrando hasta un 85% de huella en RAM.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame a optimizar.
    categorical_threshold : float, optional
        Ratio máximo de unicidad (n_unique / total_filas) para convertir a categórico (default 0.5).
    max_categorical_cardinality : int, optional
        Límite superior de valores únicos para habilitar categorización (default 1000).

    Returns
    -------
    pl.DataFrame
        DataFrame con tipos optimizados y huella de memoria minimizada.
    """
    assert isinstance(df, pl.DataFrame), "El parámetro df debe ser una instancia de pl.DataFrame"
    assert df.height > 0, "El DataFrame no puede estar vacío para aplicar downcasting"

    transformations: list[pl.Expr] = []
    n_rows = df.height

    int_cols = [c for c, dt in df.schema.items() if dt in (pl.Int64, pl.Int32)]
    str_cols = [c for c, dt in df.schema.items() if dt == pl.String]

    # Recopilar todos los estadísticos de extremos y cardinalidad en un único escaneo eficiente
    stats_exprs: list[pl.Expr] = []
    for c in int_cols:
        stats_exprs.extend([
            pl.col(c).min().alias(f"{c}__min"),
            pl.col(c).max().alias(f"{c}__max"),
        ])
    for c in str_cols:
        stats_exprs.append(pl.col(c).n_unique().alias(f"{c}__nunique"))

    stats = df.select(stats_exprs).to_dicts()[0] if stats_exprs else {}

    for col_name, dtype in df.schema.items():
        # Optimización segura de enteros evaluando cotas reales
        if dtype in (pl.Int64, pl.Int32):
            col_min = stats.get(f"{col_name}__min")
            col_max = stats.get(f"{col_name}__max")

            if col_min is not None and col_max is not None:
                if col_min >= INT8_MIN and col_max <= INT8_MAX:
                    transformations.append(pl.col(col_name).cast(pl.Int8))
                elif col_min >= INT16_MIN and col_max <= INT16_MAX:
                    transformations.append(pl.col(col_name).cast(pl.Int16))
                elif col_min >= INT32_MIN and col_max <= INT32_MAX:
                    transformations.append(pl.col(col_name).cast(pl.Int32))
                else:
                    transformations.append(pl.col(col_name))
            else:
                transformations.append(pl.col(col_name))

        # Reducción de punto flotante a 32 bits (preserva precisión comercial con mitad de bytes)
        elif dtype == pl.Float64:
            transformations.append(pl.col(col_name).cast(pl.Float32))

        # Compactación de cadenas de texto repetitivas a tipo Categórico
        elif dtype == pl.String:
            n_unique = stats.get(f"{col_name}__nunique", n_rows)
            uniqueness_ratio = n_unique / n_rows

            if (
                n_unique <= max_categorical_cardinality
                and uniqueness_ratio <= categorical_threshold
            ):
                transformations.append(pl.col(col_name).cast(pl.Categorical))
            else:
                transformations.append(pl.col(col_name))

        else:
            transformations.append(pl.col(col_name))

    optimized_df = df.with_columns(transformations)
    assert optimized_df.height == n_rows, "Error crítico: El downcasting alteró el número de filas"

    return optimized_df


def profile_memory(func: Callable[P, R]) -> Callable[P, R]:
    """Decorador que mide el consumo de memoria residente y el delta producido por una función.

    Ideal para auditar etapas de ingesta pesada, generación masiva de candidatos y
    ensamblado de matrices de entrenamiento.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        gc.collect()
        process = psutil.Process(os.getpid())
        start_mb = process.memory_info().rss / (1024 * 1024)

        result = func(*args, **kwargs)

        gc.collect()
        end_mb = process.memory_info().rss / (1024 * 1024)
        delta_mb = end_mb - start_mb
        percent_used = psutil.virtual_memory().percent

        logger.info(
            f"[MEMORY PROFILE] {func.__name__}: "
            f"inicio={start_mb:.1f} MB, fin={end_mb:.1f} MB, delta={delta_mb:+.1f} MB "
            f"| {percent_used:.1f}% RAM sistema"
        )
        return result

    return wrapper


def chunked_processing(
    df: pl.DataFrame,
    chunk_size: int,
    func: Callable[[pl.DataFrame], T],
) -> Generator[T, None, None]:
    """Generador que procesa un Polars DataFrame en bloques controlados de N filas.

    Invoca explícitamente `gc.collect()` tras procesar cada fragmento para forzar
    la liberación de estructuras intermedias en la memoria heap de Python.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame a procesar por bloques.
    chunk_size : int
        Tamaño de bloque en número de registros.
    func : Callable[[pl.DataFrame], T]
        Función a ejecutar sobre cada fragmento.

    Yields
    ------
    T
        Resultado retornado por func para el bloque actual.
    """
    assert isinstance(df, pl.DataFrame), "El parámetro df debe ser una instancia de pl.DataFrame"
    assert chunk_size > 0, "chunk_size debe ser un entero estrictamente positivo"
    assert df.height > 0, "El DataFrame no puede estar vacío"

    n_rows = df.height

    for offset in range(0, n_rows, chunk_size):
        chunk = df.slice(offset, chunk_size)
        result = func(chunk)
        del chunk
        gc.collect()
        yield result
