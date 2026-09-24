"""Suite de Pruebas Unitarias para src/utils/memory.py (Downcasting y Telemetría).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.memory import (
    chunked_processing,
    downcast_polars,
    log_memory_usage,
    profile_memory,
)


def test_log_memory_usage():
    """Valida la telemetría de memoria del proceso en ejecución."""
    rss = log_memory_usage("Test checkpoint")
    assert isinstance(rss, float)
    assert rss > 0.0


def test_downcast_polars_integers_and_floats():
    """Valida el downcast óptimo en una sola pasada para enteros y flotantes."""
    df = pl.DataFrame(
        {
            "small_int": [1, 2, 10, 50, -20],              "medium_int": [500, 1000, -2000, 15000, 0],              "large_int": [100000, -500000, 200000, 0, 10],              "float_col": [1.25, 2.50, 3.75, 4.00, 5.50],              "constant_str": ["A", "A", "B", "A", "B"],              "unique_str": ["id_1", "id_2", "id_3", "id_4", "id_5"],  # Alta cardinalidad -> Permanece String
        }
    )

    opt_df = downcast_polars(df, categorical_threshold=0.6, max_categorical_cardinality=10)

    assert opt_df.schema["small_int"] == pl.Int8
    assert opt_df.schema["medium_int"] == pl.Int16
    assert opt_df.schema["large_int"] == pl.Int32
    assert opt_df.schema["float_col"] == pl.Float32
    assert opt_df.schema["constant_str"] == pl.Categorical
    assert opt_df.schema["unique_str"] == pl.String
    assert opt_df.height == df.height


def test_profile_memory_decorator():
    """Valida el decorador de perfilado de memoria."""
    @profile_memory
    def dummy_compute(n: int) -> int:
        data = [i * 2 for i in range(n)]
        return len(data)

    res = dummy_compute(10_000)
    assert res == 10_000


def test_chunked_processing():
    """Valida el procesamiento en bloques controlados con Polars."""
    df = pl.DataFrame({"idx": list(range(25)), "val": list(range(25))})

    processed_chunks = list(chunked_processing(df, chunk_size=10, func=lambda c: c.height))
    assert processed_chunks == [10, 10, 5]
