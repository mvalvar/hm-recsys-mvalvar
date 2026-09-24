"""Configuración centralizada y constantes globales del proyecto TFM.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
Título: Motor de Recomendación Escalable para Retail de Moda (H&M Dataset)
Autor: Manuel Valdivia

Este módulo es la ÚNICA fuente de verdad arquitectónica del proyecto:
1. Jerarquía inmutable del sistema de archivos mediante `pathlib.Path`.
2. Parámetros temporales con cero fugas al futuro (Zero-Leakage Temporal Scheme).
3. Parámetros de generación de candidatos para las 8 heurísticas de recall.
4. Hiperparámetros óptimos de LGBMRanker adaptados de la solución del 6° puesto en Kaggle.
5. Flag dinámico de entorno/CLI para alternar entre evaluación ágil (--sample) y producción masiva.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Final

# Filesystem Hierarchy

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Ruta a los datos crudos originales de H&M (31.7M transacciones, 3.49 GB)
# Jerarquía de resolución: Variable TFM_DATA_RAW_DIR -> data/ local por defecto
_local_raw = BASE_DIR / "data"
if os.environ.get("TFM_DATA_RAW_DIR"):
    DATA_RAW_DIR: Final[Path] = Path(os.environ["TFM_DATA_RAW_DIR"])
else:
    DATA_RAW_DIR: Final[Path] = _local_raw

# Muestra estratificada versionada en Git para pruebas ágiles (<30s)
DATA_SAMPLE_DIR: Final[Path] = BASE_DIR / "data_sample"

# Checkpoints intermedios en formato Parquet ZSTD (lectura columnar out-of-core)
DATA_PROCESSED_DIR: Final[Path] = BASE_DIR / "data_processed"

# Checkpoints intermedios del modo muestra (--sample) aislados para evitar colisiones
DATA_PROCESSED_SAMPLE_DIR: Final[Path] = DATA_PROCESSED_DIR / "sample"

# Directorio de modelos serializados (.txt nativo de LightGBM, mapeos indexados)
MODELS_DIR: Final[Path] = BASE_DIR / "models"

# Modelos serializados del modo muestra (--sample)
MODELS_SAMPLE_DIR: Final[Path] = MODELS_DIR / "sample"

# Salidas analíticas, tablas de métricas y gráficos del EDA / XAI
RESULTS_DIR: Final[Path] = BASE_DIR / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"

# Capítulos de la memoria académica del TFM
MEMORIA_DIR: Final[Path] = BASE_DIR / "memoria"

# Control defensivo y aseguramiento de directorios de trabajo
for directory in [
    DATA_PROCESSED_DIR,
    DATA_PROCESSED_SAMPLE_DIR,
    MODELS_DIR,
    MODELS_SAMPLE_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    DATA_SAMPLE_DIR,
]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        # En entornos de contenedor con montajes de solo lectura (:ro), ignorar si no se puede crear
        pass

# Validación defensiva de la ruta de datos crudos
if not DATA_RAW_DIR.exists():
    warnings.warn(
        f"[AVISO DE RUTAS] El directorio de datos crudos '{DATA_RAW_DIR}' no existe en este entorno. "
        f"Asegúrese de activar USE_SAMPLE_DATA=True para ejecutar en modo muestra.",
        category=ResourceWarning,
        stacklevel=2,
    )


# Modo de evaluación y selección de datos


def _resolve_sample_flag() -> bool:
    """Resuelve si el modo muestra está activo mediante CLI (--sample) o variable de entorno."""
    env_val = os.getenv("USE_SAMPLE_DATA", "false").strip().lower()
    env_active = env_val in ("true", "1", "yes", "t", "y")
    cli_active = "--sample" in sys.argv
    return env_active or cli_active


# Flag global para conmutar transparentemente entre la muestra ligera y el dataset masivo
USE_SAMPLE_DATA: bool = _resolve_sample_flag()


def get_active_data_dir() -> Path:
    """Retorna el directorio de datos activo en función de USE_SAMPLE_DATA."""
    return DATA_SAMPLE_DIR if USE_SAMPLE_DATA else DATA_RAW_DIR


def get_processed_dir(use_sample: bool | None = None) -> Path:
    """Retorna el directorio de datos procesados según el modo operativo.

    Si use_sample es True, retorna DATA_PROCESSED_SAMPLE_DIR (aislamiento de Modo 1).
    Si use_sample es False, retorna DATA_PROCESSED_DIR (producción masiva para Modo 2 y 3).
    Si es None, consulta el flag global USE_SAMPLE_DATA.
    """
    active = USE_SAMPLE_DATA if use_sample is None else use_sample
    return DATA_PROCESSED_SAMPLE_DIR if active else DATA_PROCESSED_DIR


def get_models_dir(use_sample: bool | None = None) -> Path:
    """Retorna el directorio de modelos serializados según el modo operativo.

    Si use_sample es True, retorna MODELS_SAMPLE_DIR (aislamiento de Modo 1).
    Si use_sample es False, retorna MODELS_DIR (producción oficial para Modo 2 y 3).
    Si es None, consulta el flag global USE_SAMPLE_DATA.
    """
    active = USE_SAMPLE_DATA if use_sample is None else use_sample
    return MODELS_SAMPLE_DIR if active else MODELS_DIR


# Enlaces externos y reproducibilidad

KAGGLE_COMPETITION_URL: Final[str] = (
    "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations"
)
KAGGLE_DATA_URL: Final[str] = f"{KAGGLE_COMPETITION_URL}/data"


# Parámetros temporales (Zero-Leakage y estacionalidad)

# Semanas de historia reciente para entrenamiento y recall (ciclo de rotación en moda rápida)
TEMPORAL_WINDOW_WEEKS: Final[int] = 5

# Días de validación retenidos (Semana 104 del dataset; emulación de test Kaggle)
VAL_WINDOW_DAYS: Final[int] = 7


# Generación de candidatos (Fase Recall - 8 Heurísticas)

# Cuota máxima global de candidatos por usuario (Recall@100 > 75% con reducción de espacio)
N_CANDIDATES_PER_USER: Final[int] = 100

# Cuotas máximas por heurística para balancear precisión histórica, tendencias y diversidad
REPURCHASE_TOP_K: Final[int] = 24  # R1: Recompra reciente del usuario
GLOBAL_POPULAR_TOP_K: Final[int] = 20  # R2: Superventas globales con decaimiento temporal
AGE_POPULAR_TOP_K: Final[int] = 15  # R3: Artículos líderes según cohorte generacional
CHANNEL_POPULAR_TOP_K: Final[int] = (
    15  # R4: Preferencia de canal de compra (tienda física vs. online)
)
ITEM_CF_TOP_K: Final[int] = 20  # R5: Co-compras cruzadas en cesta / semana (outfits)
PRODUCT_FAMILY_TOP_K: Final[int] = 10  # R6: Artículos de los departamentos habituales del cliente
TRENDING_TOP_K: Final[int] = 10  # R7: Artículos con mayor aceleración relativa de demanda
USER_DEPT_POPULAR_TOP_K: Final[int] = (
    12  # R8: Artículos líderes en la última semana en el departamento favorito
)

# Factor de decaimiento exponencial temporal: w(t) = exp(-lambda * delta_t_weeks), vida media 6.9 sem
LAMBDA_DECAY: Final[float] = 0.1

# Segmentación etaria demográfica (5 cohortes generacionales estándar en retail)
# [0, 25): Generación Z / Estudiantes
# [25, 35): Jóvenes profesionales / Adultos jóvenes
# [35, 45): Adultos medios
# [45, 55): Adultos maduros
# [55, 120]: Senior / Tercera edad
AGE_BINS: Final[list[int]] = [0, 25, 35, 45, 55, 120]
AGE_BIN_LABELS: Final[list[str]] = ["<25", "25-34", "35-44", "45-54", "55+"]


# Re-ranking e hiperparámetros LGBMRanker (Learning to Rank)

# Ratio de ejemplos negativos por cada positivo en entrenamiento (1:5 para control de RAM <12 GB)
NEGATIVE_RATIO: Final[int] = 5

# Métrica de evaluación oficial del concurso y TFM
METRIC_EVAL_K: Final[int] = 12

# Hiperparámetros calibrados para optimización LambdaRank
LGBM_PARAMS: Final[dict[str, Any]] = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [12],
    "boosting_type": "gbdt",
    "n_estimators": 50,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "importance_type": "gain",
    "random_state": 42,
    "n_jobs": 4,
    "verbose": -1,
}

# Hiperparámetros reducidos para validación instantánea en modo muestra (--sample)
LGBM_SAMPLE_PARAMS: Final[dict[str, Any]] = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [12],
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 5,
    "n_estimators": 60,
    "importance_type": "gain",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# Determinismo y configuración de servicio web

# Semilla fija para reproducibilidad universal estricta en NumPy, LightGBM y Polars
RANDOM_SEED: Final[int] = 42

# Configuración de inferencia REST en producción (FastAPI)
API_HOST: Final[str] = os.getenv("API_HOST", "0.0.0.0")
API_PORT: Final[int] = int(os.getenv("API_PORT", "8000"))
API_WORKERS: Final[int] = int(os.getenv("API_WORKERS", "1"))
MODEL_ARTIFACT_NAME: Final[str] = "lgbm_ranker.txt"
CUSTOMER_MAPPING_NAME: Final[str] = "customer_id_mapping.parquet"
ARTICLE_MAPPING_NAME: Final[str] = "articles.parquet"


# Kaggle evaluation y submission constraints
KAGGLE_TOTAL_CUSTOMERS: Final[int] = 1_371_980
KAGGLE_TOP_K: Final[int] = 12
SUBMISSION_FILE_NAME: Final[str] = "submission.csv"
SUBMISSION_GZIP_NAME: Final[str] = "submission.csv.gz"
