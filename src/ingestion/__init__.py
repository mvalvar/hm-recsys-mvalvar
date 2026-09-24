"""Módulo de ingesta y preprocesamiento out-of-core."""

from src.ingestion.create_sample import (
    SampleExtractionResult,
    extract_consistent_sample,
)
from src.ingestion.preprocess import PreprocessSummary, run_preprocess

# Alias de compatibilidad
generate_sample_dataset = extract_consistent_sample

__all__ = [
    "extract_consistent_sample",
    "generate_sample_dataset",
    "SampleExtractionResult",
    "run_preprocess",
    "PreprocessSummary",
]
