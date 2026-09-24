"""Módulo de Pipeline Canónico de Recomendación (Arquitectura de 6 Etapas).

Implementa el patrón: Source -> Hydrator -> Filter -> Scorer -> Selector -> SideEffect.
"""

from src.recsys_pipeline.pipeline import (
    CanonicalRecsysPipeline,
    RecsysContext,
    RecsysRequest,
    RecsysResponse,
)

__all__ = [
    "CanonicalRecsysPipeline",
    "RecsysContext",
    "RecsysRequest",
    "RecsysResponse",
]
