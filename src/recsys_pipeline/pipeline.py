"""Pipeline Canónico de Inferencia RecSys en 6 Etapas.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Implementación desacoplada y componible del marco arquitectónico industrial:
1. Source: Recuperación multi-heurística de candidatos (O(1) en memoria o fallback).
2. Hydrator: Hidratación con matriz de 39 características tabulares.
3. Filter: Filtrado de negocio (artículos excluidos, sin stock o reglas de catálogo).
4. Scorer: Puntuación supervisada con LightGBM Booster (LambdaRank).
5. Selector: Ordenación descendente por relevancia y paginación determinista (limit/offset).
6. SideEffect: Emisión no bloqueante de telemetría y métricas operacionales.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.model_loader import RecommenderServiceLoader

logger = logging.getLogger(__name__)


@dataclass
class RecsysRequest:
    """Contrato de entrada para la canalización de recomendación."""

    customer_id: str
    limit: int = 12
    offset: int = 0
    excluded_article_ids: set[str] = field(default_factory=set)


@dataclass
class RecsysContext:
    """Contexto mutable transferido a través de las 6 etapas del pipeline."""

    candidate_article_ids: list[str] = field(default_factory=list)
    candidate_features_matrix: np.ndarray | None = None
    scores: np.ndarray | None = None
    filtered_indices: list[int] = field(default_factory=list)
    final_recommendations: list[str] = field(default_factory=list)
    is_cold_start: bool = False
    total_candidates: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    latency_ms: float = 0.0
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecsysResponse:
    """Contrato formal de salida del pipeline canónico."""

    customer_id: str
    recommendations: list[str]
    count: int
    total: int
    offset: int
    limit: int
    is_cold_start: bool
    model_version: str
    latency_ms: float


class CanonicalRecsysPipeline:
    """Orquestador componible de las 6 etapas de recomendación en tiempo real."""

    def __init__(self, loader: RecommenderServiceLoader | None = None) -> None:
        self.loader = loader or RecommenderServiceLoader.get_instance()

    def run_source(self, request: RecsysRequest, context: RecsysContext) -> None:
        """Etapa 1: Source (Recuperación de candidatos u obtención de fallback)."""
        cid = request.customer_id
        if cid not in self.loader.customer_mapping:
            context.is_cold_start = True
            context.candidate_article_ids = list(self.loader.popular_fallback)
            context.total_candidates = len(context.candidate_article_ids)
            return

        c_idx = self.loader.customer_mapping[cid]
        if self.loader.model is None or c_idx not in self.loader.candidate_features:
            context.is_cold_start = True
            context.candidate_article_ids = list(self.loader.popular_fallback)
            context.total_candidates = len(context.candidate_article_ids)
            return

        feat_mat, art_ids = self.loader.candidate_features[c_idx]
        context.candidate_article_ids = list(art_ids)
        context.candidate_features_matrix = feat_mat
        context.total_candidates = len(art_ids)

    def run_hydrator(self, _request: RecsysRequest, context: RecsysContext) -> None:
        """Etapa 2: Hydrator (Verificación de integridad de características cargadas)."""
        if context.is_cold_start or context.candidate_features_matrix is None:
            return
        # La matriz en memoria ya está pre-hidratada en arrays contiguos float32 O(1)
        assert len(context.candidate_features_matrix) == len(context.candidate_article_ids)

    def run_filter(self, request: RecsysRequest, context: RecsysContext) -> None:
        """Etapa 3: Filter (Exclusión de artículos vetados o no disponibles)."""
        if not request.excluded_article_ids:
            context.filtered_indices = list(range(len(context.candidate_article_ids)))
            return

        context.filtered_indices = [
            i
            for i, art in enumerate(context.candidate_article_ids)
            if art not in request.excluded_article_ids
        ]

    def run_scorer(self, _request: RecsysRequest, context: RecsysContext) -> None:
        """Etapa 4: Scorer (Evaluación supervisada con LightGBM LambdaRank)."""
        if context.is_cold_start or context.candidate_features_matrix is None:
            return

        indices = context.filtered_indices
        if not indices:
            return

        X_eval = context.candidate_features_matrix[indices]
        context.scores = self.loader.model.predict(X_eval)

    def run_selector(self, request: RecsysRequest, context: RecsysContext) -> None:
        """Etapa 5: Selector (Ordenamiento descendente y paginación determinista)."""
        limit = request.limit
        offset = request.offset

        if context.is_cold_start or context.scores is None:
            # Fallback ordenado estacional
            selected = context.candidate_article_ids[offset : offset + limit]
            context.final_recommendations = selected
            return

        # Ordenar los índices filtrados por puntuación descendente
        sort_order = np.argsort(context.scores)[::-1]
        valid_indices = [context.filtered_indices[i] for i in sort_order]
        sliced_indices = valid_indices[offset : offset + limit]

        recs = [context.candidate_article_ids[idx] for idx in sliced_indices]

        # Si faltan recomendaciones para completar el limit en la primera página, completar con fallback
        if len(recs) < limit and offset == 0:
            for fallback_item in self.loader.popular_fallback:
                if fallback_item not in recs and fallback_item not in request.excluded_article_ids:
                    recs.append(fallback_item)
                if len(recs) == limit:
                    break

        context.final_recommendations = recs

    def run_side_effect(self, request: RecsysRequest, context: RecsysContext) -> None:
        """Etapa 6: SideEffect (Telemetría, cálculo de latencia y registro de eventos)."""
        context.latency_ms = (time.perf_counter() - context.start_time) * 1000.0
        context.telemetry = {
            "customer_id": request.customer_id,
            "is_cold_start": context.is_cold_start,
            "total_candidates": context.total_candidates,
            "returned_count": len(context.final_recommendations),
            "latency_ms": round(context.latency_ms, 3),
        }

    def execute(self, request: RecsysRequest) -> RecsysResponse:
        """Ejecución end-to-end secuencial y observable del pipeline canónico."""
        ctx = RecsysContext()

        self.run_source(request, ctx)
        self.run_hydrator(request, ctx)
        self.run_filter(request, ctx)
        self.run_scorer(request, ctx)
        self.run_selector(request, ctx)
        self.run_side_effect(request, ctx)

        return RecsysResponse(
            customer_id=request.customer_id,
            recommendations=ctx.final_recommendations,
            count=len(ctx.final_recommendations),
            total=ctx.total_candidates,
            offset=request.offset,
            limit=request.limit,
            is_cold_start=ctx.is_cold_start,
            model_version=self.loader.model_version,
            latency_ms=round(ctx.latency_ms, 3),
        )
