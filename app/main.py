"""API REST de Recomendación de Moda en Tiempo Real (FastAPI).

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Arquitectura REST Endurecida:
- Ciclo de vida asíncrono gestionado mediante context manager `lifespan`.
- Inyección del Singleton `RecommenderServiceLoader` para inferencia de baja latencia (< 50 ms p95).
- Endpoints asíncronos (`async def`) desacoplados del loop mediante `asyncio.to_thread` para inferencia C++ no bloqueante.
- Validación estricta de contratos de entrada y salida con Pydantic v2 y soporte de paginación (`limit`, `offset`).
- Soporte de cold-start con degradación a popularidad global (< 5 ms).
- Manejo de errores uniforme con esquemas estructurados y protección contra fuga de trazas internas.
- Configuración flexible de CORS mediante variable de entorno `ALLOWED_ORIGINS`.
- Documentación interactiva OpenAPI/Swagger UI con ejemplos representativos.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from app.model_loader import RecommenderServiceLoader
from app.telemetry import RecsysMetricsCollector

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str | None:
    """Valida la cabecera X-API-Key si el modo protegido está activado por variable de entorno.

    - Si API_KEY_AUTH_ENABLED no está activo (default: False): Acceso libre sin restricciones.
    - Si API_KEY_AUTH_ENABLED está activo (True): Exige que X-API-Key coincida con API_KEY_SECRET.
    """
    auth_enabled_val = os.getenv("API_KEY_AUTH_ENABLED", "false").strip().lower()
    auth_enabled = auth_enabled_val in ("true", "1", "yes", "t", "y")
    if not auth_enabled:
        return None

    expected_secret = os.getenv("API_KEY_SECRET", "")
    if not api_key or (expected_secret and api_key != expected_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticación requerida: API Key inválida o ausente en cabecera 'X-API-Key'.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Gestor de ciclo de vida asíncrono: carga artefactos pesados en arranque."""
    loader = RecommenderServiceLoader.get_instance()
    loader.load_artifacts()
    yield


def create_app() -> FastAPI:
    """Fábrica de la aplicación FastAPI con configuración modular y middlewares."""
    app_instance = FastAPI(
        title="TFM H&M Fashion Recommender API",
        description=(
            "**Motor de Recomendación Escalable para Retail de Moda (Two-Stage RecSys)**\n\n"
            "Desarrollado por Manuel Valdivia como parte del Trabajo Final de Máster en Data Science, "
            "Big Data & Business Analytics de la Universidad Complutense de Madrid (2025 - 2026). "
            "Sistema multi-heurístico de recuperación de candidatos combinado con modelo de re-ranking "
            "supervisado basado en `LightGBM` (`objective='lambdarank'`).\n\n"
            "- **Latencia Operacional:** $p95 < 50$ ms en inferencia personalizada, $< 5$ ms en cold-start.\n"
            "- **Paginación Avanzada:** Soporte de parámetros `limit` y `offset` para navegación fluida.\n"
            "- **Formato de Salida:** Códigos `article_id` con formato canónico de 10 dígitos (ceros a la izquierda).\n"
            "- **Fallback Autónomo:** Detección de arranque en frío con entrega inmediata del Top-12 superventas."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configuración de CORS segura y administrada por variable de entorno
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
    allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if allowed_origins else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    return app_instance


app = create_app()


# CONTRATOS PYDANTIC V2


class HealthResponse(BaseModel):
    """Contrato de respuesta para comprobaciones de salud del servicio."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(default="ok", description="Estado de disponibilidad operativa del servicio")
    service: str = Field(default="hm_recommender", description="Nombre canónico del microservicio")
    model_loaded: bool = Field(
        ..., description="Indica si el booster LightGBM está disponible en memoria"
    )
    version: str = Field(default="1.0.0", description="Versión semántica del API")
    indexed_customers: int = Field(
        default=0, description="Clientes activos con candidatos pre-indexados"
    )
    catalog_customers: int = Field(
        default=0, description="Total de clientes mapeados en el catálogo histórico"
    )
    memory_rss_mb: float = Field(
        default=0.0, description="Consumo actual de memoria residente (RSS) en MB"
    )


class ReadinessResponse(BaseModel):
    """Sonda de preparación (Readiness probe) para orquestadores Kubernetes / Docker."""

    ready: bool = Field(..., description="Indica si el servicio está listo para recibir tráfico")
    model_loaded: bool = Field(..., description="Estado de carga del modelo en memoria")
    catalog_loaded: bool = Field(..., description="Estado de carga del catálogo de clientes")


class RecommendationResponse(BaseModel):
    """Contrato formal de respuesta para recomendaciones personalizadas con paginación."""

    model_config = ConfigDict(extra="ignore")

    customer_id: str = Field(
        ...,
        description="Identificador único del cliente (hash hexadecimal de 64 caracteres)",
        examples=["006b797de41be777b46f123bda0f66e4fa67817983afd953e153ebb0bd4a87ae"],
    )
    recommendations: list[str] = Field(
        ...,
        description="Lista ordenada de article_id recomendados (strings canónicos de 10 dígitos)",
        examples=[
            [
                "0706016001",
                "0706016003",
                "0448509014",
                "0751471001",
                "0751471043",
                "0896152002",
                "0918292001",
                "0915526001",
                "0863595006",
                "0898694001",
                "0916468003",
                "0714790020",
            ]
        ],
    )
    count: int = Field(
        default=12,
        description="Número total de recomendaciones devueltas en la respuesta actual",
        examples=[12],
    )
    total: int = Field(
        default=12,
        description="Total de candidatos viables generados para este usuario",
        examples=[100],
    )
    offset: int = Field(
        default=0,
        description="Desplazamiento aplicado en la paginación",
        examples=[0],
    )
    limit: int = Field(
        default=12,
        description="Límite máximo de artículos solicitados por página",
        examples=[12],
    )
    is_cold_start: bool = Field(
        ...,
        description="Flag booleano que indica si se aplicó recomendación fallback de arranque en frío",
        examples=[False],
    )
    model_version: str = Field(
        default="1.0.0",
        description="Versión del modelo de inferencia utilizado",
        examples=["1.0.0"],
    )
    latency_ms: float = Field(
        ...,
        description="Tiempo de inferencia registrado en milisegundos para telemetría",
        examples=[0.75],
    )


class ErrorDetail(BaseModel):
    """Estructura normalizada para detalles de error de validación u operacionales."""

    message: str
    code: str = "ERROR"
    params: dict[str, Any] | None = None


# MANEJADORES UNIFORMES DE EXCEPCIONES


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Manejo uniforme de errores de validación de entrada (HTTP 422)."""
    errors = exc.errors()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": errors,
            "message": "Error de validación en los parámetros de la solicitud.",
            "error_code": "UNPROCESSABLE_ENTITY",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Manejo uniforme de excepciones HTTP explícitas."""
    headers = getattr(exc, "headers", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
        },
        headers=headers,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Captura defensiva de errores inesperados evitando fuga de trazas internas (HTTP 500)."""
    logger.error(f"Error no controlado en el servidor: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor durante el proceso de inferencia.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


# RUTAS DE NAVEGACIÓN Y MONITOREO


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    """Ruta raíz de bienvenida con enlaces de navegación del servicio."""
    return {
        "service": "TFM H&M Fashion Recommender API",
        "docs_url": "/docs",
        "health_url": "/health",
        "recommend_endpoint": "POST /recommend/{customer_id}",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoreo"],
    summary="Comprobación de Salud y Telemetría de Memoria",
)
async def health_check() -> HealthResponse:
    """Endpoint de comprobación de salud (Liveness / Readiness probe) para balanceadores y Kubernetes."""
    loader = RecommenderServiceLoader.get_instance()
    health_data = loader.get_health_status()
    return HealthResponse(**health_data)


@app.get(
    "/health/live",
    tags=["Monitoreo"],
    summary="Sonda de Vida (Liveness Probe)",
)
async def health_live() -> dict[str, str]:
    """Confirma que el proceso ASGI está activo y respondiendo."""
    return {"status": "alive"}


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["Monitoreo"],
    summary="Sonda de Preparación (Readiness Probe)",
)
async def health_ready() -> ReadinessResponse:
    """Verifica que el modelo y los mapeos en memoria están listos para inferir tráfico."""
    loader = RecommenderServiceLoader.get_instance()
    is_ready = loader.model is not None and len(loader.customer_mapping) > 0
    return ReadinessResponse(
        ready=is_ready,
        model_loaded=loader.model is not None,
        catalog_loaded=len(loader.customer_mapping) > 0,
    )


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["Monitoreo"],
    summary="Métricas de Telemetría Prometheus",
)
async def get_metrics() -> PlainTextResponse:
    """Exporta métricas de latencia y volumen de peticiones en formato compatible con Prometheus."""
    collector = RecsysMetricsCollector.get_instance()
    return PlainTextResponse(
        content=collector.export_prometheus_format(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ENDPOINT PRINCIPAL DE INFERENCIA Y RECOMENDACIÓN


@app.post(
    "/recommend/{customer_id}",
    response_model=RecommendationResponse,
    tags=["Recomendaciones"],
    summary="Generar Recomendaciones Top-K Paginadas en Tiempo Real",
)
async def get_recommendations(
    customer_id: str = Path(
        ...,
        min_length=10,
        max_length=64,
        description="Hash hexadecimal del cliente registrado o anónimo",
        examples=["006b797de41be777b46f123bda0f66e4fa67817983afd953e153ebb0bd4a87ae"],
    ),
    limit: int = Query(
        default=12,
        ge=1,
        le=100,
        description="Número de artículos recomendados a devolver en esta página (default 12)",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Desplazamiento para paginar la lista de recomendaciones (default 0)",
    ),
    _api_key: str | None = Depends(verify_api_key),
) -> RecommendationResponse:
    r"""Genera los artículos personalizados para el cliente solicitado con soporte de paginación.

    - **Cliente en Catálogo con Candidatos:** Ejecuta scoring en C++ con `LGBMRanker`,
      ordena de forma descendente y retorna los mejores ítems con `is_cold_start=False` ($< 50$ ms).
    - **Cliente Desconocido o Sin Candidatos (Cold Start):** Retorna inmediatamente el
      vector de popularidad global reciente con `is_cold_start=True` ($< 5$ ms).
    - **Ejecución Asíncrona:** Desacoplada a hilo secundario con `asyncio.to_thread` para
      evitar bloquear el bucle de eventos durante la inferencia intensiva de CPU.
    """
    loader = RecommenderServiceLoader.get_instance()
    metrics_collector = RecsysMetricsCollector.get_instance()

    try:
        # Desacoplamiento asíncrono para inferencia de CPU no bloqueante
        pred_result = await asyncio.to_thread(
            loader.predict_for_customer,
            customer_id=customer_id,
            top_k=limit,
            offset=offset,
            return_total=True,
        )
        recs, is_cold, latency, total_cand = pred_result
        metrics_collector.record_request(latency_ms=latency, is_cold_start=is_cold)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        metrics_collector.record_request(latency_ms=0.0, is_cold_start=False, is_error=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la inferencia: {str(exc)}",
        ) from exc

    return RecommendationResponse(
        customer_id=customer_id,
        recommendations=recs,
        count=len(recs),
        total=total_cand,
        offset=offset,
        limit=limit,
        is_cold_start=is_cold,
        model_version=loader.model_version,
        latency_ms=round(latency, 3),
    )
