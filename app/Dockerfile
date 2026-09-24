# ==============================================================================
# Multi-Stage Hardened Dockerfile — API REST RecSys H&M Fashion
# Optimizado para microcontenedor ultraligero (< 200 MB) y seguridad defensiva
# ==============================================================================

# ------------------------------------------------------------------------------
# ETAPA 1: Builder (Compilación, Resolución y Poda Agresiva de Dependencias)
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

# Herramientas mínimas de compilación y binutils para ejecución de strip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    binutils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalación sin caché en prefijo de usuario
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Poda de artefactos de desarrollo para optimizar huella en disco
RUN find /root/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "*.pyc" -delete 2>/dev/null || true && \
    find /root/.local -name "*.pyo" -delete 2>/dev/null || true && \
    find /root/.local -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true


# ------------------------------------------------------------------------------
# ETAPA 2: Runtime (Microcontenedor Seguro no-root y Endurecido)
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="TFM H&M Fashion Recommender API & Pipeline" \
      org.opencontainers.image.description="Motor de recomendación y pipeline escalable para retail de moda" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Manuel Valdivia"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/home/appuser/.local/bin:$PATH"

# Librerías estrictamente necesarias en tiempo de ejecución:
# - libgomp1: GNU Offloading and OpenMP Runtime Library para LightGBM C++
# - curl: Sonda ligera de salud para Liveness/Readiness probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* /var/cache/debconf/* /var/log/*

# Crear usuario y grupo sin privilegios (UID/GID 1001) para ejecución defensiva
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copiar dependencias de Python optimizadas y podadas
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Crear directorios de montaje para volúmenes y asignar permisos
RUN mkdir -p /app/models /app/data_processed /app/data_sample /app/config /app/src /app/app /app/scripts /app/tests /app/results /app/data && \
    chown -R appuser:appgroup /app

# Copiar código fuente necesario
COPY --chown=appuser:appgroup config/ /app/config/
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup app/ /app/app/

USER appuser
EXPOSE 8000

# Sonda de salud nativa apuntando a /health/live
HEALTHCHECK --interval=20s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-access-log"]
