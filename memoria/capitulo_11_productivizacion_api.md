# Capítulo 11: Productivización, Despliegue y Gobernanza de Inferencia (FastAPI & Docker)

**Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid**  
**Autor:** Manuel Valdivia  
**Proyecto:** Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)  


## 11.1. Arquitectura de Inferencia en Producción y Requerimientos de Nivel de Servicio (SLOs)

La viabilidad técnica y económica de un sistema de recomendación se define por su capacidad de servir inferencias continuas, reproducibles y de baja latencia bajo cargas concurrentes variables. En plataformas de retail masivo como H&M, una latencia excesiva en la entrega de recomendaciones no solo degrada la experiencia de navegación, sino que induce caídas directas en las tasas de conversión y retención comercial (Kohavi et al., 2020).

Para garantizar una transición robusta desde el modelado analítico offline hacia el entorno operacional, se implementan cuatro principios de MLOps y fiabilidad de sistemas (SRE):

1. **Desacoplamiento Estricto de Ciclos de Vida (Batch vs. Online):** El procesamiento analítico masivo (DuckDB y Polars) se ejecuta fuera de línea de forma asíncrona. El microservicio de inferencia en línea (*online serving*) opera de manera autónoma, consumiendo artefactos precalculados sin competir por ciclos de CPU con las fases de ingesta o entrenamiento.
2. **Definición Cuantitativa de Objetivos de Nivel de Servicio (SLOs):** Se fijan cotas deterministas de rendimiento operacional:
   - Latencia de inferencia interna $p95 < 50\text{ ms}$ para clientes con scoring supervisado.
   - Latencia de inferencia interna $p95 < 5\text{ ms}$ para clientes en arranque en frío (*cold start*).
   - Huella de memoria residente ($\text{RSS}$) acotada a $< 600\text{ MB}$ por proceso en régimen nominal.
   - Disponibilidad del 100% mediante degradación elegante (respuestas HTTP 200 sin excepciones no controladas).
3. **Trazabilidad e Integridad Criptográfica de Artefactos:** Cada versión serializada del modelo clasificador (`lgbm_ranker.txt`) y su configuración (`lgbm_ranker_meta.json`) incorpora sumas de verificación SHA-256 (`checksums.sha256`), garantizando inmutabilidad y paridad estricta entre entrenamiento y producción.
4. **Resiliencia Operativa y Degradación Elegante (*Graceful Degradation*):** Ante identificadores fuera de catálogo, clientes inactivos sin candidatos indexados o fallos de conexión, el sistema activa de forma determinista la cascada de superventas estacionales segmentadas por cohortes de edad.


## 11.2. Arquitectura del Microservicio: FastAPI Asíncrono, ASGI y Patrón Thread-Safe Singleton

El microservicio de inferencia se implementa sobre el framework asíncrono **FastAPI** (`app/main.py`) bajo especificación ASGI sobre Python 3.12, integrando el cargador centralizado `RecommenderServiceLoader` (`app/model_loader.py`).

![Figura 11.1: Runtime de Servicio en Producción con FastAPI](../results/figures/diag_06_fastapi_serving_runtime.png)

*Figura 11.1: Arquitectura de ejecución asíncrona del microservicio FastAPI y desacoplamiento de inferencia en CPU mediante `asyncio.to_thread` con pre-indexación en memoria contigua C-order.*

### 11.2.1. Desacoplamiento CPU-Bound y Gestión de Concurrencia
En servidores asíncronos (`asyncio`), la ejecución directa de cálculos intensivos en CPU (como la evaluación de árboles LightGBM en C++ y el ordenamiento de arrays NumPy) bloquea el hilo principal del *event loop*, impidiendo procesar solicitudes concurrentes de red y serialización JSON.

Para evitar esta degradación, el endpoint principal `/recommend/{customer_id}` delega la inferencia mediante `asyncio.to_thread`:

```python
pred_result = await asyncio.to_thread(
    loader.predict_for_customer,
    customer_id=customer_id, top_k=limit, offset=offset, return_total=True
)
```

Esta instrucción traslada la evaluación del booster al pool de hilos de CPython (`ThreadPoolExecutor`), liberando el bucle de eventos para continuar gestionando tráfico concurrente y permitiendo al servicio sostener rendimientos superiores a 950 peticiones por segundo.

### 11.2.2. Patrón Singleton Thread-Safe y Ciclo de Vida `lifespan`
Para evitar la sobrecarga de instanciación múltiple y garantizar que los pesos del modelo y las matrices de características residan en una única copia por proceso, `RecommenderServiceLoader.get_instance()` implementa el patrón **Singleton con doble comprobación de bloqueo** (`threading.Lock()`).

Asimismo, la deserialización de artefactos se gobierna mediante un gestor de ciclo de vida asíncrono (`@asynccontextmanager lifespan`). De este modo, los datos se pre-cargan e indexan en memoria física antes de que Uvicorn abra el puerto HTTP (8000), erradicando penalizaciones de latencia en la primera consulta (*cold start penalty*).

### 11.2.3. Contratos de Datos con Pydantic v2
La comunicación se rige por contratos estrictamente tipados:
- `RecommendationResponse`: Valida y formatea las prendas como cadenas canónicas de 10 dígitos (`f"{art:010d}"`), reportando latencia en milisegundos (`latency_ms`), versión del modelo y parámetros de paginación (`limit`, `offset`, `count`, `total`).
- `HealthResponse` y `ReadinessResponse`: Exponen el estado de preparación y consumo de memoria RSS.
- Manejo de excepciones uniforme: Errores de validación de entrada emiten código HTTP 422 (`RequestValidationError`) con detalles estructurados sin exponer trazas internas del servidor.


## 11.3. Contenedorización Multi-Etapa y Parámetros Operativos en Docker

Para asegurar portabilidad y aislamiento estricto en cualquier clúster (Kubernetes, AWS ECS, GCP Cloud Run), el microservicio se encapsula en una imagen Docker multi-etapa (`app/Dockerfile`) coordinada por `docker-compose.yml`.

### 11.3.1. Dockerfile Multi-Etapa
1. **Etapa 1 (`builder`):** Sobre `python:3.12-slim` con `build-essential` y `binutils`. Instala dependencias (`requirements.txt`) en el prefijo `--user`, elimina directorios `__pycache__` y ficheros `.pyc`/`.pyo`, y ejecuta `strip --strip-unneeded` sobre librerías compiladas (`*.so`) para descartar tablas de símbolos de depuración.
2. **Etapa 2 (`runtime`):** Imagen limpia `python:3.12-slim` con las librerías estrictas `libgomp1` (soporte OpenMP multihilo para LightGBM C++) y `curl` (sondas de salud). Opera bajo el usuario y grupo sin privilegios `appuser:appgroup` (`UID 1001`, `GID 1001`), restringiendo vectores de escalada en el host. La exclusión en `.dockerignore` reduce el contexto de compilación a **0.21 MB** y genera una imagen final de **~295 MB**.

### 11.3.2. Gobernanza de Recursos en Docker Compose
Los parámetros operacionales configurados en `docker-compose.yml` garantizan aislamiento y estabilidad:

| Parámetro | Configuración Operativa | Justificación Técnica |
| :--- | :---: | :--- |
| **Límites de Cómputo (`limits`)** | `cpus: "4.0"`, `memory: 4.0G` | Cota máxima asignada en cgroups para absorber picos de tráfico concurrente. |
| **Reservas Mínimas (`reservations`)** | `cpus: "0.5"`, `memory: 512M` | Presupuesto basal garantizado para el arranque de Uvicorn y el modelo. |
| **Sonda de Salud (`healthcheck`)** | `curl -f /health/live` (int: 20s, to: 3s) | Certifica que el proceso ASGI responde a señales de red sin bloqueos. |
| **Montaje Volátil (`tmpfs`)** | `/tmp:rw,noexec,nosuid,size=200m` | Impide persistencia de temporales y previene ejecución de binarios arbitrarios. |
| **Concurrencia Uvicorn** | 2 workers (`--workers 2`, `--no-access-log`) | Paralelismo a nivel de proceso sin sobrecarga de I/O en stdout. |

* **Comandos de Gestión del Servicio Docker:**
  - **Construcción y arranque en segundo plano:** `docker compose up -d --build`
  - **Comprobación de estado y salud del contenedor:** `docker compose ps`
  - **Inspección de trazas en tiempo real:** `docker compose logs -f hm-recsys-api`
  - **Detención y desmontaje del servicio:** `docker compose down`

### 11.3.3. Bundle de Artefactos de Producción
La evaluación reproducible desacoplada del dataset bruto (3.5 GB) se realiza mediante el paquete `production_artifacts.zip` (**382.9 MB comprimidos**, ~399 MB en disco), el cual incluye los 14 archivos esenciales (8 Parquets base, 3 Parquets optimizados V8, modelo `lgbm_ranker.txt`, metadatos y sumas SHA-256), aprovisionable mediante `scripts/bootstrap_production_eval.sh`.


## 11.4. Telemetría Empírica, Concurrencia y Verificación de Objetivos (SLOs)

La capacidad de respuesta del microservicio se evaluó mediante `tests/benchmark_api.py`, sometiendo al contenedor a 100 peticiones concurrentes a través de 10 workers en paralelo (distribución de tráfico de producción: 80% clientes registrados con scoring supervisado, 20% clientes desconocidos en arranque en frío).

### Cuadro de Verificación de Objetivos de Nivel de Servicio (SLOs):

| Objetivo Técnico (SLO) | Umbral de Diseño | Rendimiento Registrado (Contenedor) | Estado |
| :--- | :---: | :---: | :---: |
| **Throughput Global (RPS)** | $> 500\text{ req/s}$ | **955.8 req/s** (Lote total completado en 0.105 s) | Cumplido |
| **Latencia Inferencia Clientes Activos ($p95$)** | $< 50.0\text{ ms}$ | **11.79 ms** (End-to-End contenedor) / **2.57 ms** (Booster C++) | Cumplido |
| **Latencia Inferencia Cold Start ($p95$)** | $< 5.0\text{ ms}$ | **0.002 ms** (Servidor) / **1.38 ms** (Cliente directo) | Cumplido |
| **Consumo de Memoria Residente (RSS)** | $< 600.0\text{ MB}$ | **458.9 MB** pico (Presupuesto Docker: 4.0 GB) | Cumplido |
| **Resiliencia ante Fallos de Entrada** | HTTP 200 (Top-12 fallback) | `is_cold_start=True`, 12 artículos retornados sin excepción | Cumplido |
| **Suite de Pruebas Unitarias** | 100% aprobadas | **10/10 tests superados** en `tests/test_api.py` | Cumplido |

### Segmentación de Latencias Internas y Red Exógena
La latencia interna registrada ($11.79\text{ ms}$ en $p95$) refleja estrictamente el tiempo computacional en el contenedor (deserialización Pydantic, unión en memoria contigua y scoring del booster C++). En red pública, deben añadirse las latencias exógenas: tiempo de ida y vuelta (*Round Trip Time* - RTT), negociación criptográfica TLS/SSL, resolución DNS y encolamiento en el balanceador de carga (20–40 ms típicos; Beyer et al., 2016). Mantener la latencia interna por debajo de 12 ms garantiza que el tiempo total percibido por el navegador se mantenga holgadamente inferior a los 60 ms.


## 11.5. Análisis Forense de Memoria: De Heap Fragmentado (11 GB) a Arrays Contiguos (407 MB)

Durante el desarrollo preliminar, el microservicio experimentaba cancelaciones forzadas del kernel (`OOM Killer`, exit code 137) al superar **11.043 MB de memoria residente (RSS)** durante el arranque.

### 11.5.1. Causa Raíz Algorítmica
La implementación inicial empleaba:
```python
df_dict = pl.read_parquet(feat_path).partition_by("customer_idx", as_dict=True)
```
Esta instrucción instanciaba más de **278.000 objetos `polars.DataFrame` independientes**. Cada objeto requería cabeceras estructurales de CPython (`PyObject overhead`), descriptores de columnas y punteros de Apache Arrow, provocando una severa fragmentación del heap que multiplicó la huella física por más de 20 veces.

### 11.5.2. Rediseño Vectorial Continuo
Se rediseñó la carga en `app/model_loader.py` bajo tres directrices:
1. **Matriz Unificada en C-Order:** Extracción de las 39 variables numéricas en una única matriz bidimensional contigua NumPy `float32`, y los identificadores en arrays continuos `int64`.
2. **Particionado en Pase Único:** Detección de fronteras mediante `np.unique` y división de vistas con `np.split`:
   ```python
   u_cust, split_idxs = np.unique(cust_ids, return_index=True)
   for c, x_mat, arts in zip(u_cust, np.split(X_all, split_idxs[1:]), np.split(art_ids, split_idxs[1:])):
       self.candidate_features[int(c)] = (x_mat, [f"{a:010d}" for a in arts])
   del feat_df, X_all; gc.collect()
   ```
3. **Liberación Forzada de Heap:** Eliminación explícita de referencias intermedias y recolección forzada (`gc.collect()`).

**Impacto Cuantitativo:** El tiempo de arranque disminuyó de **183.9 s a 0.31 s (593x más rápido)** y la huella física cayó de **11.043 MB a 407.5 MB RSS (-96.3% de memoria)**.


## 11.6. Catálogo de Endpoints, Especificación Técnica y Comandos Operativos

El microservicio expone un conjunto estructurado de endpoints REST para inferencia en tiempo real, observabilidad y diagnóstico de infraestructura.

### 11.6.1. Especificación del Catálogo de Endpoints

| Método | Endpoint / Ruta | Uso | Parámetros de Entrada | Códigos de Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`POST`** | `/recommend/{customer_id}` | Generación de recomendaciones Top-K personalizadas o fallback estacional por cold start. | `customer_id` (Path, hex 64), `limit` (Query, $[1, 100]$, def: 12), `offset` (Query, $\ge 0$, def: 0) | `200`, `401`, `422`, `500` |
| **`GET`** | `/health` | Diagnóstico de telemetría de memoria RSS, modelo, clientes indexados y catálogo total. | Ninguno | `200` |
| **`GET`** | `/health/live` | Sonda de vida (*Liveness probe*) para orquestadores Docker y Kubernetes. | Ninguno | `200` |
| **`GET`** | `/health/ready` | Sonda de preparación (*Readiness probe*) antes del enrutamiento de tráfico de red. | Ninguno | `200`, `503` |
| **`GET`** | `/metrics` | Exportación de telemetría operacional en formato estándar de Prometheus. | Ninguno | `200` |
| **`GET`** | `/` | Descubrimiento de rutas canónicas de servicio y enlaces a documentación. | Ninguno | `200` |
| **`GET`** | `/docs`, `/redoc` | Interfaces de documentación interactiva Swagger UI y ReDoc (OpenAPI 3.1.0). | Ninguno | `200` |

### 11.6.2. Comandos Operativos y Respuestas del Motor

#### 1. Inferencia Personalizada y Fallback de Contingencia (`POST /recommend/{customer_id}`)
* **Uso:** Evalúa candidatos indexados mediante el booster LightGBM en C++, retornando el Top-K ordenado. Ante clientes desconocidos, activa la cascada de superventas estacionales segmentadas por edad.

* **Comando (Cliente registrado con historial):**
  ```bash
  curl -X POST "http://localhost:8000/recommend/0043d69dcca282763b8db18f967a2d301e69d5966bebd74291c49a0812663223?limit=12&offset=0"
  ```
  *Respuesta representativa (HTTP 200 OK):* `{"customer_id": "0043d6...", "recommendations": ["0929165002", "0751471043", ...], "count": 12, "total": 62, "offset": 0, "limit": 12, "is_cold_start": false, "model_version": "1.0.0", "latency_ms": 2.033}`

* **Comando (Cliente no registrado / Cold start):**
  ```bash
  curl -X POST "http://localhost:8000/recommend/cliente_anonimo_evaluador_2026?limit=12"
  ```
  *Respuesta representativa (HTTP 200 OK):* `{"customer_id": "cliente_anonimo...", "recommendations": ["0924243001", "0924243002", ...], "count": 12, "total": 12, "offset": 0, "limit": 12, "is_cold_start": true, "model_version": "1.0.0", "latency_ms": 0.003}`

#### 2. Sondas de Salud y Telemetría Operativa
* **Comando Sonda de Vida (`GET /health/live`):** `curl http://localhost:8000/health/live`  
  *Uso:* Certifica que el event loop responde sin bloqueos. Retorna: `{"status": "alive"}`.
* **Comando Sonda de Preparación (`GET /health/ready`):** `curl http://localhost:8000/health/ready`  
  *Uso:* Certifica la precarga de modelo y catálogo antes del balanceo. Retorna: `{"ready": true, "model_loaded": true, "catalog_loaded": true}`.
* **Comando Diagnóstico de Memoria (`GET /health`):** `curl http://localhost:8000/health`  
  *Uso:* Inspección de memoria RSS y clientes mapeados. Retorna: `{"status": "ok", "service": "hm_recommender", "model_loaded": true, "indexed_customers": 1678, "catalog_customers": 278275, "memory_rss_mb": 381.1}`.
* **Comando Métricas Prometheus (`GET /metrics`):** `curl http://localhost:8000/metrics`  
  *Uso:* Telemetría de latencias y contadores para Grafana (`recsys_requests_total`, `recsys_latency_p95_ms 11.79`, `recsys_latency_p50_ms 9.95`, `recsys_cold_start_total`).


## 11.7. Modos de Ejecución, Hot-Cache y Gobernanza de Memoria

El cargador de inferencia adapta su estrategia de carga según los recursos asignados y la disponibilidad de almacenamiento:

![Figura 11.2: Aislamiento Operativo de los Modos de Servicio](../results/figures/diag_07_execution_modes_isolation.png)

*Figura 11.2: Aislamiento operacional y segmentación de namespaces de datos entre los tres modos de ejecución del recomendador (Modo 1 Muestra, Modo 2 Producción Estándar y Modo 3 Producción Masivo con cota anti-OOM).*

### 11.7.1. Comparativa Técnica de Modos de Ejecución

| Parámetro / Métrica | Modo 1: Muestra (CI-CD) | Modo 2: Producción Estándar (Por Defecto) | Modo 3: Producción Masivo (Alta Capacidad) |
| :--- | :--- | :--- | :--- |
| **Fuente de Candidatos** | `sample/features_matrix.parquet` | `data_processed/features_matrix.parquet` (top-100k) | `data_processed/features_matrix.parquet` (top-500k) |
| **Clientes en Catálogo / en RAM** | **2,000** / 1,710 usuarios | **278,275** / 1,678 usuarios prioritarios | **278,275** / 8,295 usuarios de alto valor |
| **Presupuesto de RAM (RSS)** | $\approx 280\text{ MB}$ | $\approx 380\text{ MB}$ | $\approx 470 - 700\text{ MB}$ |
| **Inferencia Supervisada** | Activa sobre muestra sintética | Activa para usuarios más frecuentes | Activa para >8,200 usuarios de mayor volumen |
| **Cobertura de Cold Start** | Superventas locales de muestra | Cascada estacional de otoño V8 | Cascada estacional de otoño V8 |
| **Aislamiento Operativo** | Local ágil sin dataset Kaggle | Configuración nominal en `docker-compose.yml` | Modo de alta densidad con salvaguarda SRE |

### 11.7.2. Gobernanza de Hot-Cache y Salvaguarda SRE
La feature matrix bruta (`features_matrix.parquet`) contiene **16.722.720 filas y 39 columnas float32**:

$$\text{Memoria Neta} = 16{,}722{,}720 \text{ filas} \times 39 \text{ columnas} \times 4 \text{ bytes} \approx 2.61 \text{ GB}$$

Sumando punteros, arrays de prendas y árboles de LightGBM, un proceso sin acotar exige entre **3.5 y 4.0 GB RSS**. Con 2 workers concurrentes (`API_WORKERS=2`), la demanda alcanzaría **7.6 GB de RAM**, superando el límite de 4.0 GB fijado en `docker-compose.yml` y provocando la terminación inmediata por `OOM Killer`.

Para evitar este colapso, el sistema implementa una **Arquitectura Hot-Cache de Dos Niveles**:
1. **Nivel 1 (Hot Cache en RAM $O(1)$):** Pre-indexa las 500.000 filas de los **8.295 clientes con mayor volumen transaccional**, resolviendo su scoring en $< 50\text{ ms}$.
2. **Nivel 2 (Fallback Autónomo de Alta Disponibilidad):** Para clientes esporádicos o de cola larga (*long-tail*), el motor activa la cascada V8 en $< 1\text{ ms}$, garantizando un **SLO de disponibilidad del 100%**.
3. **Salvaguarda SRE contra Parámetros No Acotados:** Si un operador configura `API_MAX_INDEXED_ROWS=0` o `auto`, `app/model_loader.py` intercepta el valor y aplica automáticamente el tope seguro de 500.000 filas, garantizando que el contenedor no supere 700 MB de RSS.

### 11.7.3. Comandos de Alternancia Operativa
* **Comando Modo 2 (Producción Estándar - Recomendado):** `docker compose up -d`
* **Comando Modo 3 (Producción Masivo 500k filas):** `API_MAX_INDEXED_ROWS=500000 docker compose up -d` (Linux/Bash) o `$env:API_MAX_INDEXED_ROWS="500000"; docker compose up -d` (PowerShell).
* **Comando Modo 1 (Muestra / Evaluación local):** `TFM_DATA_RAW_DIR=data_sample uvicorn app.main:app --host 0.0.0.0 --port 8000`


## 11.8. Extensiones Arquitectónicas del Microservicio (Wishlist, Sesión, Visión)

La arquitectura modular permite proyectar endpoints adicionales para enriquecer la personalización interactiva sin comprometer el microservicio transaccional:

| Método | Endpoint | Uso | Entrada Principal | Comportamiento del Motor |
| :--- | :--- | :--- | :--- | :--- |
| **`POST`** | `/recommend/wishlist` | Recomendación contextual basada en lista de deseos. | Lista de `article_id` guardados. | Utiliza los artículos como semillas en la matriz de afinidad $P(B|A)$ para cross-selling complementario. |
| **`POST`** | `/recommend/session` | Personalización intra-sesión en tiempo real. | Secuencia cronológica de clics recientes. | Re-pondera dinámicamente los candidatos hacia las secciones con interés activo inmediato. |
| **`POST`** | `/recommend/visual` | Recomendación multimodal asistida por imagen. | Archivo de imagen o URI externa. | Consulta representaciones latentes (*embeddings*) para recuperar prendas estéticamente afines. |

**Principio de Desacoplamiento de Cargas Heterogéneas:** Para preservar la latencia interna ($p95 < 50\text{ ms}$) y el presupuesto de memoria ($< 600\text{ MB}$ RSS), el procesamiento de imágenes o Transformers visuales no debe incorporarse en el proceso de Uvicorn; debe residir en un microservicio auxiliar con aceleración por GPU que entregue vectores latentes normalizados a la API principal.
