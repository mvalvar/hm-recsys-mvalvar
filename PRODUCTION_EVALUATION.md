# Guía de Evaluación Productiva (Modo 2)

Este documento detalla el procedimiento de evaluación del microservicio de recomendación en producción, ejecutando la API REST (FastAPI) sobre los artefactos oficiales preentrenados del modelo V8.

---

## 1. Arquitectura de Distribución de Artefactos

Para habilitar la evaluación sin requerir el reentrenamiento completo del pipeline (que procesa 31.7M de transacciones), los artefactos oficiales se distribuyen como asset de GitHub Release:

* **Release Tag:** `v1.0-production-artifacts`
* **Bundle:** `production_artifacts.zip` (~383 MB comprimido / ~399 MB en disco)
* **Contenido del bundle:**
  - `models/lgbm_ranker.txt`: Pesos del estimador LightGBM entrenado con LambdaRank.
  - `data_processed/customer_id_mapping.parquet`: Mapeo determinista hash hex (64B) a Int32 (4B).
  - `data_processed/articles.parquet`: Catálogo canónico procesado con tipos optimizados.
  - `data_processed/candidates.parquet`: Candidatos multi-heurística consolidados.
  - `data_processed/features_matrix.parquet`: Matriz de 39 variables del Feature Store.
  - Tablas auxiliares de afinidad temporal y cohorte demográfica.

---

## 2. Opción A: Despliegue en Contenedor Docker (Recomendada)

### Paso 1: Descargar y validar artefactos de producción
```bash
bash scripts/bootstrap_production_eval.sh
```
El script descarga automáticamente `production_artifacts.zip` desde la Release oficial de GitHub y extrae los archivos en sus rutas correspondientes (`data_processed/` y `models/`), validando sus hashes de integridad.

### Paso 2: Levantar el contenedor
```bash
docker compose up --build -d
```
El servicio estará disponible en `http://localhost:8000`.

### Paso 3: Ejecutar Smoke Test automático
```bash
bash scripts/smoke_test_api.sh
```

---

## 3. Opción B: Ejecución Local Directa (Python / Uvicorn)

### Paso 1: Validar artefactos
```bash
bash scripts/verify_artifacts.sh
```

### Paso 2: Iniciar servidor Uvicorn
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

### Paso 3: Ejecutar Smoke Test contra puerto local
```bash
bash scripts/smoke_test_api.sh http://127.0.0.1:8010
```

---

## 4. Verificación de Contratos y Acuerdos de Nivel de Servicio (SLAs)

### Sonda de Salud (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
**Respuesta esperada:**
```json
{
  "status": "ok",
  "service": "hm_recommender",
  "model_loaded": true,
  "version": "1.0.0",
  "indexed_customers": 1678,
  "catalog_customers": 278275,
  "memory_rss_mb": 486.6
}
```
*(Nota: En Modo 2 con `API_MAX_INDEXED_ROWS=100000`, `indexed_customers` es 1.678 sobre un catálogo activo de 278.275 clientes, garantizando una huella de memoria < 500 MB).*

### Inferencia de Cliente Activo (`POST /recommend/{customer_id}`)
```bash
curl -X POST http://localhost:8000/recommend/00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657
```
Retorna 12 recomendaciones personalizadas con latencia $p95 < 12$ ms (`is_cold_start = false`).

### Inferencia de Cliente Inactivo / Cold-Start (`POST /recommend/{customer_id}`)
```bash
curl -X POST http://localhost:8000/recommend/0000000000000000000000000000000000000000000000000000000000000000
```
Retorna 12 recomendaciones de popularidad estacional con latencia $< 1$ ms (`is_cold_start = true`).
