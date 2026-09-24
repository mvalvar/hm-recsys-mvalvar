# Capítulo 12: Análisis de Recursos de Infraestructura y Costes de Servidor

**Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid**  
**Autor:** Manuel Valdivia  
**Proyecto:** Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)  


## 12.1. Marco Metodológico: FinOps y Total Cost of Ownership (TCO) en Sistemas de Recomendación

En entornos de comercio electrónico a gran escala, la evaluación de un sistema de recomendación no puede limitarse de forma aislada a métricas de precisión analítica ($MAP@12$, $Recall@k$). Un modelo de alto desempeño predictivo resulta inviable si su complejidad computacional impone costes de infraestructura desproporcionados o latencias de inferencia que violen los acuerdos de nivel de servicio (Kohavi et al., 2020).

La disciplina de ingeniería financiera en la nube (*FinOps*) establece que el diseño de arquitecturas de aprendizaje automático debe evaluarse bajo el principio de optimización multiobjetivo del **Coste Total de Propiedad (*Total Cost of Ownership*, TCO)**. Esto exige analizar el compromiso entre tres variables cardinales:
1. **Calidad Predictiva:** Cobertura de catálogo y precisión en el ranking ($MAP@12$).
2. **Restricciones de Servicio (SLOs):** Latencia de respuesta $p95 < 50\text{ ms}$ y consumo de memoria residente ($\text{RSS}$) acotado.
3. **Gasto Operacional ($\text{OpEx}$):** Coste financiero directo de cómputo, almacenamiento y transferencia de datos en servidores de producción.


## 12.2. Modelo Formal de Estimación de Costes y Escenarios de Tráfico

El dimensionamiento de infraestructura cuantifica el coste operacional mensual ($\text{TCO}_{\text{mensual}}$) del microservicio de inferencia en función del cómputo demandado, el almacenamiento de artefactos y el ancho de banda consumido:

$$\text{TCO}_{\text{mensual}} = C_{\text{cómputo}} + C_{\text{almacenamiento}} + C_{\text{transferencia}} + C_{\text{orquestación}}$$

Para evaluar la escalabilidad del sistema bajo condiciones operativas realistas, se definen tres niveles de tráfico transaccional representativos del comercio minorista:

* **Nivel 1 (Tráfico Base / Prototipo):** $1 \times 10^6\text{ peticiones/mes}$ ($\approx 0.38\text{ req/s}$ promedio; picos de hasta $5\text{ req/s}$).
* **Nivel 2 (Tráfico Medio / Campaña Comercial):** $10 \times 10^6\text{ peticiones/mes}$ ($\approx 3.85\text{ req/s}$ promedio; picos de hasta $50\text{ req/s}$).
* **Nivel 3 (Alto Tráfico / Campaña Masiva):** $50 \times 10^6\text{ peticiones/mes}$ ($\approx 19.3\text{ req/s}$ promedio; picos de hasta $250\text{ req/s}$).


## 12.3. Evaluación Comparativa de Escenarios de Infraestructura y Despliegue

Tomando como referencia las tarifas públicas de proveedores de nube pública para la región de Europa (`eu-west-1`), se analizan cuatro opciones arquitectónicas para alojar el motor de inferencia:

### 12.3.1. Escenario A: Prototipado y Validación en Nodo Local
* **Especificaciones de Hardware:** Estación de trabajo local con procesador AMD Ryzen 5 5500 (6 núcleos / 12 hilos a 3.6 GHz base), 12 GB de memoria RAM DDR4 y almacenamiento en estado sólido NVMe PCIe 3.0 (sin acelerador gráfico GPU dedicado).
* **Coste Cloud Directo:** **0,00 €**.
* **Viabilidad Operacional:** El procesamiento out-of-core con DuckDB y Polars permitió completar la ingesta de los 31.78 millones de registros transaccionales (3.49 GB brutos), la extracción de 39 variables discriminantes, el entrenamiento supervisado en 71.65 segundos y el estudio completo de ablación sin requerir aprovisionamiento de máquinas virtuales en la nube, eliminando el coste de experimentación.

### 12.3.2. Escenario B: Microservicio en Contenedor Gestionado (AWS Fargate / GCP Cloud Run)
* **Arquitectura:** Contenedor Docker multi-etapa (`app/Dockerfile`, basado en Python 3.12-slim y Uvicorn) desplegado como tarea serverless en AWS ECS Fargate.
* **Asignación de Recursos por Tarea:** 1 vCPU y 2 GB de memoria RAM, dimensionamiento holgado para la memoria residente registrada en régimen nominal (~381 MB RSS en Modo 2; 458.9 MB pico en pruebas de carga).
* **Tarifas de Referencia (AWS Fargate, región `eu-west-1`):**
  * vCPU: $\$0.04048\text{ USD / hora}$.
  * Memoria RAM: $\$0.004445\text{ USD / GB-hora}$.
  * Coste mensual por tarea continua (730 horas):
    $$C_{\text{tarea}} = 730 \times (0.04048 + 2 \times 0.004445) \approx \$36.04\text{ USD/mes} \approx 33{,}50\text{ €/mes}$$
* **Evaluación de Rendimiento y Escalabilidad:**
  * En la batería de pruebas de carga concurrente (`tests/benchmark_api.py`), una única instancia del contenedor alcanzó un throughput de **955.8 req/s** con latencias internas $p95 = 11.79\text{ ms}$. Esto demuestra que una sola tarea atiende con un margen de seguridad superior a $19\times$ los picos de demanda de los Niveles 1 y 2.
  * Para despliegues con alta disponibilidad (2 tareas en zonas de disponibilidad segregadas multi-AZ detrás de un balanceador de carga ALB), el coste total de infraestructura se sitúa entre **85 € y 100 € mensuales**, garantizando tolerancia ante fallos de nodo.

### 12.3.3. Escenario C: Arquitectura Serverless por Petición (AWS Lambda + EFS + API Gateway)
* **Arquitectura:** Función efímera instanciada por evento, accediendo a los artefactos Parquet del modelo mediante un sistema de archivos compartido AWS EFS.
* **Coste Proyectado según Volumen:**
  * *Nivel 1 (1M req/mes):* Cómputo ($\approx \$1.00$) + API Gateway ($\$1.00$) + Invocaciones ($\$0.20$) $\approx \mathbf{2,00\text{ €/mes}}$.
  * *Nivel 2 (10M req/mes):* $\approx \mathbf{20,50\text{ €/mes}}$.
  * *Nivel 3 (50M req/mes):* $\approx \mathbf{102,00\text{ €/mes}}$.
* **Compromisos Técnicos (*Trade-offs*):** Si bien ofrece ventajas económicas en fases de prueba o tráfico intermitente, sufre penalizaciones críticas por arranques en frío (*cold starts* de 150 a 300 ms). Este retraso se produce al inicializar el entorno CPython, vincular en caliente librerías dinámicas nativas en C++ (`libgomp.so`, LightGBM) y deserializar matrices a través del protocolo NFS de EFS, violando el SLO estricto de latencia ($p95 < 50\text{ ms}$).

### 12.3.4. Escenario D: Clúster Distribuido (Apache Spark) y Endpoints GPU (SageMaker)
* **Arquitectura de Referencia:** Clúster de computación distribuida (1 nodo master y 3 workers `m5.xlarge`) acoplado a un endpoint de inferencia con GPU dedicada (`ml.g4dn.xlarge` o `ml.g5.xlarge`).
* **Coste de Mercado:** Oscila entre **1.200 € y 2.000 € mensuales** en plataformas administradas.
* **Justificación Técnica del Rechazo (Ley de Amdahl y Sobrecoste de Red):**
  * El conjunto tabular de transacciones (31.78M filas) ocupa 3.49 GB en formato CSV plano, pero su representación en Parquet columnar con compresión Snappy/ZSTD ocupa menos de 500 MB.
  * Particionar este volumen entre múltiples nodos distribuidos introduce cuellos de botella por serialización entre la máquina virtual Java (JVM) y Python (Py4J), junto con penalizaciones de latencia por intercambio de particiones en red (*shuffle overhead*).
  * Motores analíticos vectorizados mononodo como DuckDB y Polars procesan la totalidad de las transformaciones en memoria compartida aprovechando extensiones vectoriales SIMD (AVX2), resolviendo la ingesta en tiempos menores sin incurrir en costes de clúster ni complejidad de orquestación.


## 12.4. Síntesis Comparativa de Costes y Rendimiento de Servicio

| Alternativa de Cómputo | Coste (1M req/mes) | Coste (10M req/mes) | Coste (50M req/mes) | Latencia $p95$ Típica | Viabilidad Operativa |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Local / Estación de Trabajo** | 0,00 € | 0,00 € | 0,00 € | ~12 ms | Óptimo para desarrollo, entrenamiento y validación offline. |
| **Docker en AWS Fargate** | ~33,50 € | ~33,50 € | ~67,00 € (2 tareas) | **11.79 ms** | **Recomendada:** Predecible, latencia submétrica y soporte >950 RPS. |
| **Serverless (Lambda + EFS)** | ~2,00 € | ~20,50 € | ~102,00 € | ~40 ms (Cold: ~250 ms) | Variable; penalizada por cold starts de librerías nativas C++. |
| **Clúster Spark + GPU (Ref.)** | > 1.200 € | > 1.200 € | > 1.500 € | ~45 ms | Descartada: Sobredimensionada e ineficiente para 3.49 GB. |


## 12.5. Análisis de Eficiencia de Cómputo: Pipeline Two-Stage CPU vs. Redes Neuronales Profundas GPU

La elección de un pipeline híbrido en dos etapas basado en árboles de decisión supervisados (`LGBMRanker`) ejecutados sobre CPU frente a arquitecturas neuronales profundas (Two-Tower / DLRM) sobre GPU responde a criterios de rendimiento por unidad de coste:

| Dimensión Analizada | Red Neuronal Bi-Torre en GPU (DLRM / Two-Tower) | Pipeline Two-Stage Propuesto (DuckDB + Polars + LightGBM) |
| :--- | :---: | :---: |
| **Hardware Requerido** | GPU dedicada (NVIDIA A100 / V100 con $\ge 16$ GB VRAM) | **CPU estándar mononodo** (6 núcleos, 12 GB RAM) |
| **Duración del Entrenamiento** | ~18 a 24 horas por iteración | **~45 min (pipeline completo) / 71.65 s (LGBMRanker)** |
| **Memoria RAM / VRAM Pico** | $> 16\text{ GB RAM} + \text{VRAM dedicada}$ | **2.104 MB (ingesta) / 1.428 MB (ranking)** |
| **Memoria Residente de Servicio** | $> 2.500\text{ MB}$ (entorno PyTorch / TensorFlow) | **458.9 MB** (contenedor Docker optimizado) |
| **Latencia Inferencia Booster** | $> 100\text{ ms}$ (Forward-pass neuronal denso) | **$< 3\text{ ms}$ ($0.95\text{ ms}$ media booster C++)** |
| **Coste de Servidor Cloud** | $> 500\text{ USD/mes}$ (instancias aceleradas) | **$15\text{ a } 35\text{ USD/mes}$ (instancias CPU estándar)** |
| **Complejidad de Mantenimiento** | Dependencias CUDA, runtime cuDNN y compilación | Imagen Docker Debian slim (295 MB) sin capas GPU |

### 12.5.1. Fundamentación Causal de la Superioridad en Datos Tabulares Minoristas
En problemas de recomendación minorista con matrices de interacción extremadamente dispersas (99.9% de dispersión), los modelos neuronales densos requieren matrices masivas de embeddings que saturan la memoria gráfica VRAM y presentan dificultades de optimización ante el 83% de usuarios esporádicos. 

Por el contrario, el algoritmo `LGBMRanker` con función de pérdida listwise (`lambdarank`) opera directamente sobre variables agregadas de recencia temporal, frecuencia transaccional y afinidad de cesta. La evaluación de árboles de decisión en C++ se ejecuta de forma determinista mediante saltos condicionales en registros de CPU, alcanzando latencias de inferencia de **0.95 ms** sin demandar costosas unidades de cómputo tensorial.


## 12.6. Análisis de Sobriedad de Cómputo y Eficiencia Operacional en Mononodo

El análisis de recursos evidencia que la optimización algorítmica y el diseño out-of-core resuelven el cuello de botella computacional sin necesidad de sobredimensionar la infraestructura. Al reemplazar clústeres multi-nodo y servidores acelerados por GPU por procesamiento columnar vectorizado en CPU mononodo:

1. **Eliminación del Sobrecoste de Cómputo Ocioso:** A diferencia de los aceleradores GPU dedicados (que demandan asignación permanente de memoria VRAM e instancias costosas en reposo), la arquitectura basada en CPU estándar optimizada permite escalar el microservicio de forma elástica con arranques en frío mínimos y huella de memoria acotada (<460 MB RSS).
2. **Aprovechamiento Vectorial SIMD:** La ejecución de LightGBM en C++ y Polars sobre Apache Arrow aprovecha de forma nativa los registros vectoriales del procesador local, completando el entrenamiento completo en 71.65 segundos sin requerir dependencias aceleradas propietarias (CUDA/cuDNN).


## 12.7. Conclusiones del Dimensionamiento de Infraestructura

1. **Eficiencia Presupuestaria Validada:** La delimitación precisa de la escala del catálogo permite operar un microservicio de recomendación en alta disponibilidad con un coste inferior a **100 € mensuales** en nube pública, logrando una reducción superior al **90% en costes de infraestructura** frente a arquitecturas deep learning en GPU.
2. **Cumplimiento Holgado de SLOs:** La combinación de LightGBM en C++ con arrays contiguos en memoria sostiene **955.8 req/s** con latencias internas $p95 = 11.79\text{ ms}$, garantizando escalabilidad para campañas de alto tráfico sin requerir sobredimensionamiento de servidores.
3. **Cero Dependencia de Cómputo Distribuido:** En datos tabulares minoristas de escala moderada (3.49 GB), la ingeniería de datos out-of-core en nodo único supera tanto en coste financiero como en latencia a soluciones distribuidas sobre Apache Spark.
