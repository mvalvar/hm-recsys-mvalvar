# Capítulo 1: Introducción y Motivación

**Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid**  
**Autor:** Manuel Valdivia  
**Proyecto:** Motor de Recomendación Escalable para Retail de Moda  


## 1.1. Contexto Empresarial e Industrial

La transformación digital del comercio minorista ha incrementado el volumen de catálogo accesible para los consumidores finales, generando a la vez el problema habitual de sobrecarga de opciones. En cadenas de distribución de moda como H&M, con miles de tiendas físicas y presencia de comercio electrónico internacional, la interacción de millones de usuarios con catálogos de cientos de miles de artículos plantea una dificultad recurrente para encontrar rápidamente productos relevantes.

Cuando un usuario no encuentra con agilidad prendas acordes con su estilo, talla, estacionalidad o presupuesto, se manifiestan dos problemas operativos habituales:
1. **Pérdida de conversión y abandono de sesión**: Disminución del valor medio de pedido (*Average Order Value*, AOV) y caída en la recurrencia de compra.
2. **Coste e impacto logístico de las devoluciones**: En el comercio digital de moda, las tasas de devolución suelen situarse en rangos elevados (frecuentemente entre el 20 % y el 40 %). Sugerir productos poco afines a las preferencias del cliente incrementa los costes de logística inversa y reacondicionamiento de inventario.


## 1.2. Planteamiento del Problema Técnico

El dataset *H&M Personalized Fashion Recommendations* publicado en Kaggle (2022) reúne características comunes a los sistemas de recomendación en producción:

1. **Gestión de memoria y volumen de datos (*Medium Data*)**: Con 31.7 millones de transacciones, 1.37 millones de clientes y 105.542 artículos a lo largo de 104 semanas, el archivo `transactions_train.csv` ocupa 3.49 GB en disco. Aunque entra en la categoría de *Medium Data* (al caber en un disco secundario estándar), cargarlo directamente en memoria con herramientas como Pandas genera un consumo superior a los 14 GB debido a la sobrecarga de tipos y punteros de Python, superando el límite de un equipo local estándar (12 GB de RAM) y requiriendo técnicas de lectura por bloques y almacenamiento eficiente (*out-of-core*).
2. **Dispersión acusada de interacciones (*Sparsity*)**: La matriz de interacción binaria cliente-artículo presenta una dispersión del **99.9780 %** a nivel global y del **99.9955 %** en ventanas recientes de 5 semanas. Con este volumen de ceros y pocos datos por usuario, los modelos clásicos de factorización matricial (*Matrix Factorization*, como SVD o ALS) tienen dificultades para converger o generan representaciones latentes ruidosas.
3. **Estacionalidad corta y rotación de catálogo**: A diferencia de catálogos estables (como libros o películas), las prendas de moda rápida tienen ciclos de vida de pocas semanas y una fuerte dependencia del clima y las colecciones temporales.


## 1.3. Objetivos del Trabajo Fin de Máster

### 1.3.1. Objetivo General
Diseñar, implementar, validar y desplegar un motor de recomendación personalizado de dos etapas (*Two-Stage Recommender System*) para el catálogo de H&M, optimizando la métrica de ranking $MAP@12$ bajo las restricciones de un equipo local con 12 GB de RAM.

### 1.3.2. Objetivos Específicos
1. **Pipeline de Ingesta Eficiente**: Desarrollar un flujo de lectura y transformación por bloques utilizando DuckDB y Polars para generar archivos Parquet comprimidos con ZSTD, optimizando los tipos de datos (downcasting) y conteniendo el uso de memoria RAM.
2. **Recuperación Multi-Heurística de Candidatos**: Implementar un conjunto de 8 heurísticas prácticas de recuperación para equilibrar compras pasadas, tendencias generales, popularidad demográfica y co-ocurrencias en cesta.
3. **Feature Engineering sin Fuga Temporal**: Formular 39 variables predictivas calculadas de forma estrictamente previa a la ventana de evaluación, evitando cualquier tipo de fuga de información (*data leakage*).
4. **Modelo Supervisado de Ranking**: Entrenar un modelo `LGBMRanker` optimizado con función de pérdida de ordenación (*LambdaRank*) y un esquema controlado de muestreo negativo (ratio 1:5) para ordenar los candidatos.
5. **Estudio de Ablación y Sensibilidad**: Medir el impacto individual de las decisiones clave (ventana temporal, fuentes de candidatos, proporción de negativos y funciones de pérdida) mediante 6 pruebas comparativas aisladas.
6. **Interpretabilidad con SHAP**: Analizar la contribución de las variables al ranking mediante valores SHAP (`TreeExplainer`) para verificar la coherencia de las predicciones a nivel global y en perfiles de usuario concretos.
7. **Despliegue y Servicio Web**: Empaquetar la solución en una API REST con FastAPI y Docker para servir predicciones en tiempo real y validar latencias y consumo de memoria.

### 1.3.3. Criterio de Trabajo: Eficiencia Local y Despliegue en Contenedor

Para abordar el proyecto de forma realista y reproducible, se adoptó un enfoque dividido en dos entornos:

1. **Desarrollo y modelado en equipo local:**  
   Todo el pipeline de datos, análisis exploratorio y entrenamiento se ejecutó sobre un ordenador personal estándar (procesador AMD Ryzen 5 5500, 12 GB de RAM, sin tarjeta gráfica dedicada bajo Windows). El propósito de esta restricción era comprobar que, mediante herramientas modernas de procesamiento columnar y streaming (*DuckDB + Polars*), es posible trabajar con decenas de millones de registros sin necesidad de contratar clústeres distribuidos en la nube en fases tempranas de experimentación.

2. **Empaquetado para despliegue en Docker:**  
   Para facilitar la revisión y asegurar que el entorno de ejecución sea reproducible en cualquier sistema operativo (Linux, macOS o Windows), la API del recomendador se empaquetó en una imagen Docker ligera basada en **Debian 12 con Python 3.12-slim** (`app/Dockerfile` y `docker-compose.yml`), permitiendo levantar el servicio con `docker compose up --build -d`.


## 1.4. Estructura de la Memoria y Articulación del Proyecto

El desarrollo del proyecto y la redacción de la presente memoria se estructuran en cuatro bloques temáticos que articulan los trece capítulos del trabajo fin de máster:

* **Bloque I: Fundamentación y Marco Teórico (Capítulos 1 y 2)**  
  En este bloque se establece el contexto de negocio, los objetivos y las restricciones de hardware del proyecto (Capítulo 1), seguido de una revisión crítica del estado del arte en sistemas de recomendación (Capítulo 2), analizando las limitaciones del filtrado colaborativo tradicional ante catálogos con dispersión extrema y justificando el paradigma en dos etapas (*Two-Stage Recommender System*).

* **Bloque II: Infraestructura de Datos y Análisis Exploratorio (Capítulos 3 y 4)**  
  Se detalla la arquitectura de procesamiento fuera de memoria (*out-of-core*) mediante DuckDB y Polars para el tratamiento de 31.78 millones de registros sin recurrir a infraestructura distribuida en la nube (Capítulo 3). Asimismo, se expone la caracterización empírica y el análisis exploratorio de datos a través de catorce paneles analíticos de auditoría y una matriz de síntesis metodológica, caracterizando la rotación acelerada de catálogo, la concentración de ventas y los ciclos de recompra (Capítulo 4).

* **Bloque III: Pipeline de Modelado Predictivo (Capítulos 5, 6 y 7)**  
  Comprende el núcleo algorítmico del sistema: el diseño de ocho heurísticas complementarias de generación de candidatos para maximizar la cobertura (*recall*) (Capítulo 5); la extracción y codificación de 39 variables predictivas calculadas bajo corte temporal estricto para evitar cualquier fuga de información (*target leakage*) (Capítulo 6); y el entrenamiento del modelo de reordenación supervisada `LGBMRanker` con función de pérdida LambdaRank, culminando en la arquitectura en cascada híbrida multidimensional V8 (Capítulo 7).

* **Bloque IV: Validación Experimental, Explicabilidad y Despliegue (Capítulos 8 al 13)**  
  Abarca la evaluación experimental offline y el benchmark frente a la métrica oficial MAP@12 en Kaggle (Capítulo 8); el estudio sistemático de ablación para aislar el aporte de cada componente (Capítulo 9); el análisis de interpretabilidad y atribución de importancia con TreeSHAP (Capítulo 10); la productivización del motor mediante un microservicio REST en FastAPI empaquetado en Docker (Capítulo 11); la auditoría de costes de infraestructura y dimensionamiento FinOps (Capítulo 12); y las conclusiones generales y líneas de trabajo futuro (Capítulo 13).

La formulación preliminar del proyecto se conserva como referencia en la [Propuesta Inicial del Proyecto](propuesta_inicial.md).

### 1.4.1. Organización de Entregables del Repositorio

El trabajo se articula a través de tres componentes complementarios en el repositorio de código:
* **Memoria monográfica (`memoria/`):** Los trece capítulos estructurados en Markdown constituyen el documento formal de la tesis, detallando la formulación matemática, el marco teórico, el análisis exploratorio, el estudio de ablación, la interpretabilidad y el análisis de costes.
* **Cuaderno maestro ejecutable (`memoria_final_recsys.ipynb`):** Cuaderno único ubicado en la raíz del proyecto que compila el pipeline completo de ingeniería de datos, modelado supervisado y evaluación oficial, sincronizado de forma estricta con las métricas consolidadas de la memoria.
* **Cuadernos de experimentación (`notebooks/`):** Banco de trabajo modular utilizado durante la fase de desarrollo para depuración de código, inspección paso a paso de matrices intermedias y validación sobre la partición local $W_{104}$.


## 1.5. Evolución del Criterio Técnico desde la Propuesta Inicial

A lo largo de la ejecución del trabajo, el contraste empírico contra los datos reales y la necesidad de maximizar la precisión bajo cotas estrictas de memoria motivaron la revisión de premisas planteadas en la propuesta inicial, articulándose en tres transiciones arquitectónicas principales:

### 1.5.1. De la Premisa de Big Data Distribuido al Procesamiento Out-of-Core en Nodo Único
La propuesta inicial catalogaba el problema bajo el paraguas genérico de *Big Data*, sugiriendo la posible necesidad de entornos distribuidos como Apache Spark. Sin embargo, el archivo transaccional bruto ocupa 3.49 GB en disco, clasificándose formalmente como *Medium Data*: un volumen que excede la memoria RAM de un equipo doméstico si se carga ingenuamente mediante DataFrames tradicionales de Pandas (demandando más de 14 GB debido a punteros y cadenas de caracteres), pero que cabe con holgura en un disco de estado sólido local. 

La adopción de motores modernos de procesamiento columnar como DuckDB (con ejecución vectorizada y lectura por bloques en streaming) y Polars (con tipado eficiente mediante Apache Arrow) permitió ejecutar todo el pipeline de filtrado, agregación y *downcasting* manteniendo la memoria residente por debajo de 1.8 GB. Esta decisión eliminó la sobrecarga de serialización en red, la latencia de arranque y los costes económicos derivados de mantener clústeres en la nube durante la fase de experimentación.

### 1.5.2. Del Enfoque Neuronal Two-Tower a la Arquitectura Híbrida Desacoplada
El planteamiento preliminar contemplaba entrenar una red neuronal de dos torres (*Two-Tower Neural Network*) optimizada mediante *Bayesian Personalized Ranking* (BPR). El análisis del catálogo reveló dos impedimentos técnicos fundamentales: una dispersión del 99.9780% y la ausencia de GPU dedicada en el entorno de cómputo local. En ese escenario, proyectar más de 105.000 artículos y 1.37 millones de usuarios en espacios latentes densos conllevaba tiempos de convergencia desmesurados y un riesgo severo de representaciones degradadas por falta de soporte estadístico en artículos de baja rotación.

En su lugar, se adoptó el estándar industrial desacoplado en dos etapas: una fase de recuperación multi-heurística que combina ocho generadores deterministas (recencia, co-ocurrencia en cesta, popularidad demográfica y departamental), reduciendo el espacio de búsqueda en un 99.9% sin cómputo de gradientes, seguida de una etapa de reordenación supervisada con `LGBMRanker`. Esta estructura permitió entrenar modelos de gradiente sobre árboles de decisión en menos de 10 minutos en CPU, facilitando iteraciones rápidas y auditorías de interpretabilidad mediante valores SHAP.

### 1.5.3. De la Inferencia Supervisada Directa a la Cascada No Destructiva V8
Inicialmente se asumió que un único modelo supervisado clasificaría y seleccionaría directamente los doce artículos recomendados para cada usuario. La validación empírica en las primeras iteraciones (V1 a V4) demostró que confiar ciegamente en las probabilidades del modelo desplazaba las recompras inmediatas del cliente o forzaba recomendaciones ruidosas en usuarios con historiales escasos o inexistentes (*cold start*). 

Para corregir esta deficiencia, se diseñó la arquitectura en cascada híbrida V8: un esquema jerárquico no destructivo donde las recompras personales de las últimas cuatro semanas se preservan de forma prioritaria, los huecos restantes se completan mediante afinidad causal de compra conjunta $P(B|A)$ ponderada por recencia, y únicamente los espacios finales se cubren con superventas estacionales segmentadas por grupo de edad. Esta evolución permitió elevar el rendimiento en la competición pública de Kaggle desde un MAP@12 de 0.01728 en la inferencia directa hasta 0.02386 en la solución final.
