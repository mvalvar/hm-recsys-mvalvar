# Capítulo 3: Arquitectura de Datos y Pipeline de Procesamiento Out-of-Core

**Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid**  
**Autor:** Manuel Valdivia  
**Proyecto:** Motor de Recomendación Escalable para Retail de Moda  


## 3.1. Entorno de Ejecución y Restricciones de Hardware

El proyecto se desarrolló bajo una configuración de hardware común para un equipo de escritorio:
- **Procesador**: AMD Ryzen 5 5500 (6 núcleos, 12 hilos lógicos).
- **Memoria RAM**: 12 GB DDR4 disponibles.
- **Almacenamiento**: Disco de estado sólido NVMe.
- **Aceleración Gráfica**: Sin tarjeta gráfica GPU dedicada para tareas de deep learning.

El archivo principal de transacciones (`transactions_train.csv`) contiene 31.788.324 filas y ocupa 3.48 GiB en disco (3.49 GB decimales, 3.731.586.838 bytes). Al intentar cargarlo directamente con `pandas.read_csv()`, el uso de memoria RAM supera los 14 GB debido a la sobrecarga que introducen los punteros de Python y el manejo de cadenas de texto (como los hashes de 64 caracteres en `customer_id`), provocando errores por falta de memoria (`MemoryError`) en un equipo de 12 GB.


## 3.2. Criterio de Diseño: Procesamiento Out-of-Core en Nodo Local

Al enfrentarse a un dataset de decenas de millones de registros que no cabe en la memoria RAM disponible, existen dos alternativas habituales:

1. **Desplegar un clúster distribuido (como Apache Spark):** Adecuado cuando los datos ocupan terabytes y superan la capacidad de disco de una sola máquina. Sin embargo, para datasets de pocos gigabytes (*Medium Data*), introducir un clúster añade complejidad de configuración, costes de infraestructura en la nube y sobrecarga por transferencia de datos en red (McSherry et al., 2015).
2. **Utilizar herramientas analíticas de streaming y procesamiento por bloques en nodo único:** Motores como **DuckDB** y bibliotecas de procesamiento columnar como **Polars** permiten consultar y transformar datos directamente desde disco con una huella de memoria muy reducida (Raasveldt & Mühleisen, 2019).

Para este proyecto se optó por la segunda vía: procesar los 3.48 GiB (3.49 GB) de CSV mediante consultas SQL en streaming con DuckDB, transferir los resultados intermedios a Polars mediante la interfaz Apache Arrow para realizar downcasting de tipos numéricos, y almacenar los resultados en formato columnar Parquet comprimido con ZSTD. Este esquema permitió completar todo el preprocesamiento manteniendo el consumo de memoria RAM por debajo de **1.8 GB (RSS)** sin incurrir en costes de servidores cloud.


## 3.3. Ingesta y Filtrado con DuckDB

Se utilizó **DuckDB** como motor SQL embebido para leer el CSV crudo desde disco sin cargarlo por completo en memoria:
1. **Lectura por bloques (Streaming):** DuckDB procesa el archivo por lotes vectorizados directamente contra disco.
2. **Filtrado temporal en origen:** Se seleccionan únicamente las transacciones correspondientes a la ventana de trabajo (últimas 5 o 10 semanas), reduciendo el número de filas de 31.78M a aproximadamente 1.30M–1.33M (`transactions_5w.parquet`).
3. **Proyección de columnas necesarias:** Se seleccionan solo los campos requeridos para el cálculo de heurísticas y variables (`t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id`).


## 3.4. Transformación y Optimización de Tipos de Datos con Polars

Una vez extraídas las transacciones recientes, los datos se convierten a DataFrames de **Polars** mediante punteros en memoria de Apache Arrow (evitando duplicar información en RAM) para aplicar un ajuste defensivo de tipos de datos (*downcasting*):

### Comparativa de Huella de Memoria por Fila

| Campo Transaccional | Tipo en Pandas (64-bit) | Tipo en Polars | Consumo Pandas | Consumo Polars | Reducción | Justificación |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `customer_id` | `object` (Hash hex 64 chars) | `Int32` (`customer_idx`) | 72 B | 4 B | -94.4% | Indexación de 1.37M clientes en entero de 32 bits. |
| `article_id` | `object` / `int64` | `Int32` | 16 B | 4 B | -75.0% | Catálogo de 105.542 artículos indexado en 4 bytes. |
| `t_dat` | `object` / `datetime64[ns]` | `Date` (días Unix) | 8 B | 4 B | -50.0% | Resolución temporal diaria sin microsegundos. |
| `price` | `float64` (doble precisión) | `Float32` (simple precisión) | 8 B | 4 B | -50.0% | Precisión monetaria suficiente para retail. |
| `sales_channel_id`| `int64` | `Int8` | 8 B | 1 B | -87.5% | Dominio binario discreto (1: Físico, 2: Online). |
| Total por Fila | — | — | 112 B | 17 B | -84.8% | Reducción de 6.6 veces en huella de memoria. |

La correspondencia biunívoca entre los identificadores originales en texto (`customer_id`) y los índices enteros continuos (`customer_idx`) se preserva en `customer_id_mapping.parquet`, permitiendo la reconstrucción fidedigna de las claves originales en la fase final de emisión de recomendaciones.


## 3.5. Almacenamiento Intermedio en Formato Parquet (ZSTD)

Las etapas intermedias del pipeline persisten sus resultados en el directorio `data_processed/` en formato Apache Parquet con compresión Zstandard (ZSTD nivel 3), lo que acelera lecturas posteriores y reduce el espacio en disco:

### Checkpoints Principales
- `transactions_5w.parquet`: Transacciones de las últimas 5 semanas con tipos compactados.
- `transactions_10w.parquet`: Transacciones de 10 semanas para extraer agregaciones históricas de usuario.
- `customers.parquet`: Datos demográficos de clientes con tramos de edad (`age_bin`).
- `articles.parquet`: Atributos y categorías del catálogo de prendas.
- `customer_id_mapping.parquet`: Mapeo hash de 64 caracteres $\leftrightarrow$ `customer_idx` (entero 32 bits).
- `candidates.parquet`: Pares candidato usuario-artículo consolidados por las heurísticas de recall.
- `features_matrix.parquet`: Matriz tabular con las 39 variables utilizadas para entrenar y evaluar `LGBMRanker`.

### Artefactos para el Servicio de Inferencia (V8)
- `v8_bestsellers_age.parquet`: Artículos más vendidos por tramo de edad y globales con decaimiento temporal, utilizados como contingencia para clientes sin compras recientes.
- `v8_basket_affinity.parquet`: Matriz de co-ocurrencia de artículos en compras conjuntas para sugerencias de venta cruzada.
- `v8_customer_history_28d.parquet`: Historial de compras personales de los últimos 28 días por cliente.


## 3.6. Diagrama del Pipeline de Ingesta

El flujo completo de transformación e ingesta se resume en la siguiente figura:

![Figura 3.1: Pipeline de Ingesta y Procesamiento Out-of-Core](../results/figures/diag_01_data_ingestion_out_of_core.png)

*Figura 3.1: Flujo de ingesta de datos. Los 3.49 GB de transacciones se leen por streaming desde disco con DuckDB, se transfieren vía Arrow a Polars para aplicar downcasting numérico (ahorro >84% de memoria por fila) y se guardan como archivos Parquet comprimidos con Zstandard.*

Este diseño desacoplado reporta tres ventajas operativas fundamentales: en primer lugar, el consumo de memoria residente se mantiene de forma estricta por debajo de 1.8 GB RSS, habilitando la ejecución completa del preprocesamiento en una estación local estándar; en segundo lugar, la ejecución de filtros SQL tempranos asegura un corte temporal riguroso que aísla la semana de validación e imposibilita la fuga de información causal (*target leakage*); y en tercer lugar, la disposición columnar en formato Parquet optimiza el rendimiento de lectura y acelera los cruces tabulares (*joins*) en las fases subsecuentes de ingeniería de características.


## 3.7. Validación de Calidad de Datos (Aserciones en Pipeline)

Para evitar la propagación silenciosa de anomalías hacia las etapas de modelado y evaluación, el pipeline incorpora compuertas de control de calidad mediante aserciones programáticas antes de persistir cada artefacto:

```python
# Comprobaciones de calidad en el script de ingesta
assert df.height > 0, "Error: El DataFrame resultante está vacío."
assert df.filter(pl.col("price") <= 0).height == 0, "Error: Se detectaron precios menores o iguales a cero."
assert df.filter(pl.col("customer_idx").is_null()).height == 0, "Error: Valores nulos en identificador de cliente."
assert df.filter(pl.col("article_id").is_null()).height == 0, "Error: Valores nulos en identificador de artículo."
assert df["t_dat"].max() <= t_split, "Error: Transacciones posteriores a la fecha de corte detectadas (temporal data leakage)."
```

Estas comprobaciones garantizan la integridad referencial de los identificadores de artículo frente al catálogo maestro, verifican la validez semántica de los importes monetarios (descartando precios nulos o negativos) y fiscalizan la frontera temporal para asegurar que ningún registro posterior a la fecha de corte contamine los conjuntos de entrenamiento. Asimismo, los procesos de escritura se implementan con atomicidad sobre disco, garantizando la idempotencia del flujo ante eventuales interrupciones de cómputo.
