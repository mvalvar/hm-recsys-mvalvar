# Propuesta de Trabajo Fin de Máster
**Máster en Data Science, Big Data & Business Analytics**  
**Universidad Complutense de Madrid**

---

* **Título:** Motor de Recomendación Escalable para Retail de Moda
* **Autor:** Manuel Francisco Valdivia Vargas

---

## 1. Opción Elegida

El proyecto consiste en un motor de recomendación predictivo para comercio electrónico a través de procesamiento Big Data y modelos de *Learning to Rank* para personalizar la experiencia de compra y reducir las devoluciones logísticas. 

El proyecto se enmarca en la tercera opción disponible para realizar el TFM, ya que el alcance del proyecto trasciende las limitaciones individuales de la primera y segunda opción disponible.

### Justificación Técnica
El dataset de *H&M Personalized Fashion Recommendations* (Kaggle, 2022) comprende 31.7 millones de transacciones, 1.37 millones de clientes y 105.542 artículos, con un archivo de transacciones de 3.2 GB en disco. Tres restricciones técnicas concretas determinan la elección del perfil:

* **Restricción de memoria:** El archivo `transactions_train.csv`, cargado en Pandas, ocupa entre 12 y 16 GB en RAM. Esto excede la memoria disponible y obliga a un pipeline de procesamiento fuera de memoria.
* **Problema de dispersión extrema:** La matriz usuario-ítem tiene una dispersión (*sparsity*) estimada superior al 99,97 %. En ese régimen, los métodos de factorización matricial clásicos (SVD, ALS) divergen o generan embeddings estadísticamente poco fiables. Se requiere una arquitectura híbrida que combine señales de popularidad con modelos de ranking supervisado.

---

## 2. Índice Propuesto

1. **Resumen ejecutivo**
2. **Introducción y descripción del problema**
3. **Arquitectura de datos y motor de procesamiento Big Data**
4. **Auditoría de datos y análisis exploratorio (EDA)**
5. **Feature engineering y esquema de validación temporal**
6. **Modelización analítica:** arquitectura Two Tower y Learning to Rank
7. **Evaluación de resultados y métricas de ranking** (MAP@12, NDCG@12)
8. **Interpretabilidad del modelo** (XAI - Valores SHAP)
9. **Productivización:** arquitectura de inferencia y estrategia de despliegue
10. **Conclusiones y líneas de trabajo futuro**
11. **Referencias**

* **Anexo A:** Diccionario de datos
* **Anexo B:** Repositorio de código (GitHub)
* **Anexo C:** Gráficos del análisis exploratorio

---

## 3. Descripción del Proyecto

### 3.1 Problema y contexto
H&M opera más de 4.850 tiendas físicas y 53 mercados online. Un catálogo de esta magnitud produce que el cliente no encuentre rápidamente lo que busca, lo que aumenta la tasa de abandono y, cuando hay compra, nos enfrentamos a la probabilidad de devolución por selección inadecuada.

El proyecto consiste en un sistema que, a partir del historial de compras y los metadatos de producto y cliente, predice los artículos que cada usuario comprará en los 7 días siguientes al corte temporal del entrenamiento. La validación no es interna: las predicciones se envían a la competición pública de Kaggle (*H&M Personalized Fashion Recommendations*) y se evalúan contra el Ground Truth retenido por los organizadores bajo la métrica **MAP@12**.

### 3.2 Dataset

| Archivo | Descripción | Volumen |
| :--- | :--- | :--- |
| `transactions_train.csv` | Historial de compras | 31.7 M filas / 3.2 GB |
| `articles.csv` | Metadatos de producto (25 atributos) | 105.542 artículos |
| `customers.csv` | Datos demográficos de clientes | 1.37 M clientes |
| `images/` | Fotografías de producto | ~105 k imágenes |

### 3.3 Metodología y fases de desarrollo

* **Fase 1: Arquitectura de datos e ingeniería del pipeline**  
  El pipeline de ingesta usará DuckDB (motor OLAP columnar que procesa por bloques desde disco sin cargar el archivo en RAM) y Polars para las transformaciones intermedias. Cada etapa escribirá un checkpoint en formato Parquet, de modo que una interrupción no obliga a reiniciar desde cero. Se incluirán aserciones de calidad de datos (esquema, nulos, rangos) en cada punto de entrada.

* **Fase 2: Auditoría de datos y EDA**  
  El EDA no parte de hipótesis fijas. En particular se evaluarán:
  * (a) La *sparsity* real de la matriz usuario-ítem.
  * (b) La distribución de popularidad por artículo para cuantificar la cola larga.
  * (c) La varianza léxica del campo `detail_desc` para evaluar si el NLP tabular tiene señal suficiente.
  * (d) La autocorrelación temporal de las ventas por categoría.  
  *Los resultados del EDA determinan las decisiones de modelado de la Fase 4.*

* **Fase 3: Feature engineering y esquema de validación temporal**  
  El esquema de validación usa un corte temporal estricto: entrenar con datos hasta la semana $N-2$, validar en la semana $N-1$, testear en la semana $N$. El *Random K-Fold* queda descartado porque mezcla observaciones futuras con el conjunto de entrenamiento, inflando artificialmente las métricas. Las variables categóricas de alta cardinalidad (`article_id`, `colour_group`) se codifican con Target Encoding regularizado (*K-fold mean*) en lugar de One-Hot, que generaría vectores dispersos de 105 k dimensiones.

* **Fase 4: Modelización: Two Tower Learning to Rank**  
  El problema se formula como *Learning to Rank* (L2R), no como clasificación binaria. La arquitectura tiene dos etapas:
  * **(a) Retrieval:** Genera un conjunto de candidatos por cliente combinando popularidad reciente ponderada por decaimiento temporal y artículos con comportamiento histórico similar.
  * **(b) Re-Ranking:** Ordena esos candidatos con un LightGBM Ranker entrenado con *Bayesian Personalized Ranking* (BPR) loss, que optimiza la posición relativa de los ítems, no su probabilidad absoluta. El pipeline de entrenamiento opera en mini-batches para mantenerse dentro del límite de 16 GB de RAM.

* **Fase 5: Interpretabilidad (XAI)**  
  Se calcularán Valores SHAP sobre el modelo de Re-Ranking usando `TreeExplainer`. A diferencia del *Feature Importance* por ganancia (sesgado hacia variables de alta cardinalidad), los valores SHAP son aditivos, localmente exactos y consistentes por construcción. Se generarán un *Summary Plot* global y *Force Plots* para tres perfiles de usuario con comportamientos distintos.

* **Fase 6: Productivización y despliegue**  
  Se diseñará una API REST con FastAPI que, dado un `customer_id`, devuelva las 12 recomendaciones ordenadas. El diseño cubrirá: contrato de API (OpenAPI), caché de inferencia para usuarios frecuentes, estrategia de reentrenamiento periódico y detección de *drift* en la distribución de entrada. Se comparará el coste operativo entre despliegue Serverless y contenedores gestionados (Docker + Kubernetes).