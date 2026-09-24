# Capítulo 8: Evaluación Experimental y Métricas de Ranking

**Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid**  
**Autor:** Manuel Valdivia  
**Proyecto:** Motor de Recomendación Escalable para Retail de Moda  


## 8.1. Marco Formal de Evaluación y Métricas de Ranking

La evaluación cuantitativa del sistema de recomendación se fundamenta en métricas de ranking para listas ordenadas de longitud fija ($K = 12$). En entornos de comercio electrónico de moda rápida con catálogos dinámicos, la precisión de las recomendaciones no depende únicamente de la presencia de artículos relevantes en el conjunto predicho, sino de su posición relativa en la lista presentada al usuario. La cota de corte $K = 12$ responde a los condicionantes de diseño en interfaces de comercio electrónico móvil, donde las cuadrículas de producto (2 columnas $\times$ 6 filas) y los carruseles de correo transaccional definen el umbral de atención visual inmediata del consumidor.

### 8.1.1. Formulación de Average Precision at K (AP@12) y Mean Average Precision (MAP@12)

La métrica principal del proyecto corresponde al **Mean Average Precision at 12 (MAP@12)**. Para un cliente individual $u \in U$, dado el conjunto de artículos efectivamente adquiridos en el período de prueba $A_u$ y la lista ordenada de hasta 12 recomendaciones emitidas $P_u = (p_1, p_2, \dots, p_K)$ con $K \le 12$, la precisión promedio (*Average Precision*) se define formalmente como:

$$\text{AP@12}(u) = \frac{1}{\min(|A_u|, 12)} \sum_{k=1}^{\min(|P_u|, 12)} P(k) \times \text{rel}(k)$$

donde:
* $P(k) = \frac{1}{k} \sum_{i=1}^k \text{rel}(i)$ representa la precisión de corte en la posición $k$, es decir, la proporción de artículos relevantes entre las primeras $k$ sugerencias.
* $\text{rel}(k) \in \{0, 1\}$ es una función indicadora binaria que toma el valor $1$ si la prenda ubicada en el rango $k$ pertenece al conjunto real de compras $A_u$ ($\text{rel}(k) = \mathbb{I}(p_k \in A_u)$), y $0$ en caso contrario.
* El factor de normalización $\min(|A_u|, 12)$ penaliza en proporción al número de artículos comprados, fijando un valor máximo de $1.0$ cuando el sistema ubica todos los aciertos disponibles en los primeros puestos.
* **Protocolo de deduplicación estricta:** Si la lista $P_u$ contiene elementos repetidos, solo la primera ocurrencia del artículo recibe valor en $\text{rel}(k)$; las repeticiones posteriores computan como $\text{rel}(k) = 0$, incrementando el denominador $k$ y castigando la precisión del usuario.

El rendimiento agregado del sistema sobre el universo de clientes evaluados $|U|$ se obtiene mediante el promedio aritmético de las puntuaciones individuales:

$$\text{MAP@12} = \frac{1}{|U|} \sum_{u \in U} \text{AP@12}(u)$$

La estructura hiperbólica del factor de descuento $P(k) = \frac{\cdot}{k}$ impone una penalización no lineal fuertemente asimétrica. Considérese un usuario con dos compras relevantes ($|A_u| = 2$):
* **Ordenación A (aciertos en posiciones 1 y 3):** $\text{AP@12} = \frac{1}{2} \left(\frac{1}{1} + \frac{2}{3}\right) = 0.8333$.
* **Ordenación B (aciertos en posiciones 5 y 10):** $\text{AP@12} = \frac{1}{2} \left(\frac{1}{5} + \frac{2}{10}\right) = 0.2000$.

Aunque en ambos casos el sistema recupera el 100% de los artículos comprados ($\text{Recall@12} = 1.0$), la ordenación A obtiene una puntuación $+316.6\%$ superior. A diferencia de métricas basadas en curvas ROC o clasificaciones binarias indiferenciadas, esta propiedad castiga con severidad las inversiones de orden en las posiciones de cabecera.

### 8.1.2. Métricas Complementarias de Diagnóstico

Para evaluar de forma desacoplada la etapa de recuperación de candidatos y la etapa de reordenación supervisada, se emplean tres métricas complementarias:

1. **Recall at K (Recall@K):** Mide la cobertura de artículos relevantes capturados en las primeras $K$ posiciones de la lista final:
   $$\text{Recall@K} = \frac{1}{|U|} \sum_{u \in U} \frac{|A_u \cap P_u^{(K)}|}{|A_u|}$$

2. **Hit Rate at K (Hit Rate@K):** Cuantifica la proporción de usuarios que reciben al menos una recomendación acertada dentro del corte $K$:
   $$\text{Hit Rate@K} = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}\left(|A_u \cap P_u^{(K)}| > 0\right)$$

3. **Techo Teórico de Recuperación (Recall@80 de Candidatos):** Dado que la etapa de re-ranking supervisado opera exclusivamente sobre el espacio de candidatos preseleccionados $C_u$ ($|C_u| \le 80$), ningún algoritmo posterior puede clasificar una prenda relevante que no haya sido recuperada previamente. El techo de recuperación se monitoriza como:
   $$\text{Recall@80} = \frac{1}{|U|} \sum_{u \in U} \frac{|A_u \cap C_u|}{|A_u|}$$

Un $\text{Recall@80} \ge 0.65$ garantiza un espacio de búsqueda suficientemente representativo para que los modelos de reordenación alcancen niveles competitivos de MAP@12.

### 8.1.3. Rendimiento de Baselines Clásicos de Referencia

Antes de desarrollar arquitecturas de aprendizaje supervisado, se evaluaron empíricamente cuatro estrategias de referencia (*baselines*) sobre la partición temporal de validación ($W_{104}$, 68.984 usuarios activos):

| Modelo / Estrategia de Referencia | Recall@12 | MAP@12 Local | Cobertura Catálogo | Características Operativas |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline 1: Top-12 Más Vendidos Global** | 0.048 | 0.01120 | 0.011% | Asignación idéntica para toda la población; nula personalización. |
| **Baseline 2: Recompra Pura (Últimas 4 semanas)** | 0.112 | 0.01850 | 11.20% | Ordenación por recencia/frecuencia personal; colapsa en clientes nuevos. |
| **Baseline 3: Co-ocurrencia en Cesta (Item-CF)** | 0.089 | 0.01520 | 8.90% | Filtrado colaborativo ítem-ítem; penalizado por dispersión en compras esporádicas. |
| **LGBMRanker V1 (Supervisado Inicial)** | **0.185** | **0.02680** | **18.50%** | Modelo supervisado LambdaRank sobre 39 variables tabulares. |

*Tabla 8.1: Comparativa cuantitativa de baselines clásicos frente a la primera aproximación supervisada.*


## 8.2. Trayectoria Experimental: Análisis Comparativo de las Iteraciones V0 a V8

El desarrollo del motor de recomendación se estructuró a lo largo de nueve iteraciones técnicas (V0 a V8), evaluadas sistemáticamente tanto en el entorno de validación temporal local ($W_{104}$) como en la plataforma oficial de Kaggle (tableros público y privado sobre 1.371.980 usuarios).

### 8.2.1. Matriz Consolidada de Resultados Experimentales

La Tabla 8.2 resume la evolución cuantitativa, la cobertura de personalización y el rendimiento de ranking para la totalidad de las versiones:

| Versión | Paradigma de Inferencia | Clientes Personalizados | Clientes en Contingencia | MAP@12 Local ($W_{104}$) | Kaggle Public | Kaggle Private | Estado del Modelo |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **V0** | Popularidad Global No Segmentada | 0 (0.0%) | 1.371.980 (100.0%) | 0.01120 | 0.00892 | 0.00910 | Línea base estática |
| **V1** | Muestra Reducida (2k) + Fallback Histórico 2018 | 2.000 (0.14%) | 1.369.980 (99.86%) | 0.00712 | 0.00545 | 0.00567 | Cobertura insuficiente |
| **V2** | Out-of-Core Masivo + Fallback Dinámico 7d por Edad | 278.275 (20.28%) | 1.093.705 (79.72%) | 0.02105 | 0.01726 | 0.01735 | Escala masiva funcional (+206%) |
| **V3** | Desacople Temporal Estricto | 278.275 (20.28%) | 1.093.705 (79.72%) | 0.01943 | 0.01597 | 0.01619 | Contracción de entrenamiento |
| **V4** | L2R Supervisado Multi-Candidato R8 (100 cand.) | 278.275 (20.28%) | 1.093.705 (79.72%) | 0.02341 | 0.01690 | 0.01728 | Desplazamiento por hiper-generación |
| **V5** | Waterfall Estratificado (35d + Superventas por Edad) | 273.166 (19.91%) | 1.098.814 (80.09%) | 0.02703 | 0.02238 | 0.02212 | Erradicación de desfase (+27.5%) |
| **V6** | Waterfall Truncado ($k_p \le 3$) + Reglas $P(B\|A)$ | 273.166 (19.91%) | 1.098.814 (80.09%) | 0.02659 | 0.02134 | 0.02197 | Canibalización por truncamiento rígido |
| **V7** | Waterfall Híbrido No Destructivo ($k_p \le 12$) | 273.166 (19.91%) | 1.098.814 (80.09%) | 0.02805 | 0.02291 | 0.02332 | Rescate causal en huecos libres |
| **V8** | Waterfall Multidimensional (28d + Afinidad + Multi-Semana) | 233.174 (17.00%) | 1.138.806 (83.00%) | **0.02882** | **0.02347** | **0.02386** | **Récord global (+162.2% vs V0)** |

*Tabla 8.2: Evolución integral de métricas de ranking, cobertura de usuarios y resultados en el benchmark oficial de Kaggle.*

> [!NOTE]
> **Nota metodológica sobre los entornos de evaluación:**  
> Las métricas de validación local ($W_{104}$) corresponden al punto de control utilizado durante el entrenamiento y la ingeniería de características en los cuadernos de desarrollo, donde V8 registra $0.02882$ (+157.3% frente a V0 local de $0.01120$). Las columnas Kaggle Public y Private corresponden a la evaluación externa ciega sobre la semana siguiente ($W_{105}$), donde V8 registra $0.02386$ en Private (+162.2% frente a V0 en test de $0.00910$). Ambos valores son complementarios y verifican la estabilidad del ranking fuera de muestra.

### 8.2.2. Análisis Causal de las Transiciones entre Configuraciones

El análisis de los resultados cuantitativos revela los factores determinantes que explicaron las variaciones de rendimiento en cada iteración:

#### 1. Transición V1 a V2: Cobertura Poblacional y Política de Contingencia Dinámica
La entrega inicial V1 arrojó un rendimiento de 0.00545 en Public y 0.00567 en Private debido a dos factores determinantes:
* **Cobertura muestral mínima:** Al limitar la inferencia a 2.000 clientes activos (0.14% del total), el 99.86% de los usuarios evaluados dependió exclusivamente de la política de contingencia.
* **Desfase temporal del catálogo de contingencia:** La política de respaldo asignaba artículos superventas correspondientes al otoño de 2018 (e.g., artículo `0706016001`). En moda rápida, los ciclos de vida de producto oscilan entre 2 y 4 semanas, por lo que prendas de hace dos años carecían de relevancia en septiembre de 2020.

En V2 se procesaron 17.292.800 pares candidato para los 278.275 clientes activos mediante streaming out-of-core con Polars/Arrow en lotes contiguos de 2.5 millones de registros comprimidos con ZSTD nivel 3, manteniendo el consumo en 3.708 MB RSS. Paralelamente, se implementó una función de contingencia dinámica centrada en los últimos 7 días con decaimiento exponencial ($w_t = \exp(-\lambda \Delta t)$, $\lambda = 0.05$, vida media $\approx 13.8$ días) estratificada en 5 cohortes etarias (`<25`, `25-34`, `35-44`, `45-54`, `55+`). Esta modificación incrementó el score oficial a 0.01726 en Public y 0.01735 en Private (+206.0%).

#### 2. Transición V2 a V3: Efecto de la Contracción Muestral por Desacople Temporal Estricto
La versión V2 presentaba solapamiento temporal entre la ventana de características y la etiqueta objetivo, concentrando el 99.78% de la ganancia del modelo en la variable `uxa_repurchase_count`. Para aislar este efecto, la versión V3 desacopló estrictamente las particiones temporales exigiendo que los clientes hubiesen registrado transacciones simultáneas en la semana objetivo $W_{103}$ y en la ventana histórica precedente $W_{100-102}$.

Esta restricción introdujo un grave sesgo de supervivencia (*selection bias*), reduciendo las consultas de entrenamiento de 240.845 a solo 7.360 (pérdida del 96.9% de usuarios activos y del 98.9% de filas, pasando de 5.47 millones a 57.637 pares). Con un volumen tan reducido, el criterio de early stopping con paciencia de 50 rondas detuvo el entrenamiento en la **ronda 2** (`best_iteration = 2`). Un ensamble de dos árboles de decisión carece de capacidad expresiva para ordenar 18.89 millones de pares en inferencia, provocando una regresión en Kaggle a 0.01597 en Public y 0.01619 en Private (-6.7%). El experimento demostró que el aislamiento de fuga temporal debe lograrse mediante formulaciones de retraso (*lag features*) sin podar el soporte muestral indispensable para el aprendizaje de divisiones estables.

#### 3. Transición V3 a V4: Desplazamiento Estacional por Re-ranking Supervisado No Restringido
En la versión V4 se restauró el volumen de entrenamiento a escala completa (240.912 consultas, 5.572.394 pares, 50 rondas) y se amplió el espacio de búsqueda a 100 candidatos por usuario mediante la heurística departamental R8 ($d_u^* = \arg\max_d \text{compras}(u, d)$) y 39 variables explicativas.

A pesar de alcanzar un MAP@12 local aparente de 0.69122 en la muestra de pares candidatos, el score en Kaggle se estancó en 0.01690 en Public y 0.01728 en Private. El análisis de las predicciones identificó la causa: al delegar la asignación de la totalidad de las 12 posiciones al modelo supervisado sin salvaguarda de tendencia, **el 84.8% de los clientes activos no recibió ninguno de los 5 artículos superventas contemporáneos de otoño** (`0924243001`, `0924243002`, `0918522001`, `0923758001`, `0866731001`). Las predicciones quedaron saturadas por artículos secundarios o compras pasadas de verano, desplazando las prendas de alta conversión general.

#### 4. Transición V4 a V5: Estratificación Temporal a 35 Días e Inyección Forzosa de Superventas
Para resolver el desplazamiento observado en V4, se calibró empíricamente el horizonte de recencia personal sobre la partición de validación temporal ($W_{104}$, 68.984 clientes):

| Horizonte Temporal Evaluado | MAP@12 Local ($W_{104}$) | Variación vs 35 Días | Diagnóstico Empírico |
| :---: | :---: | :---: | :--- |
| **7 días** | 0.02301 | -15.3% | Ventana excesivamente estrecha; omite ciclos de rotación quincenales. |
| **14 días** | 0.02576 | -5.2% | Cobertura aceptable pero insuficiente en compradores mensuales. |
| **28 días** | **0.02882** | **+6.6%** | Óptimo para colecciones de entretiempo (adoptado en V8). |
| **35 días (V5)** | **0.02703** | **Referencia** | Filtra liquidaciones de verano de principios de agosto. |
| **42 días** | 0.02687 | -0.6% | Comienza a incorporar prendas estivales fuera de temporada climática. |

*Tabla 8.3: Calibración empírica de ventanas temporales de recencia personal.*

La versión V5 implementó una arquitectura en cascada estratificada: (1) Delimitación de recencia a 35 días exactos; (2) Priorización personal por recencia (`last_purchase_date DESC, purchase_count DESC`); y (3) Relleno obligatorio de posiciones disponibles con superventas de la última semana correspondientes a la cohorte etaria del usuario. Esta formulación redujo los clientes activos sin superventas del 84.8% al 5.36%, y elevó el rendimiento en Kaggle a **0.02238 en Public** y **0.02212 en Private** (+27.5% respecto a V2).

#### 5. Transición V5 a V6: Evaluación del Cupo Restrictivo de Recompra ($k_p \le 3$)
Con el propósito de introducir diversidad en las recomendaciones, la versión V6 limitó a priori las recompras personales a un máximo de 3 artículos ($k_p \le 3$), reservando las posiciones 4 a 7 ($k_c \le 4$) para reglas de asociación de cesta $P(B|A)$ y las posiciones 8 a 12 para superventas estacionales.

La evaluación oficial registró una regresión en Kaggle: **0.02134 en Public** y **0.02197 en Private** (frente a 0.02212 de V5). El análisis empírico reveló que **112.999 clientes activos (41.37% del total)** habían adquirido 4 o más prendas distintas en la ventana de 35 días (con un promedio de 6.8 prendas por usuario). Al restringir el cupo a 3, el sistema descartó arbitrariamente una media de 3.8 recompras directas para sustituirlas por complementos probabilísticos. Dado que la probabilidad de recompra supera empíricamente a la propensión de co-compra, la poda forzada redujo el recall en los usuarios de mayor gasto.

Asimismo, se contrastó una micro-segmentación de contingencia en 30 celdas tridimensionales (edad $\times$ gasto $\times$ canal), la cual degradó el MAP@12 local a 0.02637 debido a una excesiva varianza por tamaño muestral insuficiente (<50 transacciones por celda en 7 días).

#### 6. Transición V6 a V7: Asignación Jerárquica con Preservación Íntegra de Recompras
La iteración V7 reformuló la integración de complementos bajo un principio de asignación prioritaria no destructiva:
* **Preservación total de compras personales ($k_p \le 12$):** Se conservan todas las compras recientes del usuario sin truncamiento artificial.
* **Rescate causal condicionado a capacidad disponible:** Únicamente los usuarios con menos de 12 compras personales (95.49% de los activos) reciben complementos derivados de co-ocurrencia en cesta $P(B|A)$, utilizando exclusivamente los puestos libres ($12 - k_p$, con $k_c \le 6$).
* **Relleno demográfico final:** Si aún restan posiciones vacantes, se incorporan los superventas de la cohorte etaria.

En la validación local sobre $W_{104}$, V7 alcanzó un MAP@12 de 0.02805 (+3.78% vs V5) y elevó el catálogo explorado al 17.59%. En Kaggle, el score aumentó a **0.02291 en Public** y **0.02332 en Private** (+5.42% de incremento neto sobre V5).

#### 7. Transición V7 a V8: Optimización de Ventana, Afinidad Global en Cesta y Regularización Multi-Semana
La versión V8 introdujo cuatro refinamientos analíticos:
1. **Contracción temporal a 28 días exactos (4 semanas):** Se eliminó la quincena final de agosto, mitigando el arrastre de liquidaciones estivales y concentrando el catálogo en prendas de entretiempo y otoño.
2. **Puntuación de Afinidad Global en Cesta:** En lugar de calcular $P(B|A)$ sobre la última compra individual, V8 formuló una puntuación que integra el historial reciente:
   $$S(u, c) = \sum_{a \in H_u^{(5)}} \lambda_a^{\frac{t_{\text{ref}} - t_a}{7.0}} \cdot W_{\text{pair}}(a, c)$$
   donde $H_u^{(5)}$ representa los últimos 5 artículos adquiridos, $\lambda_a = 0.80$ es un factor de atenuación semanal y $W_{\text{pair}}(a, c)$ cuantifica la co-ocurrencia temporalizada:
   $$W_{\text{pair}}(A, B) = \sum_{t \in T(A, B)} 0.85^{\frac{d_A(t) + d_B(t)}{14.0}} \quad \text{con } N(A, B) \ge 3$$
3. **Regularización Bayesiana Multi-Semana en Contingencia:** Para evitar la exposición a quiebres de inventario (*stock-out*) de artículos calculados sobre un único período de 7 días, se aplicó un decaimiento geométrico con factor $\gamma = 0.12$ sobre 3 semanas históricas:
   $$\text{Score}_{\text{pop}}(i, c_{\text{age}}) = \text{Ventas}_{w=0}(i, c_{\text{age}}) + 0.12 \cdot \text{Ventas}_{w=1}(i, c_{\text{age}}) + 0.0144 \cdot \text{Ventas}_{w=2}(i, c_{\text{age}})$$
   Bajo este esquema, la varianza del estimador decae fuertemente: los pesos $\gamma = 0.12$ y $\gamma^2 = 0.0144$ aportan un 88.1% a la semana de corte, atenuando fluctuaciones espurias de inventario sin diluir la tendencia contemporánea.
4. **Descarte empírico de compras lejanas:** Se evaluó la hipótesis de incorporar compras antiguas (días 29 a 60) para clientes poco activos; las pruebas locales mostraron una caída de MAP@12 de 0.02825 a 0.02793 (-1.13%), confirmando que prendas de hace dos meses pertenecen a colecciones obsoletas.

### 8.2.3. Benchmark Comparativo Local de 4 Vías ($W_{104}$) y Correlación con Kaggle

Para contrastar el comportamiento de las cuatro variantes avanzadas sobre los 68.984 clientes de la semana retenida $W_{104}$, se ejecutó el protocolo de evaluación directa documentado en [`results/tables/eval_v5_v6_v7_v8_local.json`](results/tables/eval_v5_v6_v7_v8_local.json):

| Métrica de Ranking | V5 (Cascada 35d) | V6 (Truncado $k_p \le 3$) | V7 (Híbrido No Destructivo) | V8 (Afinidad Global + Multi-Semana) | $\Delta$ V8 vs V7 | $\Delta$ V8 vs V5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAP@12 Local** | 0.02703 | 0.02659 | 0.02805 | **0.02882** | **+2.73%** | **+6.62%** |
| **Recall@12** | 0.05640 | 0.05422 | 0.06040 | **0.06260** | **+3.65%** | **+10.99%** |
| **Hit Rate@12** | 0.11187 | 0.10737 | 0.11561 | **0.11945** | **+3.32%** | **+6.78%** |
| **Catalog Coverage** | 16.95% | 14.32% | **17.59%** | 17.03% | -0.56% | +0.08% |
| **Tiempo de Inferencia Local** | 3.20 s | 3.37 s | 3.48 s | 4.97 s | +1.49 s | +1.77 s |

*Tabla 8.4: Benchmark comparativo de 4 vías sobre la partición de validación temporal W104.*

La consistencia entre la evaluación local y las puntuaciones oficiales en Kaggle exhibe un coeficiente de transferencia altamente predecible:
* **V5:** Local 0.02703 $\to$ Private 0.02212 (ratio: 0.8183).
* **V6:** Local 0.02659 $\to$ Private 0.02197 (ratio: 0.8262).
* **V7:** Local 0.02805 $\to$ Private 0.02332 (ratio: 0.8314).
* **V8:** Local 0.02882 $\to$ Private 0.02386 (ratio: 0.8279).

La estabilidad del ratio ($0.826 \pm 0.005$) confirma que la partición local $W_{104}$ no presenta desalineamiento de distribución (*distribution shift*) frente a la partición privada de Kaggle ($W_{105}$), validando la ausencia de fuga retrospectiva y asegurando que las ganancias de ranking observadas en el entorno local se trasladan de forma fiable a producción.

### 8.2.4. Verificación Criptográfica de Integridad de las Entregas Oficiales

Para asegurar la trazabilidad y la reproducibilidad exacta de los resultados evaluados en la plataforma de competición, la totalidad de los archivos de predicción se encuentran inventariados en [`results/SUBMISSIONS_MANIFEST.json`](results/SUBMISSIONS_MANIFEST.json). La Tabla 8.5 detalla los identificadores criptográficos y parámetros físicos de las versiones principales:

| Versión | Tamaño CSV (Bytes) | Tamaño GZIP (Bytes) | SHA-256 (Archivo CSV) | SHA-256 (Archivo GZIP) | Private Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **V5** | 270.280.083 | 56.662.210 | `6516C735AA30038520BCBED85756250D55AE4D388FDF1D21633573175740C2E2` | `7CE096991256B21D75276192CFCFDFD566C07BA8B32E0A586C8C160F7F0F57C9` | 0.02212 |
| **V6** | 270.280.083 | 62.687.462 | `D06CEE34741ED69BDF6EA8975CE4775217100E42DEA5D737411028A7DBBB5E9D` | `D3EB5745BE623FB19AD1F209E4B5B157D31F7EFE3B0095D6F48963DB682CA0B7` | 0.02197 |
| **V7** | 270.280.083 | 65.604.405 | `F0DA19502561331D2962876A577FC054EFC962763D7D537958CDF9948D160DC8` | `AD32F0A067514FFE2D4A585704E680B55C9BE239DF029E507588437692CFC862` | 0.02332 |
| **V8** | 270.280.083 | 60.060.309 | `E34B230E3C0744C335F6E809175C115C63BB73BB5B10A9A7A2A5DBA7C53DF106` | `C766D66BF7649D713AA55D0C03C35B9D95B04D34665DBF931768633E902E635C` | **0.02386** |

*Tabla 8.5: Registro de sumas de verificación criptográfica (SHA-256) de las sumisiones evaluadas en Kaggle.*


## 8.3. Disparidad de Rendimiento por Cohortes de Actividad del Usuario

El comportamiento agregado de la métrica MAP@12 encubre diferencias cuantitativas profundas en función del nivel de actividad histórica del comprador. La evaluación sobre la semana de prueba retenida $W_{104}$ (68.984 compradores activos) permite desagregar el rendimiento según el volumen de compras del cliente:

| Cohorte de Clientes | Tamaño Muestral ($|U|$) | Proporción de Población | MAP@12 Local | Recall@12 | Hit Rate@12 | Comportamiento del Motor |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cohorte General Activa** | 68.984 | 100.0% | 0.02520 | 0.05120 | 10.03% | Rendimiento promedio ponderado sobre activos. |
| **Compradores Frecuentes ($\ge 4$ compras)** | 17.267 | 25.03% | **0.04930** | **0.09750** | **18.24%** | Fuerte captura por recompra directa y co-ocurrencia (+95.6% MAP@12). |
| **Compradores Esporádicos ($1-3$ compras)** | 51.717 | 74.97% | 0.01710 | 0.03570 | 7.29% | Alta dependencia del relleno de superventas estacionales. |
| **Segmento Inactivo / Cold-Start** | 1.138.806 | 83.00% (del total) | — | — | — | Asignación demográfica bayesiana por edad (latencia <0.01 ms). |

*Tabla 8.6: Rendimiento comparativo del sistema V8 desagregado por cohortes de actividad en la partición W104.*

### 8.3.1. Sensibilidad Hiperbólica de MAP@12 a la Recurrencia Transaccional

La disparidad observada entre la cohorte frecuente (MAP@12 = 0.04930) y la esporádica (0.01710) obedece a dos propiedades estructurales del sistema:
1. **Densidad del perfil de usuario:** Los clientes frecuentes aportan una señal de recompra reciente de alta fidelidad. Al ordenar estas prendas en los primeros 3 a 5 rangos ($k \in [1, 5]$), la precisión posicional $P(k)$ alcanza factores entre $1.0$ y $0.20$.
2. **Impacto del denominador en clientes con una sola compra:** Para un cliente con una única compra en la semana de prueba ($|A_u| = 1$), el término de normalización es $\min(1, 12) = 1$. Si su artículo adquirido se ubica en el puesto 10 mediante la política de superventas, el acierto aporta $AP@12 = 1/10 = 0.10$. En cambio, para un cliente frecuente con 4 artículos adquiridos donde el sistema acierta en las posiciones 1 y 3, su aportación se eleva a:
   $$AP@12 = \frac{1}{4} \left(\frac{1}{1} + \frac{2}{3}\right) = \frac{1.666}{4} = 0.4166$$
   Esto explica por qué la retención y la precisión de ranking sobre el cuartil más activo de usuarios determina casi el 50% de la métrica agregada.


## 8.4. Determinismo Numérico, Concurrencia Multihilo y Sensibilidad de MAP@12

Durante la verificación de reproducibilidad de la iteración V5, la regeneración del archivo de predicciones generó un fichero idéntico en número de filas (1.371.980), estructura física (270.280.083 bytes) y formato que, al ser remitido a Kaggle, registró una variación residual de **tres cienmilésimas** en la métrica privada (0.02209 frente al valor original de 0.02212).

### 8.4.1. Análisis Matemático de la Sensibilidad Posicional en Grandes Poblaciones

La variación observada se explica formalmente analizando la permutación de artículos en posiciones intermedias o bajas de la recomendación. Consideremos dos artículos candidatos con idéntica puntuación de soporte que permutan sus puestos 10 y 11:
* Si el artículo relevante se ubica en el **puesto 10**, su precisión de corte es $P(10) = 1/10 = 0.1000$.
* Si el artículo relevante desciende al **puesto 11**, su precisión de corte disminuye a $P(11) = 1/11 = 0.0909$.

La variación individual en Average Precision es:
$$\Delta AP@12 = \frac{1}{10} - \frac{1}{11} = \frac{1}{110} \approx 0.00909$$

Al promediar este desplazamiento sobre el universo total de evaluación ($|U| = 1.371.980$ usuarios), el impacto en MAP@12 derivado de que $N_{\text{afectados}}$ clientes sufran dicha permutación viene dado por:
$$\Delta MAP@12 = \frac{\Delta AP@12 \times N_{\text{afectados}}}{|U|} = \frac{0.00909 \times N_{\text{afectados}}}{1.371.980}$$

Para producir una oscilación agregada de $\Delta MAP@12 = 0.00003$, basta con que:
$$N_{\text{afectados}} = \frac{0.00003 \times 1.371.980}{0.00909} \approx 4.527 \text{ usuarios}$$

Dado que 4.527 clientes representan únicamente el **0.33% de la población**, una micro-inversión en artículos de soporte empatado en una fracción mínima de usuarios genera la discrepancia registrada.

### 8.4.2. Causa Raíz: Concurrencia Multihilo en Motores Columnares (DuckDB)

El análisis fila por fila en memoria con Polars aisló la causa técnica:
* En el subconjunto de superventas por edad de la semana 104, dos prendas dentro de la cohorte de 25 a 34 años (`0865799006` y `0935541001`) registraron **exactamente 163 compras cada una**.
* Al ejecutar la agregación en DuckDB con escaneo multihilo sobre múltiples núcleos de CPU, la consulta:
  ```sql
  -- Consulta original sin clave secundaria determinista
  QUALIFY row_number() OVER (PARTITION BY age_bin ORDER BY n DESC) <= 12
  ```
  carecía de un criterio de desempate explícito, provocando que el orden de consolidación de buffers en memoria variara según el micro-agendamiento del sistema operativo.

### 8.4.3. Corrección Arquitectónica: Endurecimiento Determinista

Para garantizar determinismo estricto bit a bit en ejecuciones sucesivas, se actualizaron las consultas de agregación en `src/evaluation/submission.py` inyectando ordenamientos léxicos inmutables como clave secundaria de desempate:
```sql
-- Consulta endurecida con ordenamiento léxico inmutable
QUALIFY row_number() OVER (PARTITION BY age_bin ORDER BY n DESC, article_id ASC) <= 12;
SELECT customer_id, list(f_art ORDER BY last_d DESC, cnt DESC, f_art ASC);
```
Con esta modificación, las ejecuciones subsiguientes reproducen de forma exacta e idéntica los resúmenes SHA-256 registrados en el manifiesto oficial.


## 8.5. Aislamiento Estructural de los Tres Modos de Ejecución

Para prevenir interferencias operativas entre el desarrollo experimental, el servicio productivo y la generación masiva de predicciones, el repositorio desacopla formalmente tres modos de ejecución mediante namespaces disjuntos:

![Figura 8.1: Aislamiento estructural y trazabilidad de los tres modos de ejecución del sistema (entorno de pruebas en muestra, microservicio REST y pipeline de replicación masiva out-of-core).](../results/figures/diag_07_execution_modes_isolation.png)

*Figura 8.1: Diagrama de arquitectura que ilustra el desacoplamiento de entornos operativos en el repositorio. El Modo 1 garantiza validaciones rápidas para integración continua; el Modo 2 empaqueta el servicio en contenedor Docker para inferencia online de baja latencia; y el Modo 3 procesa la totalidad del catálogo en flujo continuo asegurando la integridad de los resultados oficiales.*

La Tabla 8.7 resume las especificaciones operativas y de rendimiento de cada modo:

| Dimensión Operativa | Modo 1: Muestra Rápida (CI/CD) | Modo 2: Microservicio REST (Producción) | Modo 3: Inferencia Masiva (Benchmark) |
| :--- | :---: | :---: | :---: |
| **Namespace Aislado** | `sample/` (`data_processed/sample/`) | Contenedor Docker / Runtime FastAPI | `results/archive_submissions/` |
| **Población Objetivo** | 2.000 clientes activos | Cliente individual síncrono (`customer_id`) | 1.371.980 clientes oficiales |
| **Transacciones Procesadas** | Muestra representativa reducida | Matriz de afinidad y prior en memoria | 31.78 millones de registros |
| **Presupuesto de Memoria** | $< 400$ MB RSS | $< 500$ MB RSS | $< 2.000$ MB RSS (1.435.8 MB observados) |
| **Tiempo de Ejecución** | $< 60$ segundos | Latencia p95 $< 50$ ms | 47.21 segundos (254.731 clientes/s) |
| **Mecanismo de Verificación** | Pruebas unitarias y cobertura `pytest` | Endpoint de salud y validación de esquema | Manifiesto criptográfico SHA-256 / MD5 |

*Tabla 8.7: Matriz comparativa de requerimientos y especificaciones operativas entre los tres modos de ejecución.*


## 8.6. Principios de Ingeniería para Sistemas de Recomendación en Moda Rápida

La trayectoria empírica del proyecto permite consolidar cuatro principios rectores para el diseño y puesta en producción de sistemas de recomendación en comercio minorista con catálogos de alta rotación:

1. **Escalabilidad out-of-core en arquitecturas de hardware convencional:**  
   El uso de estructuras de datos columnares (Apache Arrow) junto con motores analíticos vectorizados (Polars y DuckDB) permite manipular decenas de millones de registros transaccionales sin requerir infraestructura distribuida pesada (e.g., clústeres Spark). La partición por lotes con escritura directa a disco en formato Parquet comprimido con ZSTD mantiene el consumo de memoria por debajo de 2.0 GB RSS durante todo el ciclo de cómputo.

2. **Horizonte temporal restringido frente a la obsolescencia de catálogo:**  
   En comercio de moda rápida, los historiales transaccionales dilatados introducen ruido estacional perjudicial. Limitar la ventana de observación a 28 días exactos purga las colecciones climáticas anteriores (verano) y alinea la señal de compra con la demanda contemporánea (otoño), mejorando la precisión de ranking frente a ventanas de 35 o 60 días.

3. **Jerarquía de inferencia con preservación de señales de alta certidumbre:**  
   La probabilidad de recompra reciente directa supera sustancialmente a las probabilidades derivadas de filtrado colaborativo o reglas de asociación de cesta. Los mecanismos de complementariedad ($P(B|A)$) deben utilizarse exclusivamente para rellenar la capacidad disponible del escaparate ($12 - k_p$), evitando en todo momento truncar o canibalizar las compras personales auténticas del usuario.

4. **Regularización bayesiana en estrategias de contingencia:**  
   Para la mayoría inactiva de la población (83,0% en cold start bajo la ventana de 28 días adoptada en V8 —233.174 clientes activos sobre 1.371.980 totales—, frente al 80,09% registrado en la ventana cruda general de 35 días), la estratificación demográfica por cohortes de edad proporciona un compromiso adecuado entre personalización estilística y estabilidad estadística. La aplicación de un suavizado multi-semana con decaimiento exponencial geométrico ($\gamma = 0.12$) protege al sistema frente a quiebres de inventario monosemanales sin desdibujar las tendencias de venta inmediatas.
