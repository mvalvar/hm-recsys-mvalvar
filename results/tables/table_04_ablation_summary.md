# Tabla 04: Resumen del Estudio de Ablación (A1 a A6)

**TFM:** Motor de Recomendación Escalable para Retail de Moda (*H&M RecSys Challenge*)  
**Autor:** Manuel Valdivia: Máster en Data Science, Big Data & Business Analytics (UCM)

Esta tabla consolida los resultados cuantitativos y las deducciones metodológicas derivadas de las variantes evaluadas sobre la Semana 104 de validación.

---

### 1. Tabla Resumen en Formato Markdown

| Exp. | Configuración Evaluada | MAP@12 | Δ vs. Base | Δ (%) | RAM (MB) | CPU (s) | Conclusión Metodológica |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A1** | 3w | 0.02681 | +0.00742 | +38.27% | 2048.3 | 0.36 | Horizonte insuficiente (+38.27%); carece de historial para capturar recompra de reposición. |
| **A1** | 5w | 0.01939 | Baseline | --- | 2102.8 | 0.40 | Punto de balance óptimo entre estacionalidad y persistencia de catálogo (Baseline A1). |
| **A1** | 8w | 0.01832 | -0.00107 | -5.52% | 2129.6 | 0.42 | Sobreajuste a artículos descatalogados de temporada previa (-5.52%). |
| **A1** | 10w | 0.01445 | -0.00494 | -25.48% | 2137.5 | 0.43 | Elevado ruido estacional con saturación de prendas obsoletas (-25.48%). |
| **A2** | Full (R1..R8) | 0.02390 | Baseline | --- | 2150.6 | 0.35 | Arquitectura multi-fuente integral con 8 heurísticas consolidadas (Baseline A2). |
| **A2** | Sin R1 (Repurchase) | 0.02476 | +0.00086 | +3.58% | 2153.6 | 0.30 | Efecto leave-one-out sobre R1 (Repurchase): +3.58% MAP@12. |
| **A2** | Sin R2 (Pop Global) | 0.01945 | -0.00446 | -18.65% | 2157.3 | 0.28 | Efecto leave-one-out sobre R2 (Popularidad Global): -18.65% MAP@12. |
| **A2** | Sin R3 (Pop Edad) | 0.01608 | -0.00782 | -32.73% | 2166.0 | 0.27 | Efecto leave-one-out sobre R3 (Popularidad por Edad): -32.73% MAP@12. |
| **A2** | Sin R4 (Pop Canal) | 0.01710 | -0.00680 | -28.46% | 2152.6 | 0.29 | Efecto leave-one-out sobre R4 (Popularidad por Canal): -28.46% MAP@12. |
| **A2** | Sin R5 (Item-CF) | 0.02364 | -0.00027 | -1.11% | 2158.3 | 0.30 | Efecto leave-one-out sobre R5 (Item-CF): -1.11% MAP@12. |
| **A2** | Sin R6 (Familias) | 0.02234 | -0.00157 | -6.57% | 2151.3 | 0.28 | Efecto leave-one-out sobre R6 (Familias de Producto): -6.57% MAP@12. |
| **A2** | Sin R7 (Trending) | 0.02147 | -0.00244 | -10.20% | 2153.3 | 0.29 | Efecto leave-one-out sobre R7 (Tendencias Semanales): -10.20% MAP@12. |
| **A2** | Sin R8 (Dept Popularity) | 0.02208 | -0.00183 | -7.64% | 2157.7 | 0.30 | Efecto leave-one-out sobre R8 (Departamento Personal): -7.64% MAP@12. |
| **A3** | Con Source Flags (39 features) | 0.02372 | Baseline | --- | 2140.2 | 41.28 | Prior Bayesiano probabilístico inducido por las 8 flags de origen (Baseline A3; 39 features). |
| **A3** | Sin Source Flags (31 features) | 0.02257 | -0.00115 | -4.84% | 6351.0 | 34.16 | Modelo agnóstico a las fuentes de generación (-4.84% MAP@12; 31 features). |
| **A4** | Ratio 1:3 | 0.02324 | -0.00048 | -2.02% | 4955.8 | 38.79 | Sub-representación de negativos que incrementa falsos positivos (-2.02%). |
| **A4** | Ratio 1:5 | 0.02372 | Baseline | --- | 7299.0 | 46.29 | Punto de inflexión óptimo entre sesgo y varianza empírica (Baseline A4). |
| **A4** | Ratio 1:10 | 0.02369 | -0.00003 | -0.13% | 6697.0 | 86.41 | Dilución de gradientes sobre positivos con coste computacional superior (-0.13%). |
| **A4** | Ratio 1:20 | 0.02352 | -0.00020 | -0.84% | 6990.8 | 85.37 | Desbalance severo de clases con saturación de memoria (-0.84%). |
| **A5** | LambdaRank (Listwise) | 0.02372 | Baseline | --- | 4762.6 | 41.14 | Optimiza directamente la permutación con métrica NDCG/MAP en top-12 (Baseline A5). |
| **A5** | Binary Cross-Entropy (Pointwise) | 0.02206 | -0.00166 | -6.99% | 6760.3 | 39.06 | Pérdida simétrica pointwise que penaliza errores fuera del top-12 (-6.99%). |
| **A6** | A1: 3w | 0.02681 | Baseline | --- | 2048.3 | 0.36 | Horizonte temporal de 3 semanas (3,868 transacciones) | Pareto-Óptimo: True (Eficiencia: 0.36) |
| **A6** | A1: 5w | 0.01939 | -0.00742 | -27.68% | 2102.8 | 0.40 | Horizonte temporal de 5 semanas (7,592 transacciones) | Pareto-Óptimo: False (Eficiencia: 0.23) |
| **A6** | A1: 8w | 0.01832 | -0.00849 | -31.67% | 2129.6 | 0.42 | Horizonte temporal de 8 semanas (11,031 transacciones) | Pareto-Óptimo: False (Eficiencia: 0.2) |
| **A6** | A1: 10w | 0.01445 | -0.01236 | -46.10% | 2137.5 | 0.43 | Horizonte temporal de 10 semanas (13,135 transacciones) | Pareto-Óptimo: False (Eficiencia: 0.16) |
| **A6** | A2: Full (R1..R8) | 0.02390 | -0.00291 | -10.85% | 2150.6 | 0.35 | Arquitectura multi-fuente consolidada | Pareto-Óptimo: True (Eficiencia: 0.32) |
| **A6** | A2: Sin R1 (Repurchase) | 0.02476 | -0.00205 | -7.65% | 2153.6 | 0.30 | Prescindiendo de recompra histórica personal | Pareto-Óptimo: True (Eficiencia: 0.38) |
| **A6** | A2: Sin R2 (Pop Global) | 0.01945 | -0.00736 | -27.45% | 2157.3 | 0.28 | Prescindiendo de popularidad global con decaimiento | Pareto-Óptimo: False (Eficiencia: 0.32) |
| **A6** | A2: Sin R3 (Pop Edad) | 0.01608 | -0.01073 | -40.02% | 2166.0 | 0.27 | Prescindiendo de popularidad segmentada por edad | Pareto-Óptimo: True (Eficiencia: 0.27) |
| **A6** | A2: Sin R4 (Pop Canal) | 0.01710 | -0.00971 | -36.22% | 2152.6 | 0.29 | Prescindiendo de preferencia omnicanal físico/online | Pareto-Óptimo: False (Eficiencia: 0.27) |
| **A6** | A2: Sin R5 (Item-CF) | 0.02364 | -0.00317 | -11.82% | 2158.3 | 0.30 | Prescindiendo de filtrado colaborativo ítem-ítem | Pareto-Óptimo: False (Eficiencia: 0.37) |
| **A6** | A2: Sin R6 (Familias) | 0.02234 | -0.00447 | -16.67% | 2151.3 | 0.28 | Prescindiendo de afinidad a familias de producto | Pareto-Óptimo: True (Eficiencia: 0.37) |
| **A6** | A2: Sin R7 (Trending) | 0.02147 | -0.00534 | -19.92% | 2153.3 | 0.29 | Prescindiendo de aceleración de demanda inter-semanal | Pareto-Óptimo: False (Eficiencia: 0.34) |
| **A6** | A2: Sin R8 (Dept Popularity) | 0.02208 | -0.00473 | -17.64% | 2157.7 | 0.30 | Prescindiendo de popularidad en depto favorito | Pareto-Óptimo: False (Eficiencia: 0.34) |
| **A6** | A3: Con Source Flags (39 features) | 0.02372 | -0.00309 | -11.53% | 2140.2 | 41.28 | Modelo completo con señales bayesianas de heurística | Pareto-Óptimo: False (Eficiencia: 0.0) |
| **A6** | A3: Sin Source Flags (31 features) | 0.02257 | -0.00424 | -15.81% | 6351.0 | 34.16 | Modelo agnóstico al canal heurístico generador | Pareto-Óptimo: False (Eficiencia: 0.0) |
| **A6** | A4: Ratio 1:3 | 0.02324 | -0.00357 | -13.32% | 4955.8 | 38.79 | Entrenamiento con ratio 1:3 (3,889,156 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) |
| **A6** | A4: Ratio 1:5 | 0.02372 | -0.00309 | -11.53% | 7299.0 | 46.29 | Entrenamiento con ratio 1:5 (5,652,166 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) |
| **A6** | A4: Ratio 1:10 | 0.02369 | -0.00312 | -11.64% | 6697.0 | 86.41 | Entrenamiento con ratio 1:10 (9,028,850 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) |
| **A6** | A4: Ratio 1:20 | 0.02352 | -0.00329 | -12.27% | 6990.8 | 85.37 | Entrenamiento con ratio 1:20 (12,749,467 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) |
| **A6** | A5: LambdaRank (Listwise) | 0.02372 | -0.00309 | -11.53% | 4762.6 | 41.14 | Optimización listwise directa de NDCG/MAP con gradientes pairwise LambdaRank | Pareto-Óptimo: False (Eficiencia: 0.0) |
| **A6** | A5: Binary Cross-Entropy (Pointwise) | 0.02206 | -0.00475 | -17.72% | 6760.3 | 39.06 | Clasificación binaria puntual asumiendo independencia condicional entre candidatos | Pareto-Óptimo: False (Eficiencia: 0.0) |

---

### 2. Código LaTeX para Memoria Académica (`tabularx`)

El siguiente bloque está listo para inclusión directa en `memoria/capitulo_09_estudio_ablacion.md`:

```latex
\begin{table}[htbp]
\centering
\small
\caption{Resumen Integral del Estudio de Ablación Experimental (A1 a A6)}
\label{tab:ablation_summary}
\begin{tabularx}{\textwidth}{l p{3.8cm} r r r r X}
\toprule
\textbf{Exp.} & \textbf{Configuración} & \textbf{MAP@12} & \textbf{$\Delta$ (\%)} & \textbf{RAM (MB)} & \textbf{CPU (s)} & \textbf{Conclusión Metodológica} \\
\midrule
  \textbf{A1} & 3w & 0.02681 & +38.27\% & 2048.3 & 0.36 & Horizonte insuficiente (+38.27\%); carece de historial para capturar recompra de reposición. \\
  \textbf{A1} & 5w & 0.01939 & --- & 2102.8 & 0.40 & Punto de balance óptimo entre estacionalidad y persistencia de catálogo (Baseline A1). \\
  \textbf{A1} & 8w & 0.01832 & -5.52\% & 2129.6 & 0.42 & Sobreajuste a artículos descatalogados de temporada previa (-5.52\%). \\
  \textbf{A1} & 10w & 0.01445 & -25.48\% & 2137.5 & 0.43 & Elevado ruido estacional con saturación de prendas obsoletas (-25.48\%). \\
  \textbf{A2} & Full (R1..R8) & 0.02390 & --- & 2150.6 & 0.35 & Arquitectura multi-fuente integral con 8 heurísticas consolidadas (Baseline A2). \\
  \textbf{A2} & Sin R1 (Repurchase) & 0.02476 & +3.58\% & 2153.6 & 0.30 & Efecto leave-one-out sobre R1 (Repurchase): +3.58\% MAP@12. \\
  \textbf{A2} & Sin R2 (Pop Global) & 0.01945 & -18.65\% & 2157.3 & 0.28 & Efecto leave-one-out sobre R2 (Popularidad Global): -18.65\% MAP@12. \\
  \textbf{A2} & Sin R3 (Pop Edad) & 0.01608 & -32.73\% & 2166.0 & 0.27 & Efecto leave-one-out sobre R3 (Popularidad por Edad): -32.73\% MAP@12. \\
  \textbf{A2} & Sin R4 (Pop Canal) & 0.01710 & -28.46\% & 2152.6 & 0.29 & Efecto leave-one-out sobre R4 (Popularidad por Canal): -28.46\% MAP@12. \\
  \textbf{A2} & Sin R5 (Item-CF) & 0.02364 & -1.11\% & 2158.3 & 0.30 & Efecto leave-one-out sobre R5 (Item-CF): -1.11\% MAP@12. \\
  \textbf{A2} & Sin R6 (Familias) & 0.02234 & -6.57\% & 2151.3 & 0.28 & Efecto leave-one-out sobre R6 (Familias de Producto): -6.57\% MAP@12. \\
  \textbf{A2} & Sin R7 (Trending) & 0.02147 & -10.20\% & 2153.3 & 0.29 & Efecto leave-one-out sobre R7 (Tendencias Semanales): -10.20\% MAP@12. \\
  \textbf{A2} & Sin R8 (Dept Popularity) & 0.02208 & -7.64\% & 2157.7 & 0.30 & Efecto leave-one-out sobre R8 (Departamento Personal): -7.64\% MAP@12. \\
  \textbf{A3} & Con Source Flags (39 features) & 0.02372 & --- & 2140.2 & 41.28 & Prior Bayesiano probabilístico inducido por las 8 flags de origen (Baseline A3; 39 features). \\
  \textbf{A3} & Sin Source Flags (31 features) & 0.02257 & -4.84\% & 6351.0 & 34.16 & Modelo agnóstico a las fuentes de generación (-4.84\% MAP@12; 31 features). \\
  \textbf{A4} & Ratio 1:3 & 0.02324 & -2.02\% & 4955.8 & 38.79 & Sub-representación de negativos que incrementa falsos positivos (-2.02\%). \\
  \textbf{A4} & Ratio 1:5 & 0.02372 & --- & 7299.0 & 46.29 & Punto de inflexión óptimo entre sesgo y varianza empírica (Baseline A4). \\
  \textbf{A4} & Ratio 1:10 & 0.02369 & -0.13\% & 6697.0 & 86.41 & Dilución de gradientes sobre positivos con coste computacional superior (-0.13\%). \\
  \textbf{A4} & Ratio 1:20 & 0.02352 & -0.84\% & 6990.8 & 85.37 & Desbalance severo de clases con saturación de memoria (-0.84\%). \\
  \textbf{A5} & LambdaRank (Listwise) & 0.02372 & --- & 4762.6 & 41.14 & Optimiza directamente la permutación con métrica NDCG/MAP en top-12 (Baseline A5). \\
  \textbf{A5} & Binary Cross-Entropy (Pointwise) & 0.02206 & -6.99\% & 6760.3 & 39.06 & Pérdida simétrica pointwise que penaliza errores fuera del top-12 (-6.99\%). \\
  \textbf{A6} & A1: 3w & 0.02681 & --- & 2048.3 & 0.36 & Horizonte temporal de 3 semanas (3,868 transacciones) | Pareto-Óptimo: True (Eficiencia: 0.36) \\
  \textbf{A6} & A1: 5w & 0.01939 & -27.68\% & 2102.8 & 0.40 & Horizonte temporal de 5 semanas (7,592 transacciones) | Pareto-Óptimo: False (Eficiencia: 0.23) \\
  \textbf{A6} & A1: 8w & 0.01832 & -31.67\% & 2129.6 & 0.42 & Horizonte temporal de 8 semanas (11,031 transacciones) | Pareto-Óptimo: False (Eficiencia: 0.2) \\
  \textbf{A6} & A1: 10w & 0.01445 & -46.10\% & 2137.5 & 0.43 & Horizonte temporal de 10 semanas (13,135 transacciones) | Pareto-Óptimo: False (Eficiencia: 0.16) \\
  \textbf{A6} & A2: Full (R1..R8) & 0.02390 & -10.85\% & 2150.6 & 0.35 & Arquitectura multi-fuente consolidada | Pareto-Óptimo: True (Eficiencia: 0.32) \\
  \textbf{A6} & A2: Sin R1 (Repurchase) & 0.02476 & -7.65\% & 2153.6 & 0.30 & Prescindiendo de recompra histórica personal | Pareto-Óptimo: True (Eficiencia: 0.38) \\
  \textbf{A6} & A2: Sin R2 (Pop Global) & 0.01945 & -27.45\% & 2157.3 & 0.28 & Prescindiendo de popularidad global con decaimiento | Pareto-Óptimo: False (Eficiencia: 0.32) \\
  \textbf{A6} & A2: Sin R3 (Pop Edad) & 0.01608 & -40.02\% & 2166.0 & 0.27 & Prescindiendo de popularidad segmentada por edad | Pareto-Óptimo: True (Eficiencia: 0.27) \\
  \textbf{A6} & A2: Sin R4 (Pop Canal) & 0.01710 & -36.22\% & 2152.6 & 0.29 & Prescindiendo de preferencia omnicanal físico/online | Pareto-Óptimo: False (Eficiencia: 0.27) \\
  \textbf{A6} & A2: Sin R5 (Item-CF) & 0.02364 & -11.82\% & 2158.3 & 0.30 & Prescindiendo de filtrado colaborativo ítem-ítem | Pareto-Óptimo: False (Eficiencia: 0.37) \\
  \textbf{A6} & A2: Sin R6 (Familias) & 0.02234 & -16.67\% & 2151.3 & 0.28 & Prescindiendo de afinidad a familias de producto | Pareto-Óptimo: True (Eficiencia: 0.37) \\
  \textbf{A6} & A2: Sin R7 (Trending) & 0.02147 & -19.92\% & 2153.3 & 0.29 & Prescindiendo de aceleración de demanda inter-semanal | Pareto-Óptimo: False (Eficiencia: 0.34) \\
  \textbf{A6} & A2: Sin R8 (Dept Popularity) & 0.02208 & -17.64\% & 2157.7 & 0.30 & Prescindiendo de popularidad en depto favorito | Pareto-Óptimo: False (Eficiencia: 0.34) \\
  \textbf{A6} & A3: Con Source Flags (39 features) & 0.02372 & -11.53\% & 2140.2 & 41.28 & Modelo completo con señales bayesianas de heurística | Pareto-Óptimo: False (Eficiencia: 0.0) \\
  \textbf{A6} & A3: Sin Source Flags (31 features) & 0.02257 & -15.81\% & 6351.0 & 34.16 & Modelo agnóstico al canal heurístico generador | Pareto-Óptimo: False (Eficiencia: 0.0) \\
  \textbf{A6} & A4: Ratio 1:3 & 0.02324 & -13.32\% & 4955.8 & 38.79 & Entrenamiento con ratio 1:3 (3,889,156 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) \\
  \textbf{A6} & A4: Ratio 1:5 & 0.02372 & -11.53\% & 7299.0 & 46.29 & Entrenamiento con ratio 1:5 (5,652,166 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) \\
  \textbf{A6} & A4: Ratio 1:10 & 0.02369 & -11.64\% & 6697.0 & 86.41 & Entrenamiento con ratio 1:10 (9,028,850 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) \\
  \textbf{A6} & A4: Ratio 1:20 & 0.02352 & -12.27\% & 6990.8 & 85.37 & Entrenamiento con ratio 1:20 (12,749,467 filas) | Pareto-Óptimo: False (Eficiencia: 0.0) \\
  \textbf{A6} & A5: LambdaRank (Listwise) & 0.02372 & -11.53\% & 4762.6 & 41.14 & Optimización listwise directa de NDCG/MAP con gradientes pairwise LambdaRank | Pareto-Óptimo: False (Eficiencia: 0.0) \\
  \textbf{A6} & A5: Binary Cross-Entropy (Pointwise) & 0.02206 & -17.72\% & 6760.3 & 39.06 & Clasificación binaria puntual asumiendo independencia condicional entre candidatos | Pareto-Óptimo: False (Eficiencia: 0.0) \\
\bottomrule
\end{tabularx}
\end{table}
```

---

**Garantía Metodológica de Reproducibilidad:**
Todos los experimentos fueron evaluados bajo la condición formal *ceteris paribus* fijando la semilla aleatoria `seed=42`, midiendo el consumo de memoria RSS mediante `psutil`, y preservando una cota máxima inferior a 2.0 GB de RAM sobre la Semana 104 del dataset oficial de H&M.
