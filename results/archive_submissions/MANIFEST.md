# Registro de Archivo Histórico de Submissions (V1 a V8)

**Proyecto:** Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)  
**Máster:** Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid  
**Autor:** Manuel Valdivia  

Este directorio almacena las entregas generadas durante las distintas fases experimentales del Trabajo de Fin de Máster. Se registran los checksums y hashes de integridad de los archivos generados.

---

## 1. Registro de Checksums y Hashes

| Versión Oficial | Archivo Físico Original | Archivo Archivado | Tamaño (Bytes) | Hash MD5 | Hash SHA-256 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Submission V1** | `submission.csv` | `submission_v1.csv` | 270.280.083 | `10C30F578F4BBA82EE307F419EF07B49` | `097BB4187F1D5980C56D9088AB6DF39324DB8BD541C691154CE998B6A9A85AD0` |
| **V1 Comprimido** | `submission.csv.gz` | `submission_v1.csv.gz` | 56.693.753 | `7FF8625041F5E42D8D11533A1BB46421` | `7D1F4476756F4B4618F1704068C452D8D3DF271581101D9916093348AAA84FAB` |
| **Submission V2** | `submission_v2.csv` | `submission_v2.csv` | 270.280.083 | `3BFEB8EE90C8657F792AD247306F6D0A` | `6128F79D1C773AA93BB2580AE90D0AFBCDBF8A723E91027F80F871C63F93625F` |
| **V2 Comprimido** | `submission_v2.csv.gz` | `submission_v2.csv.gz` | 61.768.752 | `5CA2F52ECF12107C73A146B03D42DBCD` | `1EBABB75A801D69FADB12CC23450D3933D6C9B3ACD5CA8C1EF566ADF5E8A818B` |
| **Submission V3** | `submission_v3.csv` | `submission_v3.csv` | 270.280.083 | `738B558E20972D74C59FE8968F75EB64` | `19D08257657612D71AF60C7E95F9365BEDA0BCE6855D38171BCB68D9EBC4C691` |
| **V3 Comprimido** | `submission_v3.csv.gz` | `submission_v3.csv.gz` | 59.881.110 | `04BABF631F73C3BBBB8AE49030CE1E36` | `7002F21B0358928487E0A342780E913852C8D423030BD130ADBE5A6B51F40894` |
| **Submission V4** | `submission_v4.csv` | `submission_v4.csv` | 270.280.083 | `C74D1FE6A9D74456C9779D2275D64896` | `9C26C0B1E54F2AB48F2496D8D0042321250F4965FA351F40CF04334DD7F16703` |
| **V4 Comprimido** | `submission_v4.csv.gz` | `submission_v4.csv.gz` | 62.479.099 | `E5ABA0A7A7530AD3C982E71663812765` | `FB0A15A17ACF068DCA2FB3CAD3100221BA3D42C9C04B2923079F2EBDAAAE2BFF` |
| **Submission V5** | `submission_v5.csv` | `submission_v5.csv` | 270.280.083 | `1EE36703A0B723DBEDA52F9CFF917584` | `6516C735AA30038520BCBED85756250D55AE4D388FDF1D21633573175740C2E2` |
| **V5 Comprimido** | `submission_v5.csv.gz` | `submission_v5.csv.gz` | 56.662.210 | `3B4B27C1E70366ECDBDFFA8B198B75EF` | `7CE096991256B21D75276192CFCFDFD566C07BA8B32E0A586C8C160F7F0F57C9` |
| **Submission V6** | `submission_v6.csv` | `submission_v6.csv` | 270.280.083 | `17407A3C1DD2E1A8674E8719D62D044A` | `D06CEE34741ED69BDF6EA8975CE4775217100E42DEA5D737411028A7DBBB5E9D` |
| **V6 Comprimido** | `submission_v6.csv.gz` | `submission_v6.csv.gz` | 62.687.462 | `B88E00064CBBC4321BB311A3122E51DB` | `D3EB5745BE623FB19AD1F209E4B5B157D31F7EFE3B0095D6F48963DB682CA0B7` |
| **Submission V7** | `submission_v7.csv` | `submission_v7.csv` | 270.280.083 | `4AC7EB802DC7C8EF6F09DAA978C0B4C2` | `F0DA19502561331D2962876A577FC054EFC962763D7D537958CDF9948D160DC8` |
| **V7 Comprimido** | `submission_v7.csv.gz` | `submission_v7.csv.gz` | 65.604.405 | `619BF58F3D1BAF48130326E0E7534B2C` | `AD32F0A067514FFE2D4A585704E680B55C9BE239DF029E507588437692CFC862` |
| **Submission V8** | `submission_v8.csv` | `submission_v8.csv` | 270.280.083 | `E842B6E8092DF684665C2DA5EF9B4214` | `E34B230E3C0744C335F6E809175C115C63BB73BB5B10A9A7A2A5DBA7C53DF106` |
| **V8 Comprimido** | `submission_v8.csv.gz` | `submission_v8.csv.gz` | 60.060.309 | `26A84DB15246DC6CE164B22714F1B4BC` | `C766D66BF7649D713AA55D0C03C35B9D95B04D34665DBF931768633E902E635C` |

---

## 2. Resumen Experimental y Diagnóstico por Versión

### Submission V1 (PoC Inicial: Muestra Reducida)
* **Score Oficial Kaggle:** Public: **0.00545** | Private: **0.00567**
* **Enfoque:** Ejecución con `--sample` sobre 2.000 clientes activos evaluados con LGBMRanker inicial y fallback estático de artículos de 2018 para los 1.369.980 clientes restantes (99,86% del universo).
* **Diagnóstico del Error:** Desfase estacional crítico (artículos de hace dos años) y cobertura poblacional insuficiente.

### Submission V2 (Escalado Masivo Out-of-Core a Catálogo Completo)
* **Score Oficial Kaggle:** Public: **0.01726** | Private: **0.01735**
* **Enfoque:** Escalado a 278.275 clientes activos con LGBMRanker (17,29M pares procesados) y fallback dinámico de los últimos 7 días con decaimiento exponencial ($\lambda = 0.05$) particionado en 5 cohortes etarias.
* **Resultado:** Salto cualitativo $> 3\times$ (+206% en Private), estableciendo el benchmark de partida a gran escala.

### Submission V3 (Desacople Causal Estricto Anti-Leakage)
* **Score Oficial Kaggle:** Public: **0.01597** | Private: **0.01619**
* **Enfoque:** Intento de aislamiento causal estricto exigiendo compras en semanas consecutivas para entrenar.
* **Diagnóstico del Error:** Pérdida del 96,9% de clientes de entrenamiento por contracción muestral; early stopping se detuvo prematuramente en la Ronda 2 (*modelo de 2 árboles*), perdiendo expresividad.

### Submission V4 (Re-ranking Supervisado Multi-Candidato con R8)
* **Score Oficial Kaggle:** Public: **0.01690** | Private: **0.01728**
* **Enfoque:** 100 candidatos por usuario con 8 heurísticas de recall (incorporando R8 de departamento favorito), 39 características y 50 árboles LightGBM.
* **Diagnóstico del Error:** *Fenómeno del Desplazamiento por Hiper-Generación*. El modelo supervisado saturó los 12 puestos con compras secundarias de agosto, expulsando del ranking los 5 superventas determinantes de otoño para el 84,8% de los usuarios activos.

### Submission V5 (Waterfall Estratificado con Recencia de 35 Días)
* **Score Oficial Kaggle:** Public: **0.02238** | Private: **0.02212** (+27,5% vs V2)
* **Enfoque:** Cascada determinista con retención de historial de 35 días (5 semanas) e inyección forzosa de superventas otoñales de la última semana por cohortes de edad.
* **Resultado:** Gran avance competitivo superando el umbral de 0.022 en Private Leaderboard.
* **Limitación Identificada:** *Saturación por sobre-personalización*: a usuarios activos con muchas compras se les asignaban hasta 12 recompras antiguas, bloqueando el espacio para novedades otoñales y artículos complementarios.

### Submission V6 (Waterfall Híbrido Causal con Co-ocurrencia P(B|A) y Truncamiento Rígido)
* **Score Oficial Kaggle:** Public: **0.02134** | Private: **0.02197** | Validación Local ($W_{104}$): **0.02659**
* **Enfoque:**
  1. **Slots 1 a 3 ($k_p \le 3$):** Recompras personales acotadas a la ventana activa (35 días), ordenadas por recencia y frecuencia.
  2. **Slots 4 a 7 ($k_c \le 4$):** Recuperación causal por co-ocurrencia en cesta transaccional $P(B|A)$, minada sobre 288.389 transacciones multi-artículo recientes.
  3. **Slots 8 a 12:** Superventas otoñales micro-segmentadas por cohorte de edad y ticket medio, con relleno defensivo global.
* **Diagnóstico del Descenso en Rendimiento:**
  * Al topar rígidamente las compras personales a $k_p \le 3$, se truncaron y descartaron las compras habituales (ítems 4º al 12º) de **112.999 clientes frecuentes (41.37% de los compradores activos)**, quienes aportan más del 70% de las transacciones evaluadas.
  * Reemplazar compras personales auténticas por predicciones asociativas probabilísticas provocó una pérdida neta en Leaderboard (Private 0.02197 vs 0.02212 de V5; Public 0.02134 vs 0.02238 de V5), fundamentando la necesidad de la arquitectura no destructiva V7.

### Submission V7 (Waterfall Híbrido Causal No Destructivo)
* **Score Oficial Kaggle:** Public: **0.02291** | Private: **0.02332** (+5.42% vs V5 en Private)
* **Score de Validación Local ($W_{104}$):** **0.02805** (+3.78% vs V5: 0.02703; +5.49% vs V6: 0.02659)
* **Recall@12 Local:** **0.06040** (+7.09% vs V5) | **Hit Rate@12 Local:** **0.11561** (+3.34% vs V5) | **Catalog Coverage:** **0.17590** (+3.78% vs V5)
* **Enfoque Arquitectónico:**
  1. **Slots Personales No Destructivos ($k_p \le 12$):** Preservación estricta del 100% de las compras personales en la ventana de 35 días, garantizando que ningún cliente frecuente sufra descarte forzado de su historial de recompra.
  2. **Slots de Rescate Causal Adaptativo ($k_c \le 6$ en huecos libres):** Minería de co-ocurrencia en cesta $P(B|A)$ con soporte $n \ge 2$, aplicada dinámicamente sobre los puestos vacantes ($12 - k_p$) para el 95.49% de clientes que tienen $<12$ compras (la mediana es 3 compras).
  3. **Slots de Fallback Bestseller Otoñal:** Bestsellers de la última semana por cohortes de edad para completar cualquier lista hasta los 12 puestos requeridos y para el 80.09% de clientes en cold-start.
* **Métricas de Ejecución:** Generada en 26.01 segundos (DuckDB out-of-core, 394.256 usuarios/segundo), consumo RAM de 1.963 MB (<2.000 MB) y 100% libre de nulos o anomalías.

### Submission V8 (Waterfall Híbrido Multidimensional con Afinidad Global Ponderada)
* **Score Oficial Kaggle:** Public: **0.02347** | Private: **0.02386** (Mejor resultado del proyecto, +2.32% vs V7 y +7.87% vs V5 en Private Leaderboard).
* **Score de Validación Local ($W_{104}$):** **0.02882** (+2.73% vs V7: 0.02805; +6.62% vs V5: 0.02703).
* **Recall@12 Local:** **0.06260** (+3.65% vs V7; +10.99% vs V5).
* **Hit Rate@12 Local:** **0.11945** (+3.32% vs V7; +6.78% vs V5).
* **Catalog Coverage:** **0.17518**.
* **Enfoque Arquitectónico:**
  1. **Ventana Personal Óptima de 28 Días ($k_p \le 12$):** Depuración del ruido de fin de agosto (días 29 a 35), concentrando la personalización en artículos con alta vigencia de temporada.
  2. **Puntuación de Afinidad Global en Cesta ($k_c \le 6$):** Matriz temporal causal con soporte $\ge 3$ ponderada por recencia del par, integrando la afinidad agregada $S(u, c) = \sum 0.80^{d/7} \cdot W_{\text{pair}}$ a través de los últimos 5 artículos del cliente.
  3. **Suavizado Bayesiano Multi-Semana de Bestsellers ($\gamma = 0.12$):** Agregación de ventas de 3 semanas con decaimiento temporal geométrico por cohorte de edad, amortiguando quiebres de stock sin comprometer la tendencia contemporánea.
* **Métricas de Ejecución:** Generada en 28.78 segundos en cómputo directo en memoria RAM (DuckDB out-of-core, 276.067 usuarios/segundo, consumo RAM de 1.357 MB < 2.000 MB) y 47.21 segundos en pipeline completo de exportación a disco con cálculo simultáneo de firma criptográfica SHA-256 (254.731 usuarios/segundo, consumo RAM pico de 1.435,8 MB).
* **Nota Metodológica de Comparación ($\Delta\%$):** Las mejoras reportadas en Kaggle toman como base la evaluación externa ciega de Private Test (+2.32% vs V7 y +7.87% vs V5; +162.2% vs V0), mientras que en validación local toman como base la partición retenida $W_{104}$ (+2.73% vs V7 y +6.62% vs V5; +157.3% vs V0). Ambas son complementarias y ratifican la estabilidad del ranking.

---

## 3. Versión Final del Proyecto

La versión final evaluada es:
* **`submission_v8.csv`** (Ubicada en la raíz del repositorio)
* **SHA-256:** `E34B230E3C0744C335F6E809175C115C63BB73BB5B10A9A7A2A5DBA7C53DF106`
* **MD5:** `E842B6E8092DF684665C2DA5EF9B4214`
* **GZIP SHA-256:** `C766D66BF7649D713AA55D0C03C35B9D95B04D34665DBF931768633E902E635C`
* **GZIP MD5:** `26A84DB15246DC6CE164B22714F1B4BC`

---

## 4. Consideraciones de Reproducibilidad e Integridad Multiplataforma

1. **Sensibilidad de los Hashes SHA-256:** Un hash SHA-256 evalúa la integridad binaria exacta. En un archivo de 270 MB con 1.371.980 registros, la mínima variación de un solo byte altera el hash por completo, aunque el tamaño físico (270.280.083 bytes) y las recomendaciones coincidan.
2. **Desempates en DuckDB (Multi-Threading):** Las consultas de minería de co-ocurrencia $P(B|A)$ y agregación temporal se ejecutan en paralelo sobre múltiples hilos de CPU. Cuando dos artículos complementarios comparten exactamente la misma puntuación de afinidad, el orden en que los buffers de memoria se consolidan puede provocar inversiones en los puestos 11º y 12º para una fracción residual de usuarios.
3. **Diferencias de Entorno Operativo:** La compilación del motor DuckDB y Polars en Linux (Docker) frente a Windows (Python 3.14) presenta ligeras variaciones de redondeo en operaciones de coma flotante de decaimiento temporal.
4. **Verificación Local:** Para auditar y registrar la submission generada en un entorno específico, ejecute:
   ```bash
   python scripts/07_submission.py --register --version v8
   python scripts/07_submission.py --check-file submission_v8.csv --version v8
   ```


