"""Generación de diagramas arquitectónicos del pipeline en imágenes PNG.

Máster en Data Science, Big Data & Business Analytics — Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Genera 8 diagramas arquitectónicos de alta resolución en results/figures/:
- diag_01_data_ingestion_out_of_core.png (Ingesta DuckDB + Polars downcasting + Parquet ZSTD)
- diag_02_candidate_retrieval_pool.png (Embudo multi-heurística R1..R8 y capping a 100)
- diag_03_feature_engineering_dag.png (Grafo de 39 características en 4 namespaces y anti-leakage)
- diag_04_lgbm_ranker_training.png (Entrenamiento supervisado LambdaRank y early stopping)
- diag_05_v8_waterfall_architecture.png (Arquitectura SOTA V8 Multidimensional en 4 niveles)
- diag_06_fastapi_serving_runtime.png (Serving en memoria, C++ predict y telemetría Prometheus)
- diag_07_execution_modes_isolation.png (Matriz de aislamiento de datos: Modos 1, 2 y 3)
- diag_08_business_conversion_lifecycle.png (Ciclo de conversión de negocio, GMV, LTV y Eficiencia de Infraestructura)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

C_PRIMARY = "#003f5c"
C_SECONDARY = "#2f4b7c"
C_ACCENT = "#ff6361"
C_HIGHLIGHT = "#ffa600"
C_MUTED = "#665191"
C_DARK = "#1e272e"
C_LIGHT = "#f8f9fa"
C_BORDER = "#dcdde1"
C_GREEN = "#2ed573"
C_TEAL = "#00a8ff"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"


def add_card(ax, x, y, w, h, title, subtitle="", bg=C_LIGHT, border=C_SECONDARY, lw=1.5, title_color=C_PRIMARY, sub_color=C_DARK, fontsize=10, sub_size=8, radius=0.02):
    """Dibuja una tarjeta con bordes redondeados y texto formateado."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius},rounding_size={radius*2}",
        facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
    )
    ax.add_patch(box)
    
    if title:
        ax.text(
            x + w / 2.0, y + h - 0.04 if subtitle else y + h / 2.0,
            title, color=title_color, fontsize=fontsize, fontweight="bold",
            ha="center", va="center" if not subtitle else "top", zorder=3
        )
    if subtitle:
        ax.text(
            x + w / 2.0, y + 0.04,
            subtitle, color=sub_color, fontsize=sub_size,
            ha="center", va="bottom", zorder=3
        )
    return box


def add_arrow(ax, x1, y1, x2, y2, color=C_SECONDARY, lw=2.0, label=""):
    """Dibuja una flecha de conexión entre dos componentes."""
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15,
        color=color, linewidth=lw, zorder=4
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        ax.text(mid_x, mid_y + 0.02, label, color=color, fontsize=8, fontweight="bold",
                ha="center", va="bottom", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.5))


# Diagrama: INGESTA Y DOWNCASTING OUT-OF-CORE
def generate_diag_01():
    print("-> Generando diag_01_data_ingestion_out_of_core.png...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Arquitectura de Ingesta Fuera de Memoria (Out-of-Core) y Downcasting Defensivo",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.91, "Procesamiento de 31.78M de transacciones bajo cota estricta de 12 GB RAM | DuckDB + Polars + Parquet ZSTD",
            fontsize=9.5, color=C_MUTED, ha="center")

    add_card(ax, 0.03, 0.58, 0.26, 0.28, "1. Datos Crudos en Disco", 
             "transactions_train.csv (3.49 GB)\n• 31.788.324 filas | 5 columnas\n• customer_id: String Hex (64 Bytes)\n• article_id: String (10 Bytes)\n• price: Float64 (8 Bytes)\n• sales_channel_id: Int64 (8 Bytes)\n\n[Inviable en RAM: Pandas > 14 GB]",
             bg="#f1f2f6", border=C_ACCENT, lw=2, fontsize=11, sub_size=8.5)

    add_card(ax, 0.03, 0.18, 0.26, 0.34, "Catálogos Adicionales",
             "customers.csv (1.37M usuarios)\n• age: Float64 -> mediana 32\n• club_member_status, fashion_news\n\narticles.csv (105.5K productos)\n• taxonomía, color, departamento\n\nsample_submission.csv (1.37M filas)",
             bg="#f1f2f6", border=C_MUTED, lw=1.5, fontsize=10.5, sub_size=8.5)

    add_card(ax, 0.36, 0.45, 0.28, 0.41, "2. DuckDB Streaming Engine",
             "Consultas SQL Vectorizadas directas a disco\n\n• Filtrado temporal de 5 semanas:\n  WHERE t_dat >= '2020-08-19'\n  (Contracción: 31.7M -> 1.30M filas, -95.9%)\n\n• Proyección estricta de columnas\n• Cero sobrecarga de punteros de objetos\n• Punteros compartidos Apache Arrow hacia Polars\n\n[Consumo residente en disco: < 400 MB RSS]",
             bg="#eef2f7", border=C_SECONDARY, lw=2, fontsize=11, sub_size=8.5)

    add_card(ax, 0.70, 0.50, 0.27, 0.36, "3. Downcasting Polars Arrow",
             "Mapeo de Tipos en Memoria Contigua:\n\n• customer_id (64B) -> customer_idx (Int32 4B)\n  [Ahorro de memoria: -93.75%]\n• article_id (10B) -> article_id (Int32 4B)\n  [Ahorro de memoria: -60.00%]\n• price (Float64 8B) -> price (Float32 4B)\n  [Ahorro de memoria: -50.00%]\n• sales_channel_id (8B) -> Int8 (1B) [-87.5%]",
             bg="#f0f9f4", border=C_GREEN, lw=2, title_color=C_PRIMARY, fontsize=11, sub_size=8.5)

    add_card(ax, 0.36, 0.12, 0.61, 0.28, "4. Checkpoints Columnar Parquet ZSTD (data_processed/)",
             "• transactions_5w.parquet (1.30M filas, 24.3 MB, ZSTD Nivel 3) | transactions_10w.parquet (48.1 MB)\n• customers.parquet (1.37M filas, 9.4 MB) con segmentación age_bin (<25, 25-34, 35-44, 45-54, 55+)\n• articles.parquet (105.5K filas, 3.2 MB) con categorías compactadas\n• customer_id_mapping.parquet (Mapeo bidireccional hash <-> Int32)\n\n[Balance Final: Reducción de almacenamiento > 75% | Consumo total en modelado: < 1.8 GB RAM]",
             bg="#fef9e7", border=C_HIGHLIGHT, lw=2, title_color=C_PRIMARY, fontsize=11, sub_size=8.5)

    add_arrow(ax, 0.29, 0.72, 0.36, 0.68, color=C_SECONDARY, lw=2.5, label="SQL Chunking")
    add_arrow(ax, 0.29, 0.35, 0.36, 0.50, color=C_MUTED, lw=1.8)
    add_arrow(ax, 0.64, 0.68, 0.70, 0.68, color=C_GREEN, lw=2.5, label="Zero-Copy Arrow")
    add_arrow(ax, 0.83, 0.50, 0.83, 0.40, color=C_HIGHLIGHT, lw=2.0)
    add_arrow(ax, 0.83, 0.40, 0.68, 0.40, color=C_HIGHLIGHT, lw=2.0, label="Escritura ZSTD")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_01_data_ingestion_out_of_core.png", dpi=300)
    plt.close(fig)


# Diagrama: EMBUDO DE CANDIDATOS (R1 A R8)
def generate_diag_02():
    print("-> Generando diag_02_candidate_retrieval_pool.png...")
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Embudo Multi-Fuente de Candidatos: 8 Heurísticas de Recall (R1 a R8)",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.92, "Reducción de 105.542 prendas a un pool de 100 candidatos por usuario activo | Techo de Recall@80: 8.44% (Hit Rate: 16.52%)",
            fontsize=9.5, color=C_MUTED, ha="center")

    heuristics = [
        ("R1: Recompra Reciente", "Top 24 | Compras en 5 semanas\nOrdenado: max(t_dat), count(*)", C_PRIMARY, 0.03, 0.68),
        ("R2: Popularidad Global", "Top 20 | Decaimiento temporal\nw(t) = exp(-0.1 * delta_w)", C_SECONDARY, 0.03, 0.48),
        ("R3: Popularidad por Edad", "Top 15 | 5 Cohortes generacionales\n<25, 25-34, 35-44, 45-54, 55+", C_MUTED, 0.03, 0.28),
        ("R4: Canal Preferente", "Top 15 | Sesgo físico vs online\nsales_channel_id dominante", C_TEAL, 0.03, 0.08),
        ("R5: Item-CF (Cesta 14d)", "Top 20 | Co-ocurrencias cruzadas\nP(B|A) atuendos / outfits", C_ACCENT, 0.27, 0.68),
        ("R6: Familias de Producto", "Top 10 | Artículos líderes en los\n2 departamentos preferidos", C_HIGHLIGHT, 0.27, 0.48),
        ("R7: Aceleración / Tendencias", "Top 10 | Mayor aceleración de ventas\n(V_t - V_t-1) / (V_t-1 + 1)", "#e056fd", 0.27, 0.28),
        ("R8: Departamento Favorito", "Top 12 | Novedades de última semana\nen el departamento prioritario d_u*", "#eb4d4b", 0.27, 0.08),
    ]

    for name, desc, col, x, y in heuristics:
        add_card(ax, x, y, 0.21, 0.17, name, desc, bg="white", border=col, lw=1.8, title_color=col, fontsize=9.5, sub_size=7.5)
        add_arrow(ax, x + 0.21, y + 0.085, 0.54, 0.52, color=col, lw=1.2)

    add_card(ax, 0.54, 0.32, 0.22, 0.42, "Consolidación & Meta-Features",
             "• Unión Vectorizada Polars\n• Deduplicación de pares (u, a)\n• Banderas de origen is_R1 .. is_R8\n• Conteo de consenso n_sources\n• Rango mínimo best_rank\n\nOrdenación jerárquica:\n1. n_sources DESC (consenso)\n2. best_rank ASC (fuerza)\n\nCapping estricto a 100 candidatos",
             bg="#f1f2f6", border=C_PRIMARY, lw=2.5, fontsize=10.5, sub_size=8)

    add_card(ax, 0.81, 0.35, 0.16, 0.36, "candidates.parquet",
             "Pool Consolidado Oficial:\n\n• 278.275 clientes activos\n• 18.828.129 pares (u, a)\n• Cobertura: 100% activos\n• 0 nulos | 0 duplicados\n• Tiempo: 10.13 segundos\n• RAM pico: 1.241 MB RSS\n• Compresión: ZSTD",
             bg="#f0f9f4", border=C_GREEN, lw=2.5, title_color=C_PRIMARY, fontsize=11, sub_size=8.5)

    add_arrow(ax, 0.76, 0.53, 0.81, 0.53, color=C_GREEN, lw=3.0, label="Cap 100")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_02_candidate_retrieval_pool.png", dpi=300)
    plt.close(fig)


# Diagrama: GRAFO DE INGENIERÍA DE CARACTERÍSTICAS (39 FEATURES)
def generate_diag_03():
    print("-> Generando diag_03_feature_engineering_dag.png...")
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Grafo de Ingeniería de Características: 39 Variables en 4 Namespaces",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.92, "Esquema temporal causal con corte en semana 104 (Anti-Leakage) | Serialización en streaming Parquet ZSTD",
            fontsize=9.5, color=C_MUTED, ha="center")

    namespaces = [
        ("NS1: Usuario (9 Variables)", 
         "• u_total_transactions (Int32)\n• u_unique_articles (Int32)\n• u_mean_price (Float32)\n• u_std_price (Float32)\n• u_online_ratio (Float32)\n• u_age (Int16, mediana 32)\n• u_club_status (Int8: 1=ACT, 2=PRE)\n• u_fashion_news (Int8: 0=NONE, 2=REG)\n• u_last_purchase_days_ago (Int16)",
         C_PRIMARY, 0.03, 0.52, 0.22, 0.36),
        
        ("NS2: Artículo (9 Variables)", 
         "• a_total_sales (Int32 volumen)\n• a_unique_customers (Int32)\n• a_mean_price (Float32)\n• a_online_ratio (Float32 canal)\n• a_product_type_no (Int16 tipo)\n• a_department_no (Int16 depto)\n• a_index_group_no (Int8 sección)\n• a_sales_decayed (Float32 decaim.)\n• a_recency_days (Int16 días activo)",
         C_SECONDARY, 0.03, 0.10, 0.22, 0.36),

        ("NS3: Interacción U x A (8 Vars)", 
         "• ua_purchase_count (Int16 compras)\n• ua_days_since_last_purchase (Int16)\n• ua_price_diff_user_mean (Float32)\n• ua_same_dept_purchase_ratio (Float32)\n• ua_same_index_purchase_ratio (Float32)\n• ua_channel_match (Int8 sesgo)\n• ua_repurchase_flag (Int8 0/1)\n• ua_age_fit_score (Float32 afín edad)",
         C_ACCENT, 0.29, 0.52, 0.23, 0.36),

        ("NS4: Meta-Features (13 Vars)", 
         "• is_R1 .. is_R8 (8 Banderas Int8)\n• source_count (Int8 total fuentes)\n• r1_rank (Int16 rango recompra)\n• r2_rank (Int16 rango global)\n• r3_rank (Int16 rango edad)\n• r5_rank (Int16 rango co-ocurr.)",
         C_MUTED, 0.29, 0.10, 0.23, 0.36),
    ]

    for title, desc, col, x, y, w, h in namespaces:
        add_card(ax, x, y, w, h, title, desc, bg="white", border=col, lw=2, title_color=col, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.56, 0.40, 0.20, 0.48, "Corte Temporal Causal",
             "Particionado Anti-Leakage:\n\n• Historial observable: Semanas 100-103\n  (2020-08-19 a 2020-09-15)\n  Cálculo de las 39 variables\n\n• Semana 104 Retenida:\n  (2020-09-16 a 2020-09-22)\n  Ground Truth: target = 1 si compra,\n  target = 0 si candidato no comprado\n\n[Inmunidad temporal 100% certificada]",
             bg="#fef9e7", border=C_HIGHLIGHT, lw=2.2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8)

    add_card(ax, 0.80, 0.40, 0.17, 0.48, "features_matrix.parquet",
             "Matriz Tabular Final:\n\n• 41 columnas (2 keys + 39 feats)\n• 5.572.394 pares en train (1:5)\n• 18.828.129 pares en inferencia\n• Tamaño ZSTD: 342.92 MB\n• Tiempo: 28.42 segundos\n• RAM: 1.923 MB RSS\n• Tipado estricto:\n  0 Float64 | 0 Int64",
             bg="#f0f9f4", border=C_GREEN, lw=2.5, title_color=C_PRIMARY, fontsize=11, sub_size=8.5)

    add_arrow(ax, 0.25, 0.70, 0.29, 0.70, color=C_PRIMARY, lw=1.5)
    add_arrow(ax, 0.25, 0.28, 0.29, 0.28, color=C_SECONDARY, lw=1.5)
    add_arrow(ax, 0.52, 0.70, 0.56, 0.65, color=C_ACCENT, lw=2.0, label="Join keys (u, a)")
    add_arrow(ax, 0.52, 0.28, 0.56, 0.45, color=C_MUTED, lw=2.0)
    add_arrow(ax, 0.76, 0.64, 0.80, 0.64, color=C_GREEN, lw=2.8, label="ParquetWriter")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_03_feature_engineering_dag.png", dpi=300)
    plt.close(fig)


# Diagrama: ENTRENAMIENTO LGBMRANKER Y RANKING SUPERVISADO
def generate_diag_04():
    print("-> Generando diag_04_lgbm_ranker_training.png...")
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Pipeline de Modelado y Re-Ranking Supervisado: LGBMRanker con LambdaRank",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.91, "Optimización de permutaciones listwise | Negative Downsampling 1:5 | Validación oficial MAP@12 en W104",
            fontsize=9.5, color=C_MUTED, ha="center")

    add_card(ax, 0.03, 0.35, 0.25, 0.50, "1. Preparación de Consultas",
             "Estructuración de Grupos (query_groups):\n\n• 240.912 consultas activas en train\n• Agrupamiento estricto por customer_idx\n\nMuestreo Negativo Estratificado 1:5:\n• Por cada positivo (y=1) se muestrean\n  exactamente 5 negativos difíciles (y=0)\n• Reduce el desbalance extremo de 1:80 a 1:5\n• Contracción a 5.572.394 filas de train\n• Aceleración computacional > 10x\n• Preserva hard negatives de alta ganancia",
             bg="#f1f2f6", border=C_PRIMARY, lw=2, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.34, 0.35, 0.31, 0.50, "2. Optimizador LambdaRank",
             "Hiperparámetros de Producción (Taichi Iki 6th):\n• objective = 'lambdarank' | metric = 'map@12'\n• boosting_type = 'gbdt' (Leaf-wise split)\n• n_estimators = 50 | learning_rate = 0.05\n• num_leaves = 63 | max_depth = 7\n• min_child_samples = 30 (Regularizador)\n• colsample_bytree = 0.8 | subsample = 0.8\n\nGradiente de Ranking Pairwise:\nlambda_ij = -sigma / (1 + exp(sigma*(s_i - s_j))) * |Delta NDCG_ij|\n\n37 de las 39 variables activas en divisiones de árbol",
             bg="#eef2f7", border=C_SECONDARY, lw=2.2, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.71, 0.50, 0.26, 0.35, "3. Validación MAP@12",
             "Cohorte de Prueba W104 (68.984 clientes):\n\n• MAP@12 Local: 0.02341 (V4)\n• Recall@12: 94.09% (en pares puntuados)\n• Hit Rate@12: 96.10%\n• Early Stopping con paciencia de 20 rondas\n• Gráfico Feature Importance exportado",
             bg="#fef9e7", border=C_HIGHLIGHT, lw=2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.71, 0.12, 0.26, 0.32, "Modelos Serializados",
             "• models/lgbm_ranker.txt (203 KB)\n  Booster en C++ para inferencia rápida\n• models/lgbm_ranker_meta.json\n  Esquema de variables y normalizaciones\n• SHA-256 verificado en manifiesto",
             bg="#f0f9f4", border=C_GREEN, lw=2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8.2)

    add_arrow(ax, 0.28, 0.60, 0.34, 0.60, color=C_PRIMARY, lw=2.5, label="X_train, y, groups")
    add_arrow(ax, 0.65, 0.67, 0.71, 0.67, color=C_SECONDARY, lw=2.5, label="Early Stopping")
    add_arrow(ax, 0.84, 0.50, 0.84, 0.44, color=C_GREEN, lw=2.2, label="Save Model")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_04_lgbm_ranker_training.png", dpi=300)
    plt.close(fig)


# Diagrama: ARQUITECTURA V8 WATERFALL CASCADE
def generate_diag_05():
    print("-> Generando diag_05_v8_waterfall_architecture.png...")
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Arquitectura SOTA V8: Waterfall Híbrido Multidimensional",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.92, "Kaggle Private: 0.02386 (+7.87% vs V5) | Kaggle Public: 0.02347 | Local MAP@12: 0.02882 (+10.99% Recall@12)",
            fontsize=9.5, color=C_ACCENT, fontweight="bold", ha="center")

    add_card(ax, 0.03, 0.77, 0.20, 0.12, "Cliente Solicitante",
             "customer_id (Hex 64)\n-> Mapeo a customer_idx", bg="#f1f2f6", border=C_PRIMARY, lw=1.8, fontsize=10, sub_size=8)

    # Nivel 1: Recompra Personal (28 días)
    add_card(ax, 0.28, 0.70, 0.29, 0.19, "Nivel 1: Recompra 28d (kp <= 12)",
             "• Historial reciente a 28 días exactos (4 semanas)\n• Purga el ruido estival de fin de agosto\n• Ordenación: max(t_dat) DESC, count(*) DESC\n• 233.174 clientes activos reciben recompras\n• Preservación no destructiva de clientes VIP",
             bg="#f0f9f4", border=C_GREEN, lw=2.2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8)

    # Nivel 2: Afinidad Global en Cesta P(B|A)
    add_card(ax, 0.62, 0.70, 0.35, 0.19, "Nivel 2: Cesta P(B|A) (kc <= 6 en huecos)",
             "• Minería causal de co-ocurrencia multi-semana (N >= 3)\n• Afinidad multivariada: S(u, c) = sum(0.80^(d/7) * W_pair)\n• 217.240 clientes (93.17% de activos) reciben complementos\n• Rellena slots vacantes (12 - kp) sin canibalizar recompra",
             bg="#eef2f7", border=C_SECONDARY, lw=2.2, fontsize=10.5, sub_size=8)

    # Nivel 3: Bestsellers Multi-Semana por Edad
    add_card(ax, 0.28, 0.38, 0.33, 0.22, "Nivel 3: Bestsellers Edad (gamma=0.12)",
             "• Regularización Bayesiana de 3 Semanas:\n  Score = Ventas_w0 + 0.12*Ventas_w1 + 0.014*Ventas_w2\n• Amortigua roturas de stock en almacén\n• Microsegmentación en 5 cohortes: <25, 25-34, 35-44, 45-54, 55+\n• Inyección de rescate en clientes con < 12 prendas",
             bg="#fef9e7", border=C_HIGHLIGHT, lw=2.2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8)

    # Nivel 4: Superventas Globales Estacionales
    add_card(ax, 0.66, 0.38, 0.31, 0.22, "Nivel 4: Fallback Estacional Global",
             "• 1.138.806 clientes cold-start (83.00% de la base)\n• Top-12 superventas agregados con decaimiento\n• Cero escaparates vacíos en la tienda\n• Latencia de respuesta en servidor: < 0.01 ms",
             bg="#fde2e2", border=C_ACCENT, lw=2.2, title_color=C_ACCENT, fontsize=10.5, sub_size=8)

    # Salida Final
    add_card(ax, 0.35, 0.06, 0.50, 0.20, "Top-12 Recomendaciones Oficiales (Kaggle Submission Format)",
             "• Exactamente 12 códigos article_id de 10 dígitos con ceros a la izquierda (str.zfill(10))\n• Longitud exacta por fila: 131 caracteres | 1.371.980 clientes certificados\n• Inferencia masiva en 47.21 segundos (254.731 clientes/s) con 1.435 MB RSS\n• SHA-256 verificado: C766D66BF7649D713AA55D0C03C35B9D95B04D34665DBF931768633E902E635C",
             bg="#f8f9fa", border=C_PRIMARY, lw=2.5, title_color=C_PRIMARY, fontsize=11, sub_size=8.5)

    add_arrow(ax, 0.23, 0.83, 0.28, 0.83, color=C_PRIMARY, lw=2.0, label="Historial 28d")
    add_arrow(ax, 0.57, 0.80, 0.62, 0.80, color=C_GREEN, lw=2.0, label="kp < 12")
    add_arrow(ax, 0.79, 0.70, 0.50, 0.60, color=C_SECONDARY, lw=1.5)
    add_arrow(ax, 0.50, 0.60, 0.50, 0.38, color=C_HIGHLIGHT, lw=2.0, label="Slots vacíos")
    add_arrow(ax, 0.13, 0.77, 0.13, 0.49, color=C_ACCENT, lw=1.8)
    add_arrow(ax, 0.13, 0.49, 0.28, 0.49, color=C_ACCENT, lw=1.8, label="Cold-Start")
    add_arrow(ax, 0.60, 0.49, 0.66, 0.49, color=C_ACCENT, lw=1.8, label="Sin edad")
    add_arrow(ax, 0.60, 0.38, 0.60, 0.26, color=C_PRIMARY, lw=2.5, label="Normalizar Top-12")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_05_v8_waterfall_architecture.png", dpi=300)
    plt.close(fig)


# Diagrama: RUNTIME DE LA API REST (FASTAPI + DOCKER)
def generate_diag_06():
    print("-> Generando diag_06_fastapi_serving_runtime.png...")
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Arquitectura de Servicio y Runtime de Inferencia en Tiempo Real (FastAPI + Docker)",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.91, "Throughput: 955.8 req/s | p95 personalizado < 50 ms (media 9.98 ms) | Cold-start < 0.01 ms | RSS < 460 MB",
            fontsize=9.5, color=C_MUTED, ha="center")

    # Paso 1: Petición HTTP
    add_card(ax, 0.03, 0.52, 0.21, 0.33, "1. Petición Entrante",
             "POST /recommend/{customer_id}\n\nParámetros:\n• customer_id (Path, hex 64)\n• limit: int = 12 (Query)\n• offset: int = 0 (Query)\n\nSeguridad:\n• Cabecera X-API-Key opcional\n• HTTP 401 si clave inválida",
             bg="#f1f2f6", border=C_PRIMARY, lw=1.8, fontsize=10.5, sub_size=8)

    # Paso 2: Router y Asincronía
    add_card(ax, 0.28, 0.52, 0.22, 0.33, "2. FastAPI ASGI Router",
             "Ciclo de Vida Lifespan:\n• Precarga única en arranque\n• Cero retraso en primera petición\n\nDesacoplamiento No Bloqueante:\n• asyncio.to_thread()\n• Libera el bucle de eventos\n• Inferencia de CPU delegada a\n  hilos de trabajo nativos",
             bg="#eef2f7", border=C_SECONDARY, lw=2, fontsize=10.5, sub_size=8)

    # Paso 3: Motor en Memoria Singleton
    add_card(ax, 0.54, 0.40, 0.24, 0.45, "3. RecommenderServiceLoader",
             "Patrón Singleton Thread-Safe:\n\n• Arrays contiguos C-Order en NumPy:\n  float32 contiguos para acceso O(1)\n• Gobernanza API_MAX_INDEXED_ROWS:\n  100k filas (~380 MB) / 500k (~700 MB)\n• Guardarraíl SRE anti-OOM\n\nResolución Multinivel:\n1. Historial 28d V8 (Recompra + Cesta)\n2. Inferencia C++ Booster LightGBM\n3. Fallback Bestsellers por Cohorte\n4. Popularidad Global (< 5 ms)",
             bg="#f0f9f4", border=C_GREEN, lw=2.2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8)

    # Paso 4: Telemetría y Salida
    add_card(ax, 0.81, 0.52, 0.16, 0.33, "4. Respuesta JSON",
             "HTTP 200 OK:\n\n• customer_id\n• recommendations (10 dig)\n• count: 12\n• is_cold_start: bool\n• latency_ms: float\n• model_version: '1.0.0'\n\nContratos Pydantic v2",
             bg="#fef9e7", border=C_HIGHLIGHT, lw=2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8)

    add_card(ax, 0.03, 0.12, 0.94, 0.20, "Gobernanza Operativa y Métricas Prometheus (/metrics)",
             "• GET /health: Telemetría en vivo (indexed_customers, catalog_customers, memory_rss_mb) | GET /health/live: Sonda liveness (< 1 ms)\n• GET /health/ready: Sonda readiness (modelo y catálogo listos) | GET /metrics: Exportador Prometheus (latencias p50/p95, total peticiones, cold starts)\n• Empaquetado Docker: Linux Debian 12 / Python 3.12-slim bajo usuario non-root appuser (UID 1001) | Imagen de 295 MB con contexto de 0.21 MB",
             bg="#f8f9fa", border=C_MUTED, lw=1.5, title_color=C_PRIMARY, fontsize=10, sub_size=8)

    add_arrow(ax, 0.24, 0.68, 0.28, 0.68, color=C_PRIMARY, lw=2.0)
    add_arrow(ax, 0.50, 0.68, 0.54, 0.68, color=C_SECONDARY, lw=2.0, label="asyncio.to_thread")
    add_arrow(ax, 0.78, 0.68, 0.81, 0.68, color=C_GREEN, lw=2.0, label="Pydantic v2")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_06_fastapi_serving_runtime.png", dpi=300)
    plt.close(fig)


# Diagrama: MATRIZ DE AISLAMIENTO POR MODOS DE EJECUCIÓN
def generate_diag_07():
    print("-> Generando diag_07_execution_modes_isolation.png...")
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Matriz de Aislamiento y Flujos de Datos entre los Tres Modos de Ejecución",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.92, "Desacoplamiento estricto de namespaces: Cero riesgo de sobreescritura entre desarrollo ágil y producción masiva",
            fontsize=9.5, color=C_MUTED, ha="center")

    modes = [
        ("MODO 1: Verificación Rápida (--sample)",
         "• Objetivo: CI/CD, tests unitarios y validación de lógica end-to-end.\n• Entrada: data_sample/ (~3 MB versionados en Git, 2.000 clientes).\n• Checkpoints: data_processed/sample/*.parquet (Aislado).\n• Modelo: models/sample/lgbm_ranker.txt (LGBM_SAMPLE_PARAMS).\n• Tiempo: < 1 minuto | Consumo RAM: ~280 MB RSS.\n• Salida: submission de muestra (2.000 filas, no apta para Kaggle).",
         C_TEAL, 0.03, 0.48, 0.29, 0.38),

        ("MODO 2: Evaluación Productiva (Docker / Local)",
         "• Objetivo: Defensa técnica, auditoría y servicio web de baja latencia.\n• Entrada: production_artifacts.zip descomprimido (382.9 MB release).\n• Checkpoints: data_processed/*.parquet (Oficiales inmutables).\n• Modelo: models/lgbm_ranker.txt (Oficial de producción).\n• Tiempo: Inmediato (5-10 s arranque) | Consumo RAM: ~380 MB RSS.\n• Salida: API REST FastAPI en http://localhost:8000 (955 req/s).",
         C_GREEN, 0.355, 0.48, 0.29, 0.38),

        ("MODO 3: Replicación Científica Completa",
         "• Objetivo: Reconstrucción íntegra del pipeline desde datos crudos.\n• Entrada: data/*.csv oficiales descargados de Kaggle (3.49 GB, 31.7M tx).\n• Checkpoints: data_processed/*.parquet (Genera artefactos oficiales).\n• Modelo: Re-entrena models/lgbm_ranker.txt con 240k consultas.\n• Tiempo: ~30-45 minutos | Consumo RAM: <= 12 GB (Ryzen 5 5500).\n• Salida: submission_v8.csv.gz (1.371.980 clientes, SHA-256 verificado).",
         C_ACCENT, 0.68, 0.48, 0.29, 0.38),
    ]

    for title, desc, col, x, y, w, h in modes:
        add_card(ax, x, y, w, h, title, desc, bg="white", border=col, lw=2.2, title_color=col, fontsize=10.5, sub_size=8.2)

    # Bloque de Seguridad y Aislamiento de Namespaces
    add_card(ax, 0.03, 0.12, 0.94, 0.28, "Arquitectura de Protección de Namespaces (Zero-Collision Guarantee)",
             "• Todas las ejecuciones con el flag --sample leen y escriben exclusivamente en los subdirectorios data_processed/sample/ y models/sample/.\n• Los artefactos oficiales de producción en data_processed/ (429 MB) y models/ (203 KB) permanecen inmutables y 100% protegidos contra sobreescritura accidental.\n• El Modo 2 monta los directorios en modo de solo lectura (:ro) dentro de Docker Compose, garantizando que el microservicio opere sin efectos secundarios.\n• El Modo 3 es el generador legítimo de los checkpoints oficiales, requiriendo ejecución explícita sin el flag --sample para sobrescribir los pesos finales.",
             bg="#fef9e7", border=C_HIGHLIGHT, lw=2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_07_execution_modes_isolation.png", dpi=300)
    plt.close(fig)


# Diagrama: CICLO DE CONVERSIÓN Y VALOR DE NEGOCIO
def generate_diag_08():
    print("-> Generando diag_08_business_conversion_lifecycle.png...")
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Ciclo de Conversión y Propuesta de Valor de Negocio en Retail de Moda",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.92, "Alineación cuantitativa del sistema de recomendación con Facturación (GMV), Ticket Medio (AOV), Fidelización (LTV) y Eficiencia Operativa",
            fontsize=9.5, color=C_MUTED, ha="center")

    stages = [
        ("1. Adquisición (Cold-Start)",
         "Población: 83.00% de clientes (1.138.806)\nSesiones anónimas o sin compras recientes.\n\nEstrategia V8:\n• Superventas multi-semana (gamma=0.12)\n• Segmentación en 5 cohortes de edad\n• Respuesta instantánea < 0.01 ms en servidor\n\nImpacto de Negocio:\n• Cero escaparates vacíos en la tienda\n• Reducción radical de la tasa de rebote\n• +10.9% MAP@12 vs recomendador genérico",
         C_PRIMARY, 0.03, 0.40, 0.29, 0.46),

        ("2. Túnel de Compra (Cross-Selling)",
         "Población: 93.17% de activos (217.240)\nClientes con al menos 1 compra en 28 días.\n\nEstrategia V8 Cesta P(B|A):\n• Minería causal de co-ocurrencias (N >= 3)\n• Afinidad multivariada S(u, c) ponderada\n• Módulos 'Completa tu Look' (outfits)\n\nImpacto de Negocio:\n• Aumento de Units Per Transaction (UPT)\n• Elevación directa del ticket medio (AOV)\n• Venta cruzada de prendas de alto margen",
         C_SECONDARY, 0.355, 0.40, 0.29, 0.46),

        ("3. Retención y Repetición (VIP)",
         "Población: 41.37% de compradores (113.000)\nClientes frecuentes que generan > 70% ventas.\n\nEstrategia V8 No Destructiva:\n• Slots prioritarios kp <= 12 en ventana de 28d\n• Preservación íntegra de compras habituales\n• Rescate causal solo en puestos vacíos\n\nImpacto de Negocio:\n• Maximización del Customer Lifetime Value (LTV)\n• Fricción cero en reposición de básicos\n• Blindaje contra fuga hacia la competencia",
         C_GREEN, 0.68, 0.40, 0.29, 0.46),
    ]

    for title, desc, col, x, y, w, h in stages:
        add_card(ax, x, y, w, h, title, desc, bg="white", border=col, lw=2.2, title_color=col, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.03, 0.08, 0.94, 0.26, "Retorno de Inversión (ROI), Eficiencia de Costes Cloud y Merchandising Auditable",
             "• Ahorro de Costes Cloud: 0,00 € de cómputo local (AMD Ryzen 5 5500, 12 GB RAM) frente a ~185 €/mes en instancias dedicadas AWS EC2 r6i.xlarge.\n• Eficiencia Operativa de Cómputo: Procesamiento out-of-core y booster C++ sin sobrecoste de clústeres elásticos ni aceleradores GPU dedicados.\n• Captura de Ventas Reales: Recall@12 sube +10.99% (0.05640 a 0.06260), rescatando 11 de cada 100 compras efectivas que antes se perdían por desalineación.\n• Merchandising Confiable (XAI): Explicabilidad matemática aditiva con TreeSHAP sobre 3 arquetipos; elimina la desconfianza comercial de la 'caja negra'.",
             bg="#fef9e7", border=C_HIGHLIGHT, lw=2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8.2)

    add_arrow(ax, 0.32, 0.63, 0.355, 0.63, color=C_PRIMARY, lw=2.2, label="1ª Compra")
    add_arrow(ax, 0.645, 0.63, 0.68, 0.63, color=C_SECONDARY, lw=2.2, label="Recurrencia")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_08_business_conversion_lifecycle.png", dpi=300)
    plt.close(fig)


# DIAGRAMA CAP 02-01: TAXONOMÍA DE ENFOQUES DE RECOMENDACIÓN
def generate_diag_cap02_01():
    print("-> Generando diag_cap02_01_recsys_approaches.png...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Taxonomía Formal de Enfoques de Recomendación en Comercio Electrónico",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.91, "Evolución metodológica: de filtrado heurístico unidimensional a arquitecturas híbridas bietápicas desacopladas",
            fontsize=9.5, color=C_MUTED, ha="center")

    add_card(ax, 0.03, 0.38, 0.29, 0.48, "1. Filtrado Basado en Contenido (CBF)",
             "Principio Teórico:\nSimilitud Coseno / TF-IDF sobre atributos de catálogo\n(categoría, color, departamento, descripción textual).\n\n• Ventajas Operativas:\n  - Resuelve arranque en frío de ítems (item cold start)\n  - Independiente de transacciones de otros usuarios\n\n• Limitaciones en Fast-Fashion:\n  - Sobre-especialización severa (burbuja de filtro)\n  - Incapaz de capturar impulso de demanda y estacionalidad\n  - Recomendaciones repetitivas de baja diversidad",
             bg="#f8f9fa", border=C_SECONDARY, lw=2, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.355, 0.38, 0.29, 0.48, "2. Filtrado Colaborativo Clásico (CF)",
             "Principio Teórico:\nFactorización matricial en factores latentes\n(SVD, WALS, ALS, BPR sobre matriz usuario-ítem).\n\n• Ventajas Operativas:\n  - Descubrimiento de afinidades no lineales entre clientes\n  - Alta serendipia en espacios densos de interacción\n\n• Limitaciones en Fast-Fashion:\n  - Dispersión extrema (> 99.97% en H&M; > 99.99% en 5w)\n  - Cold start en > 80% de usuarios en ventana reciente\n  - Colapso hacia medias poblacionales por regularización",
             bg="#f8f9fa", border=C_ACCENT, lw=2, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.68, 0.38, 0.29, 0.48, "3. Arquitectura Híbrida en Dos Etapas",
             "Principio Teórico:\nDesacoplamiento estricto entre recuperación rápida de candidatos\ny reordenación supervisada optimizada para ranking.\n\n• Etapa 1 (Retrieval / Recall Rápido):\n  - 8 heurísticas vectorizadas multicanal (R1 a R8)\n  - Poda del 99.92% del catálogo irrelevante (<= 100 ítems/u)\n\n• Etapa 2 (Supervised Re-Ranking / Precision):\n  - Clasificador listwise LGBMRanker (LambdaRank)\n  - Optimización posicional sobre 39 variables tabulares",
             bg="#f0f9f4", border=C_GREEN, lw=2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.03, 0.08, 0.94, 0.24, "Conclusión de Diseño: Superación de Compensaciones Arquitectónicas (Trade-Offs)",
             "• La arquitectura híbrida bietápica adoptada supera simultáneamente la sobre-especialización del filtrado por contenido y el colapso por dispersión de la factorización matricial.\n• La combinación de un pool diverso de candidatos con un ranker supervisado listwise y cascada no destructiva garantiza personalización profunda, adaptabilidad estacional y cobertura al 100% de la población.",
             bg="#eef2f7", border=C_PRIMARY, lw=1.8, title_color=C_PRIMARY, fontsize=10, sub_size=8.3)

    add_arrow(ax, 0.32, 0.62, 0.355, 0.62, color=C_SECONDARY, lw=2.0)
    add_arrow(ax, 0.645, 0.62, 0.68, 0.62, color=C_SECONDARY, lw=2.0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_cap02_01_recsys_approaches.png", dpi=300)
    plt.close(fig)


# DIAGRAMA CAP 02-02: EMBUDO TWO-STAGE
def generate_diag_cap02_02():
    print("-> Generando diag_cap02_02_two_stage_funnel.png...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Arquitectura Bietápica Desacoplada: Embudo de Reducción y Precisión",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.91, "Descomposición del espacio combinatorio de 1.44 x 10^11 pares usuario-artículo | Covington et al. (YouTube, 2016)",
            fontsize=9.5, color=C_MUTED, ha="center")

    add_card(ax, 0.03, 0.38, 0.26, 0.48, "Catálogo Total",
             "Universo de Artículos en Inventario:\n\n• 105.542 prendas registradas\n• 1.371.980 clientes registrados\n• 31.78M transacciones históricas\n\nEspacio Combinatorio Teórico:\n1.37M × 105.5K = 1.44 × 10^11 pares\n\n[Inviable computacionalmente en tiempo real:\nScoring global O(|U| × |I|) saturaría servidores]",
             bg="#fdf0f0", border=C_ACCENT, lw=2, fontsize=11, sub_size=8.5)

    add_card(ax, 0.37, 0.38, 0.26, 0.48, "Pool de Candidatos",
             "Espacio Acotado de Alta Cobertura:\n\n• <= 100 candidatos por cliente activo\n• Poda selectiva del 99.92% del catálogo\n• 16.57M pares viables en 1.4 GB RAM\n\n8 Heurísticas Vectorizadas:\nRecompra, Co-ocurrencias, Cohortes de edad,\nCanal, Bestsellers con decaimiento (tau=7d)\n\n[Recall@80 = 8.44% | Hit Rate = 16.52%]",
             bg="#eef2f7", border=C_SECONDARY, lw=2, fontsize=11, sub_size=8.5)

    add_card(ax, 0.71, 0.38, 0.26, 0.48, "Top-12 Recomendaciones",
             "Selección Final Optimizada:\n\n• Exactamente 12 prendas ordenadas\n• Evaluación listwise supervisada (MAP@12)\n• 39 variables tabulares de interacción\n\nCascada No Destructiva V8:\n1. Recompra priorizada por ranker\n2. Cesta causal P(B|A)\n3. Superventas por cohorte etaria\n4. Popularidad global decaída (tau=7d)",
             bg="#f0f9f4", border=C_GREEN, lw=2, title_color=C_PRIMARY, fontsize=11, sub_size=8.5)

    add_arrow(ax, 0.29, 0.62, 0.37, 0.62, color=C_SECONDARY, lw=2.5, label="Etapa 1: Recall\n8 Heurísticas Rápidas")
    add_arrow(ax, 0.63, 0.62, 0.71, 0.62, color=C_GREEN, lw=2.5, label="Etapa 2: Precision\nLGBMRanker + Cascada V8")

    add_card(ax, 0.03, 0.08, 0.94, 0.24, "Garantías de Ingeniería, Eficiencia y Rendimiento Operativo (SLAs)",
             "• Complejidad Asintótica Reducida: La descomposición bietápica contrae la complejidad de O(|U| × |I|) a O(|U| × K_cand × T × D), reduciendo el cómputo en cuatro órdenes de magnitud.\n• Presupuesto de Latencia: Inferencia completa en < 35 ms p95 en FastAPI (candidatos precalculados ~15 ms + scoring LightGBM ~1 ms + ensamblaje de cascada < 1 ms).\n• Contención en Memoria: Ejecutable íntegramente en un equipo local con 12 GB RAM sin requerir clústeres distribuidos ni infraestructura de GPUs dedicadas.",
             bg="#f8f9fa", border=C_PRIMARY, lw=1.8, title_color=C_PRIMARY, fontsize=10, sub_size=8.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_cap02_02_two_stage_funnel.png", dpi=300)
    plt.close(fig)


# DIAGRAMA CAP 02-03: TAXONOMÍA LEARNING TO RANK (L2R)
def generate_diag_cap02_03():
    print("-> Generando diag_cap02_03_l2r_taxonomy.png...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Taxonomía Formal de Paradigmas de Learning to Rank (L2R)",
            fontsize=15, fontweight="bold", color=C_PRIMARY, ha="center")
    ax.text(0.5, 0.91, "Comparación teórica y matemática de funciones de pérdida para optimización posicional en comercio electrónico",
            fontsize=9.5, color=C_MUTED, ha="center")

    add_card(ax, 0.03, 0.38, 0.29, 0.48, "1. Paradigma Pointwise",
             "Formulación de Pérdida:\nEntropía cruzada binaria sobre cada par (u, i) aislado:\nL(y, s) = -[y log sigma(s) + (1-y) log(1-sigma(s))]\n\n• Limitación Teórica Fundamental:\n  - Trata todos los pares por igual sin noción de lista.\n  - Un error en posición 1 frente a 2 recibe idéntica penalización que en posición 50 frente a 51.\n\n• Evidencia Experimental en Proyecto:\n  - MAP@12 = 0.02206 | Recall@12 = 0.05163\n  - Inadecuado para evaluar la cabeza del ranking Top-12.",
             bg="#f8f9fa", border=C_MUTED, lw=2, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.355, 0.38, 0.29, 0.48, "2. Paradigma Pairwise (RankNet)",
             "Formulación de Pérdida:\nSigmoide sobre diferencias relativas de puntuación:\nL(s_i, s_j) = log(1 + e^-(s_i - s_j)) para pares y_i > y_j\n\n• Avance Metodológico:\n  - Modela el orden relativo entre pares de prendas (comprada frente a no comprada).\n\n• Limitación Teórica Persistente:\n  - Ponderación uniforme a lo largo de toda la lista.\n  - Insensible a la posición ordinal absoluta: no prioriza maximizar la métrica en las primeras posiciones.",
             bg="#f8f9fa", border=C_SECONDARY, lw=2, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.68, 0.38, 0.29, 0.48, "3. Paradigma Listwise (LambdaRank)",
             "Formulación de Pseudo-Gradientes:\nGradientes de pares escalados por la ganancia posicional:\nlambda_ij = [-sigma / (1 + e^sigma(s_i - s_j))] * |Delta NDCG_ij|\n\n• Superación del Reto No Derivable:\n  - MAP y NDCG son discontinuas; LambdaRank resuelve la optimización mediante saltos proporcionales a Delta NDCG.\n\n• Evidencia Experimental en Proyecto:\n  - MAP@12 = 0.02372 | Recall@12 = 0.05239\n  - Ganancia neta de +7.5% en MAP@12 sobre pointwise.",
             bg="#f0f9f4", border=C_GREEN, lw=2, title_color=C_PRIMARY, fontsize=10.5, sub_size=8.2)

    add_card(ax, 0.03, 0.08, 0.94, 0.24, "Justificación de la Selección de LambdaRank en LGBMRanker para Producción",
             "• Alineamiento con la Métrica de Negocio: En comercio electrónico minorista, los ingresos y la satisfacción dependen exclusivamente de los ítems mostrados en el Top-12. LambdaRank concentra los gradientes de ajuste en la cabeza de la lista recomendada.\n• Eficiencia en Gradientes por Histograma: La implementación de LightGBM sobre 39 variables tabulares permite un entrenamiento rápido (< 10 minutos en 12 GB RAM) con convergencia estable en 50 árboles de decisión, satisfaciendo el estándar SOTA.",
             bg="#eef2f7", border=C_PRIMARY, lw=1.8, title_color=C_PRIMARY, fontsize=10, sub_size=8.3)

    add_arrow(ax, 0.32, 0.62, 0.355, 0.62, color=C_SECONDARY, lw=2.0)
    add_arrow(ax, 0.645, 0.62, 0.68, 0.62, color=C_SECONDARY, lw=2.0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diag_cap02_03_l2r_taxonomy.png", dpi=300)
    plt.close(fig)


def main():
    print("=" * 75)
    print("  GENERANDO DIAGRAMAS ARQUITECTÓNICOS EN IMÁGENES PNG")
    print(f"  Destino: {FIGURES_DIR}")
    print("=" * 75)
    generate_diag_01()
    generate_diag_02()
    generate_diag_03()
    generate_diag_04()
    generate_diag_05()
    generate_diag_06()
    generate_diag_07()
    generate_diag_08()
    generate_diag_cap02_01()
    generate_diag_cap02_02()
    generate_diag_cap02_03()
    print("=" * 75)
    print("  [OK] Todos los 11 diagramas generados exitosamente en results/figures/")
    print("=" * 75)


if __name__ == "__main__":
    main()
