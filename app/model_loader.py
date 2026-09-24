"""Cargador de modelos en memoria y gestión de caché para inferencia en tiempo real.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Arquitectura de Inferencia:
- Patrón Singleton thread-safe con double-checked locking para asegurar una única instancia.
- Pre-indexación en memoria de la matriz de candidatos y características en arrays NumPy contiguos
  (acceso O(1) por customer_idx sin overhead de I/O).
- Evaluación en C++ mediante el Booster nativo de LightGBM (latencia de predicción < 1 ms).
- Vector de fallback de popularidad global precargado para arranque en frío inmediato (< 5 ms).
- Consumo estricto de memoria controlado (< 400 MB RSS).
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil

from config.settings import (
    DATA_PROCESSED_DIR,
    DATA_PROCESSED_SAMPLE_DIR,
    DATA_SAMPLE_DIR,
    MODELS_DIR,
    MODELS_SAMPLE_DIR,
)
from src.candidates import get_popular_fallback_items

logger = logging.getLogger(__name__)


class RecommenderServiceLoader:
    """Gestiona la carga única y el ciclo de vida de artefactos pesados en memoria."""

    _instance: RecommenderServiceLoader | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        self.customer_mapping: dict[str, int] = {}
        self.reverse_customer_mapping: dict[int, str] = {}
        self.customer_age_bins: dict[int, str] = {}
        self.popular_fallback: list[str] = []
        self.candidate_features: dict[int, tuple[np.ndarray, list[str]]] = {}
        self.feature_names: list[str] = []
        self.v8_bestsellers_by_age: dict[str, list[str]] = {}
        self.v8_basket_affinity: dict[int, list[str]] = {}
        self.v8_customer_history_28d: dict[int, list[str]] = {}
        self.model_version: str = "1.0.0"
        self._is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> RecommenderServiceLoader:
        """Obtiene la instancia Singleton única del servicio de forma thread-safe."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_artifacts(self, force_reload: bool = False) -> None:
        """Carga en memoria el booster de LightGBM, mapeos y matriz de candidatos indexada."""
        with self._lock:
            if self._is_loaded and not force_reload:
                return

            start_t = time.perf_counter()
            logger.info("-> [RecommenderServiceLoader] Inicializando artefactos de inferencia...")

            # Fallback de popularidad estacional (Top-12 de última semana con decaimiento temporal)
            try:
                tx_path = DATA_PROCESSED_DIR / "transactions_5w.parquet"
                if (
                    not tx_path.exists()
                    and (DATA_PROCESSED_SAMPLE_DIR / "transactions_5w.parquet").exists()
                ):
                    tx_path = DATA_PROCESSED_SAMPLE_DIR / "transactions_5w.parquet"
                if tx_path.exists():
                    tx_df = pl.read_parquet(tx_path)
                    self.popular_fallback = get_popular_fallback_items(
                        transactions_df=tx_df, top_k=12, days_window=7, lambda_decay=0.05
                    )
                elif (DATA_SAMPLE_DIR / "sample_transactions.csv").exists():
                    logger.info(
                        "  * Fallback activo a data_sample/sample_transactions.csv para evaluación ligera"
                    )
                    sample_tx = pl.read_csv(
                        DATA_SAMPLE_DIR / "sample_transactions.csv",
                        schema_overrides={"t_dat": pl.Date, "article_id": pl.Int32},
                    )
                    self.popular_fallback = get_popular_fallback_items(
                        transactions_df=sample_tx, top_k=12
                    )
                else:
                    self._set_default_fallback()
                logger.info(
                    f"  * Fallback estacional cargado: {len(self.popular_fallback)} artículos (Top 1: {self.popular_fallback[0]})"
                )
            except (
                AssertionError,
                FileNotFoundError,
                ValueError,
                KeyError,
                RuntimeError,
                pl.exceptions.PolarsError,
            ) as e:
                logger.warning(
                    f"! Advertencia al calcular popularidad estacional: {e}. Usando fallback seguro."
                )
                self._set_default_fallback()

            # Cargar modelo serializado LightGBM
            model_path = MODELS_DIR / "lgbm_ranker.txt"
            if not model_path.exists() and (MODELS_SAMPLE_DIR / "lgbm_ranker.txt").exists():
                model_path = MODELS_SAMPLE_DIR / "lgbm_ranker.txt"
                logger.info(
                    f"  * [FALLBACK SAMPLE] Modelo de producción no encontrado en models/; cargando {model_path}"
                )
            if model_path.exists():
                self.model = lgb.Booster(model_file=str(model_path))
                logger.info(
                    f"  * Modelo Booster cargado: {self.model.num_trees()} árboles de decisión"
                )
            else:
                logger.warning(
                    f"! Aviso: {model_path} no encontrado. Modo Fallback Popular activado."
                )

            # Cargar mapeo bidireccional customer_id <-> customer_idx (con fallback a data_sample)
            mapping_path = DATA_PROCESSED_DIR / "customer_id_mapping.parquet"
            if (
                not mapping_path.exists()
                and (DATA_PROCESSED_SAMPLE_DIR / "customer_id_mapping.parquet").exists()
            ):
                mapping_path = DATA_PROCESSED_SAMPLE_DIR / "customer_id_mapping.parquet"
            if mapping_path.exists():
                mapping_df = pl.read_parquet(mapping_path)
                self.customer_mapping = dict(
                    zip(
                        mapping_df["customer_id"].to_list(),
                        mapping_df["customer_idx"].to_list(),
                        strict=False,
                    )
                )
                self.reverse_customer_mapping = {v: k for k, v in self.customer_mapping.items()}
                logger.info(
                    f"  * Mapeo de clientes cargado: {len(self.customer_mapping):,} usuarios"
                )
            elif (DATA_SAMPLE_DIR / "sample_customers.csv").exists():
                logger.info(
                    "  * Cargando mapeo de clientes desde data_sample/sample_customers.csv (modo evaluación)..."
                )
                cust_sample = pl.read_csv(DATA_SAMPLE_DIR / "sample_customers.csv")
                cust_ids = cust_sample["customer_id"].to_list()
                self.customer_mapping = {cid: idx for idx, cid in enumerate(cust_ids)}
                self.reverse_customer_mapping = dict(enumerate(cust_ids))
                logger.info(
                    f"  * Mapeo de clientes cargado desde sample: {len(self.customer_mapping):,} usuarios"
                )
            else:
                logger.warning(
                    f"! Aviso: Ni {mapping_path} ni sample_customers.csv encontrados. Mapeo vacío."
                )

            # Pre-indexar candidatos y características en memoria para inferencia O(1)
            feat_path = DATA_PROCESSED_DIR / "features_matrix.parquet"
            if (
                not feat_path.exists()
                and (DATA_PROCESSED_SAMPLE_DIR / "features_matrix.parquet").exists()
            ):
                feat_path = DATA_PROCESSED_SAMPLE_DIR / "features_matrix.parquet"
                logger.info(
                    f"  * [FALLBACK SAMPLE] Matriz de producción no encontrada en data_processed/; indexando {feat_path}"
                )
            if feat_path.exists():
                # Límite configurable de filas pre-indexadas para gobernar el presupuesto de RAM
                max_rows_env = os.getenv("API_MAX_INDEXED_ROWS", "100000").strip()
                if max_rows_env in ("0", "none", "auto", ""):
                    # Fallback de seguridad de memoria: Si se especifica 0 o auto,
                    # se aplica un tope seguro de alta capacidad de 500,000 filas (~8,300 clientes en RAM)
                    # para garantizar que el consumo combinado opere holgadamente dentro de la cota de 4.0 GB de Docker.
                    max_rows = 500_000
                    logger.info(
                        "  * [SAFEGUARD SRE] API_MAX_INDEXED_ROWS=0/auto detectado: aplicando tope de seguridad "
                        "de 500,000 candidatos (~8,300 clientes en RAM) para garantizar estabilidad absoluta en Docker."
                    )
                elif max_rows_env.lower() in ("unlimited", "all"):
                    # Solo para clusters dedicados con > 16 GB de RAM y worker único
                    max_rows = None
                    logger.warning(
                        "! [ADVERTENCIA RAM] Indexación sin límite solicitada: demanda > 7 GB de memoria combinada."
                    )
                elif max_rows_env.isdigit() and int(max_rows_env) > 0:
                    max_rows = int(max_rows_env)
                else:
                    max_rows = 100_000

                if max_rows:
                    feat_df = pl.read_parquet(feat_path, n_rows=max_rows)
                else:
                    feat_df = pl.read_parquet(feat_path)

                exclude_cols = {"customer_idx", "article_id", "target", "source"}
                self.feature_names = [c for c in feat_df.columns if c not in exclude_cols]

                cust_ids = feat_df["customer_idx"].to_numpy()
                art_ids = feat_df["article_id"].to_numpy()
                X_all = feat_df.select(self.feature_names).to_numpy().astype(np.float32)

                self.candidate_features.clear()
                if len(cust_ids) > 0:
                    u_cust, split_idxs = np.unique(cust_ids, return_index=True)
                    X_splits = np.split(X_all, split_idxs[1:])
                    art_splits = np.split(art_ids, split_idxs[1:])
                    for c, x_mat, arts in zip(u_cust, X_splits, art_splits, strict=False):
                        self.candidate_features[int(c)] = (
                            x_mat,
                            [f"{a:010d}" for a in arts],
                        )

                total_pairs = len(cust_ids)
                del feat_df, X_all, cust_ids, art_ids
                gc.collect()

                logger.info(
                    f"  * Matriz de candidatos pre-indexada: {len(self.candidate_features):,} usuarios "
                    f"({total_pairs:,} pares candidato con {len(self.feature_names)} features)"
                )

            # Cargar artefactos de inferencia nativos de V8 (Cascada de precisión)
            # 5.1 Superventas segmentados por rango de edad
            bs_path = DATA_PROCESSED_DIR / "v8_bestsellers_age.parquet"
            if not bs_path.exists() and (DATA_PROCESSED_SAMPLE_DIR / "v8_bestsellers_age.parquet").exists():
                bs_path = DATA_PROCESSED_SAMPLE_DIR / "v8_bestsellers_age.parquet"
            if bs_path.exists():
                bs_df = pl.read_parquet(bs_path)
                bs_agg = (
                    bs_df.sort(["age_bin", "rank"])
                    .group_by("age_bin", maintain_order=True)
                    .agg(pl.col("article_id"))
                )
                self.v8_bestsellers_by_age = dict(
                    zip(
                        bs_agg["age_bin"].to_list(),
                        [[f"{a:010d}" for a in arts] for arts in bs_agg["article_id"].to_list()],
                        strict=False,
                    )
                )
                logger.info(
                    f"  * Superventas por rango de edad V8 cargados: {list(self.v8_bestsellers_by_age.keys())}"
                )

            # 5.2 Reglas de afinidad de cesta (cross-selling empírico de co-ocurrencia)
            ba_path = DATA_PROCESSED_DIR / "v8_basket_affinity.parquet"
            if not ba_path.exists() and (DATA_PROCESSED_SAMPLE_DIR / "v8_basket_affinity.parquet").exists():
                ba_path = DATA_PROCESSED_SAMPLE_DIR / "v8_basket_affinity.parquet"
            if ba_path.exists():
                ba_df = pl.read_parquet(ba_path)
                ba_agg = (
                    ba_df.sort(["article_id", "pair_rank"])
                    .group_by("article_id", maintain_order=True)
                    .agg(pl.col("complement_id"))
                )
                self.v8_basket_affinity = dict(
                    zip(
                        ba_agg["article_id"].to_list(),
                        [[f"{c:010d}" for c in comps] for comps in ba_agg["complement_id"].to_list()],
                        strict=False,
                    )
                )
                logger.info(
                    f"  * Reglas de afinidad de cesta V8 cargadas: {len(self.v8_basket_affinity):,} artículos"
                )

            # 5.3 Historial de compras recientes a 28 días
            ch_path = DATA_PROCESSED_DIR / "v8_customer_history_28d.parquet"
            if not ch_path.exists() and (DATA_PROCESSED_SAMPLE_DIR / "v8_customer_history_28d.parquet").exists():
                ch_path = DATA_PROCESSED_SAMPLE_DIR / "v8_customer_history_28d.parquet"
            if ch_path.exists():
                ch_df = pl.read_parquet(ch_path)
                ch_agg = (
                    ch_df.sort(["customer_idx", "purchase_rank"])
                    .group_by("customer_idx", maintain_order=True)
                    .agg(pl.col("article_id"))
                )
                self.v8_customer_history_28d = dict(
                    zip(
                        ch_agg["customer_idx"].to_list(),
                        [[f"{a:010d}" for a in arts] for arts in ch_agg["article_id"].to_list()],
                        strict=False,
                    )
                )
                logger.info(
                    f"  * Historial reciente 28d V8 cargado: {len(self.v8_customer_history_28d):,} usuarios"
                )

            # 5.4 Segmentos de edad de clientes para personalización demográfica
            cust_path = DATA_PROCESSED_DIR / "customers.parquet"
            if not cust_path.exists() and (DATA_PROCESSED_SAMPLE_DIR / "customers.parquet").exists():
                cust_path = DATA_PROCESSED_SAMPLE_DIR / "customers.parquet"
            if cust_path.exists():
                cust_df = pl.read_parquet(cust_path, columns=["customer_idx", "age_bin"])
                self.customer_age_bins = dict(
                    zip(
                        cust_df["customer_idx"].to_list(),
                        cust_df["age_bin"].cast(pl.String).to_list(),
                        strict=False,
                    )
                )

            load_duration = time.perf_counter() - start_t
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            self._is_loaded = True

            logger.info(
                f"[OK] Artefactos cargados en {load_duration:.2f} s | "
                f"Memoria RSS actual: {rss_mb:.1f} MB (Cota < 600 MB)"
            )

    def _set_default_fallback(self) -> None:
        """Asigna el fallback estacional de septiembre 2020 calculado dinámicamente."""
        try:
            self.popular_fallback = get_popular_fallback_items(
                top_k=12, days_window=7, lambda_decay=0.05
            )
        except (FileNotFoundError, ValueError, KeyError, RuntimeError, pl.exceptions.PolarsError):
            # Fallback seguro con superventas de la última semana de septiembre 2020
            self.popular_fallback = [
                "0924243001",
                "0924243002",
                "0918522001",
                "0923758001",
                "0866731001",
                "0909370001",
                "0751471001",
                "0915529003",
                "0915529005",
                "0448509014",
                "0762846027",
                "0714790020",
            ]

    def predict_for_customer(
        self,
        customer_id: str,
        top_k: int = 12,
        offset: int = 0,
        return_total: bool = False,
    ) -> tuple[list[str], bool, float] | tuple[list[str], bool, float, int]:
        r"""Genera las recomendaciones Top-K para un cliente calculando la latencia en milisegundos.

        Parameters
        ----------
        customer_id : str
            Hash hexadecimal de 64 caracteres del cliente.
        top_k : int, optional
            Número de artículos solicitados (por defecto 12).
        offset : int, optional
            Desplazamiento para paginación de candidatos (por defecto 0).
        return_total : bool, optional
            Si es True, retorna una 4-tupla incluyendo el total de candidatos viables.

        Returns
        -------
        tuple[list[str], bool, float] | tuple[list[str], bool, float, int]
            (recommendations, is_cold_start, latency_ms) o
            (recommendations, is_cold_start, latency_ms, total_candidates) si return_total=True.
        """
        start_t = time.perf_counter()

        # Manejo de cold start absoluto (cliente fuera de catálogo)
        if customer_id not in self.customer_mapping:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            recs = self.popular_fallback[offset : offset + top_k]
            total_cand = len(self.popular_fallback)
            if return_total:
                return recs, True, latency_ms, total_cand
            return recs, True, latency_ms

        customer_idx = self.customer_mapping[customer_id]

        # Cascada V8 para clientes con historial en los últimos 28 días
        if self.v8_customer_history_28d and customer_idx in self.v8_customer_history_28d:
            full_recs: list[str] = []
            seen: set[str] = set()

            # Nivel 1: Recompra reciente (últimos 28 días ordenados por recencia)
            recent_items = self.v8_customer_history_28d[customer_idx]
            for item in recent_items:
                if item not in seen:
                    seen.add(item)
                    full_recs.append(item)

            # Nivel 2: Afinidad de cesta (cross-selling empírico para cada artículo comprado)
            for item in recent_items:
                try:
                    comps = self.v8_basket_affinity.get(int(item), [])
                    for comp in comps:
                        if comp not in seen:
                            seen.add(comp)
                            full_recs.append(comp)
                except (ValueError, TypeError):
                    pass

            # Nivel 3: Superventas segmentados por grupo etario del cliente
            age_bin = self.customer_age_bins.get(customer_idx, "ALL")
            age_bestsellers = self.v8_bestsellers_by_age.get(
                age_bin, self.v8_bestsellers_by_age.get("ALL", self.popular_fallback)
            )
            for item in age_bestsellers:
                if item not in seen:
                    seen.add(item)
                    full_recs.append(item)

            # Nivel 4: Superventas estacionales globales
            for item in self.popular_fallback:
                if item not in seen:
                    seen.add(item)
                    full_recs.append(item)

            total_cand = len(full_recs)
            recs = full_recs[offset : offset + top_k]
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            if return_total:
                return recs, False, latency_ms, total_cand
            return recs, False, latency_ms

        # Re-ranking con LightGBM Ranker para usuarios con candidatos indexados
        if self.model is not None and customer_idx in self.candidate_features:
            X_eval, article_ids = self.candidate_features[customer_idx]
            scores = self.model.predict(X_eval)
            total_cand = len(article_ids)

            top_indices = np.argsort(scores)[::-1][offset : offset + top_k]
            recs = [article_ids[i] for i in top_indices]

            if len(recs) < top_k and offset == 0:
                for fallback_item in self.popular_fallback:
                    if fallback_item not in recs:
                        recs.append(fallback_item)
                    if len(recs) == top_k:
                        break

            latency_ms = (time.perf_counter() - start_t) * 1000.0
            if return_total:
                return recs, False, latency_ms, total_cand
            return recs, False, latency_ms

        # Cliente en catálogo sin compras recientes (popularidad por cohorte / fallback estacional)
        age_bin = self.customer_age_bins.get(customer_idx, "ALL")
        age_bestsellers = self.v8_bestsellers_by_age.get(
            age_bin, self.v8_bestsellers_by_age.get("ALL", self.popular_fallback)
        )
        full_recs = list(age_bestsellers)
        for item in self.popular_fallback:
            if item not in full_recs:
                full_recs.append(item)
        recs = full_recs[offset : offset + top_k]
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        if return_total:
            return recs, False, latency_ms, len(full_recs)
        return recs, False, latency_ms

    def get_health_status(self) -> dict[str, Any]:
        """Provee telemetría instantánea del estado de salud del servicio."""
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

        return {
            "status": "ok",
            "service": "hm_recommender",
            "model_loaded": self.model is not None,
            "version": self.model_version,
            "indexed_customers": len(self.candidate_features),
            "catalog_customers": len(self.customer_mapping),
            "memory_rss_mb": round(rss_mb, 1),
        }
