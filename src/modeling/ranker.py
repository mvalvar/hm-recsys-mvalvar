"""Módulo de Modelización y Re-Ranking Supervisado (LGBMRanker con LambdaRank).

Implementa la arquitectura Two-Stage RecSys optimizada para ranking de catálogo:
1. Función de pérdida pairwise/listwise (objective='lambdarank') calibrada contra MAP@12.
2. Gestión estricta del vector 'group' (query lengths) por usuario contiguo.
3. Early stopping sobre conjunto de validación temporal retenido.
4. Extracción formal de Feature Importance por Ganancia (Gain) y Frecuencia de Corte (Split).
5. Inferencia vectorizada con empaquetado Top-12 y cálculo de MAP@12.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from config.settings import LGBM_PARAMS, MODELS_DIR, NEGATIVE_RATIO, RANDOM_SEED

logger = logging.getLogger(__name__)


class LGBMRankerModel:
    """Wrapper de producción para entrenamiento, serialización e inferencia con LGBMRanker."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or LGBM_PARAMS.copy()
        self.model: lgb.LGBMRanker | None = None
        self._booster: lgb.Booster | None = None
        self.feature_names: list[str] = []

    @property
    def booster(self) -> lgb.Booster | None:
        """Retorna el booster subyacente de LightGBM (entrenado o cargado)."""
        if self.model is not None and hasattr(self.model, "booster_"):
            return self.model.booster_
        return self._booster

    def fit(
        self,
        X: np.ndarray | pl.DataFrame,
        y: np.ndarray | pl.Series,
        groups: np.ndarray | list[int],
        feature_names: list[str] | None = None,
        eval_set: list[tuple[Any, Any]] | None = None,
        eval_group: list[list[int]] | None = None,
        early_stopping_rounds: int | None = 20,
        verbose_eval: int | bool = 50,
    ) -> LGBMRankerModel:
        r"""Entrena el modelo LGBMRanker con las consultas agrupadas contiguamente por usuario.

        Parameters
        ----------
        X : np.ndarray | pl.DataFrame
            Matriz de características de entrenamiento.
        y : np.ndarray | pl.Series
            Etiquetas binarias de relevancia (1: comprado, 0: no comprado).
        groups : np.ndarray | list[int]
            Longitudes de cada consulta por usuario ($\sum g_i == \text{len}(X)$).
        feature_names : list[str] | None, optional
            Nombres de las características.
        eval_set : list[tuple[Any, Any]] | None, optional
            Conjunto de validación temporal [(X_val, y_val)].
        eval_group : list[list[int]] | None, optional
            Vector de grupos correspondiente a eval_set.
        early_stopping_rounds : int | None, optional
            Rondas de paciencia antes de detener el entrenamiento (defecto 20).
        verbose_eval : int | bool, optional
            Frecuencia de logging durante el boosting (defecto cada 50 iteraciones).

        Returns
        -------
        LGBMRankerModel
            Instancia entrenada.
        """
        assert len(groups) > 0, "El array de grupos no puede estar vacío"
        n_rows = X.height if isinstance(X, pl.DataFrame) else len(X)
        assert sum(groups) == n_rows, (
            f"La suma de elementos de grupo ({sum(groups)}) no coincide con el total de filas ({n_rows})"
        )

        if isinstance(X, pl.DataFrame):
            self.feature_names = feature_names or X.columns
            X_train = X.to_numpy()
        else:
            self.feature_names = feature_names or []
            X_train = X

        y_train = y.to_numpy() if isinstance(y, pl.Series) else y
        groups_train = list(groups)

        # Preparar callbacks de LightGBM
        callbacks = []
        if early_stopping_rounds and eval_set:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=bool(verbose_eval)
                )
            )
        if verbose_eval and isinstance(verbose_eval, (int, bool)):
            period = verbose_eval if isinstance(verbose_eval, int) else 50
            callbacks.append(lgb.log_evaluation(period=period))

        # Preparar eval_set convirtiendo a numpy si son DataFrames
        formatted_eval_set = None
        if eval_set:
            formatted_eval_set = []
            for ev_X, ev_y in eval_set:
                x_arr = ev_X.to_numpy() if isinstance(ev_X, pl.DataFrame) else ev_X
                y_arr = ev_y.to_numpy() if isinstance(ev_y, pl.Series) else ev_y
                formatted_eval_set.append((x_arr, y_arr))

        self.model = lgb.LGBMRanker(**self.params)

        logger.info(
            f"-> Entrenando LGBMRanker (LambdaRank) con {len(groups_train):,} consultas "
            f"({len(y_train):,} pares candidato)..."
        )
        self.model.fit(
            X_train,
            y_train,
            group=groups_train,
            eval_set=formatted_eval_set,
            eval_group=eval_group,
            callbacks=callbacks if callbacks else None,
        )

        self._booster = self.model.booster_
        best_iter = getattr(self.model, "best_iteration_", self.model.n_estimators)
        logger.info(f"[OK] Entrenamiento completado. Mejor iteración alcanzada: {best_iter}")
        return self

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Genera puntuaciones continuas de relevancia para los pares provistos."""
        assert self.booster is not None, "El modelo debe ser entrenado o cargado antes de inferir"
        if isinstance(X, pl.DataFrame):
            if self.feature_names and set(self.feature_names).issubset(set(X.columns)):
                X_eval = X.select(self.feature_names).to_numpy()
            else:
                X_eval = X.to_numpy()
        else:
            X_eval = X
        return self.booster.predict(X_eval)

    def get_feature_importance(self) -> pl.DataFrame:
        r"""Calcula y retorna la importancia de las características por Ganancia y Frecuencia de División.

        Returns
        -------
        pl.DataFrame
            DataFrame con ['feature', 'gain_importance', 'gain_pct', 'split_importance'] ordenado por gain desc.
        """
        assert self.booster is not None, "El modelo no ha sido entrenado"
        gain_imp = self.booster.feature_importance(importance_type="gain")
        split_imp = self.booster.feature_importance(importance_type="split")

        names = self.feature_names or [f"feature_{i}" for i in range(len(gain_imp))]
        total_gain = float(np.sum(gain_imp)) or 1.0

        imp_df = (
            pl.DataFrame(
                {
                    "feature": names,
                    "gain_importance": gain_imp.astype(float),
                    "split_importance": split_imp.astype(int),
                }
            )
            .with_columns(
                (pl.col("gain_importance") / total_gain * 100.0).cast(pl.Float32).alias("gain_pct")
            )
            .sort("gain_importance", descending=True)
        )
        return imp_df

    def save(self, output_path: Path | str = MODELS_DIR / "lgbm_ranker.txt") -> None:
        """Serializa el modelo a disco en formato nativo de LightGBM junto con sus metadatos."""
        assert self.booster is not None, "No hay modelo entrenado para serializar"
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(out_file))

        # Registro de orden de features para reproducibilidad de inferencia
        meta_file = out_file.with_name(f"{out_file.stem}_meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"feature_names": self.feature_names, "params": self.params}, f, indent=2)

        logger.info(f"[OK] Modelo LGBMRanker serializado en {out_file.name} ({meta_file.name})")

    @classmethod
    def load(cls, model_path: Path | str) -> LGBMRankerModel:
        """Carga un modelo booster y sus metadatos desde disco."""
        model_file = Path(model_path)
        assert model_file.exists(), f"Archivo de modelo no encontrado: {model_file}"

        meta_file = model_file.with_name(f"{model_file.stem}_meta.json")
        params = LGBM_PARAMS.copy()
        feature_names: list[str] = []

        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
                feature_names = meta.get("feature_names", [])
                params.update(meta.get("params", {}))

        instance = cls(params=params)
        booster = lgb.Booster(model_file=str(model_file))
        instance._booster = booster
        instance.feature_names = feature_names
        return instance


def prepare_ranking_data(
    feature_matrix: pl.DataFrame,
    ground_truth: pl.DataFrame,
    negative_ratio: int | None = NEGATIVE_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pl.DataFrame, pl.Series, list[int], list[str]]:
    """Construye las matrices de entrenamiento delegando en prepare_ranker_split.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz con candidatos y características calculadas.
    ground_truth : pl.DataFrame
        Transacciones reales de compra con columnas ['customer_idx', 'article_id'].
    negative_ratio : int | None, optional
        Número de ejemplos negativos retenidos por cada positivo (por defecto NEGATIVE_RATIO = 5).
    seed : int, optional
        Semilla determinista.

    Returns
    -------
    tuple
        (X_df, y_series, groups, feature_names)
    """
    from src.utils.validation import prepare_ranker_split

    split_data = prepare_ranker_split(
        feature_matrix=feature_matrix,
        ground_truth_df=ground_truth,
        negative_ratio=negative_ratio,
        drop_empty_queries=True,
        seed=seed,
    )
    return split_data.X, split_data.y, split_data.groups, split_data.feature_names
