"""Módulo de explicabilidad de modelos e interpretabilidad (XAI)."""

from src.xai.shap_analysis import explain_global_shap, explain_local_customer

__all__ = ["explain_global_shap", "explain_local_customer"]
