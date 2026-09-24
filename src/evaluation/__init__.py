"""Módulo de evaluación y experimentos de ablación."""

from src.evaluation.ablation import run_ablation_experiment
from src.evaluation.metrics import (
    ap_at_k,
    catalog_coverage,
    evaluate_ranking_df,
    hit_rate_at_k,
    map_at_k,
    recall_at_k,
)
from src.evaluation.submission import (
    build_submission,
    load_all_customer_ids,
    run_submission_pipeline,
    score_candidates,
    validate_submission,
    write_submission,
)

__all__ = [
    "ap_at_k",
    "map_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "catalog_coverage",
    "evaluate_ranking_df",
    "run_ablation_experiment",
    "score_candidates",
    "load_all_customer_ids",
    "build_submission",
    "validate_submission",
    "write_submission",
    "run_submission_pipeline",
]
