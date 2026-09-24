"""Módulo de generación y consolidación de candidatos de recomendación."""

from src.candidates.generators import (
    consolidate_candidates,
    generate_age_group_popularity,
    generate_channel_popularity,
    generate_global_popularity,
    generate_item_cf,
    generate_product_family,
    generate_repurchase,
    generate_trending_items,
    generate_user_dept_popularity,
    get_age_group_fallback_items,
    get_popular_fallback_items,
)

__all__ = [
    "generate_repurchase",
    "generate_global_popularity",
    "generate_age_group_popularity",
    "generate_channel_popularity",
    "generate_item_cf",
    "generate_product_family",
    "generate_trending_items",
    "generate_user_dept_popularity",
    "consolidate_candidates",
    "get_popular_fallback_items",
    "get_age_group_fallback_items",
]
