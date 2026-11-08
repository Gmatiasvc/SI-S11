"""Paquete de segmentación no supervisada de clientes de comercio electrónico (Trabajo 01)."""
from src.ecommerce.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_DICTIONARY,
    build_customer_rfm_dataset,
)
from src.ecommerce.pca_transform import run_pca_analysis
from src.ecommerce.clustering import (
    compute_k_distance_curve,
    explore_dbscan,
    explore_kmeans,
)
from src.ecommerce.evaluation import (
    build_fair_comparison_matrix,
    evaluate_stability,
)
from src.ecommerce.business_insights import (
    characterize_clusters,
    generate_commercial_strategies,
    run_full_ecommerce_pipeline,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_DICTIONARY",
    "build_customer_rfm_dataset",
    "run_pca_analysis",
    "compute_k_distance_curve",
    "explore_dbscan",
    "explore_kmeans",
    "build_fair_comparison_matrix",
    "evaluate_stability",
    "characterize_clusters",
    "generate_commercial_strategies",
    "run_full_ecommerce_pipeline",
]
