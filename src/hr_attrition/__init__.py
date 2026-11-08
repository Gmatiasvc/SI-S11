"""Paquete supervisado para predicción de rotación de empleados (Grupo 08 - HR Analytics)."""
from src.hr_attrition.preprocessor import (
    prepare_features_and_target,
    split_train_test,
    build_preprocessor,
)
from src.hr_attrition.models import get_base_pipelines, get_hyperparameter_grids
from src.hr_attrition.training import (
    get_shared_cv,
    evaluate_baseline_cv,
    optimize_hyperparameters,
    find_optimal_threshold_oof,
)
from src.hr_attrition.evaluation import (
    evaluate_on_test_once,
    analyze_interpretability_and_tenure,
)
from src.hr_attrition.inference import (
    get_five_synthetic_profiles,
    predict_employee_profiles,
    run_full_hr_pipeline,
)

__all__ = [
    "prepare_features_and_target",
    "split_train_test",
    "build_preprocessor",
    "get_base_pipelines",
    "get_hyperparameter_grids",
    "get_shared_cv",
    "evaluate_baseline_cv",
    "optimize_hyperparameters",
    "find_optimal_threshold_oof",
    "evaluate_on_test_once",
    "analyze_interpretability_and_tenure",
    "get_five_synthetic_profiles",
    "predict_employee_profiles",
    "run_full_hr_pipeline",
]
