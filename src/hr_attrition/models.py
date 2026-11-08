"""Etapa 08 y Sección 5 del Grupo 08: Construcción de Pipelines y grillas de hiperparámetros."""
from typing import Any, Dict, List
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.hr_attrition.preprocessor import build_preprocessor


def get_base_pipelines(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Pipeline]:
    """
    Etapa 08:
    Integra preprocesamiento (ColumnTransformer) y estimador en un Pipeline por algoritmo:
    1. DecisionTree: DecisionTreeClassifier(random_state=42, class_weight='balanced')
    2. RandomForest: RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    3. GradientBoosting: GradientBoostingClassifier(random_state=42)
    """
    pipelines: Dict[str, Pipeline] = {
        "DecisionTree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return pipelines


def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """
    Sección 5 del PDF Grupo 08:
    Espacios de búsqueda de hiperparámetros obligatorios para los 3 algoritmos.
    """
    return {
        "DecisionTree": {
            "classifier__max_depth": [3, 4, 5, 6, 8, 10, 12, 15, None],
            "classifier__min_samples_split": [2, 5, 10, 15, 20, 30],
            "classifier__min_samples_leaf": [1, 2, 4, 6, 8, 10, 15],
            "classifier__max_features": [None, "sqrt", "log2", 0.7, 0.85],
            "classifier__ccp_alpha": [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01],
            "classifier__class_weight": ["balanced", None],
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 150, 200, 250, 300, 400],
            "classifier__max_depth": [4, 6, 8, 10, 12, 15, 20, None],
            "classifier__min_samples_split": [2, 5, 10, 15, 20],
            "classifier__min_samples_leaf": [1, 2, 3, 5, 8, 10],
            "classifier__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "classifier__bootstrap": [True, False],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [80, 120, 150, 200, 250, 300],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
            "classifier__max_depth": [2, 3, 4, 5, 6],
            "classifier__min_samples_split": [2, 5, 10, 15, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 6, 10],
            "classifier__subsample": [0.65, 0.75, 0.85, 0.9, 1.0],
        },
    }
