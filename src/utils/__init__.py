"""Utilidades compartidas de registro de métricas y contenido académico."""
from src.utils.academic_content import (
    ECOMMERCE_ACADEMIC_CONTENT,
    HR_ACADEMIC_CONTENT,
    format_ecommerce_defense_answers,
    format_hr_defense_answers,
    get_ecommerce_commercial_strategies,
)
from src.utils.metrics_logger import load_metrics_json, save_metrics_json

__all__ = [
    "ECOMMERCE_ACADEMIC_CONTENT",
    "HR_ACADEMIC_CONTENT",
    "format_ecommerce_defense_answers",
    "format_hr_defense_answers",
    "get_ecommerce_commercial_strategies",
    "load_metrics_json",
    "save_metrics_json",
]

