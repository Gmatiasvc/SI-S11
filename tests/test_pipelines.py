"""Suite de pruebas automatizadas de extremo a extremo y auditoría metodológica (QA & MLOps)."""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ECOMMERCE_METRICS_JSON_PATH,
    ECOMMERCE_PIPELINE_PATH,
    ECOMMERCE_RESULTS_PATH,
    HR_ALL_PIPELINES_PATH,
    HR_BEST_PIPELINE_PATH,
    HR_DROP_COLUMNS,
    HR_METRICS_JSON_PATH,
)
from src.data.download_datasets import ensure_datasets_available, validate_dataset_integrity
from src.data.loaders import load_raw_hr
from src.ecommerce.business_insights import run_full_ecommerce_pipeline
from src.hr_attrition.inference import (
    get_five_synthetic_profiles,
    predict_employee_profiles,
    run_full_hr_pipeline,
)
from src.hr_attrition.preprocessor import prepare_features_and_target, split_train_test
from src.hr_attrition.training import get_shared_cv
from src.utils.academic_content import (
    ECOMMERCE_ACADEMIC_CONTENT,
    HR_ACADEMIC_CONTENT,
    format_ecommerce_defense_answers,
    format_hr_defense_answers,
)


def test_datasets_exist_and_valid():
    """Verifica que ambos datasets estén disponibles y cumplan integridad estructural."""
    paths = ensure_datasets_available(sync_to_data_dir=True)
    assert paths["ecommerce"].exists()
    assert paths["hr_attrition"].exists()

    df_hr = load_raw_hr()
    val_hr = validate_dataset_integrity("hr_attrition", df_hr)
    assert val_hr["is_valid"] is True
    assert val_hr["row_count"] == 1470


def test_hr_no_data_leakage_and_shared_folds():
    """
    Audita la prevención estricta de data leakage en el Módulo 2 (Grupo 08):
    - Exclusión de columnas constantes e identificadores.
    - Split 80/20 estratificado reproducible.
    - Identidad exacta de los 5 folds entre llamadas a get_shared_cv().
    """
    df_hr = load_raw_hr()
    X, y, num_cols, cat_cols = prepare_features_and_target(df_hr)

    for dropped in HR_DROP_COLUMNS:
        assert dropped not in X.columns

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    assert len(X_train) == 1176
    assert len(X_test) == 294
    assert abs(y_train.mean() - y_test.mean()) < 0.005

    cv1 = get_shared_cv()
    cv2 = get_shared_cv()
    splits1 = list(cv1.split(X_train, y_train))
    splits2 = list(cv2.split(X_train, y_train))
    assert len(splits1) == 5
    for (tr1, val1), (tr2, val2) in zip(splits1, splits2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(val1, val2)


def test_hr_artifacts_and_five_synthetic_profiles():
    """Verifica que el Pipeline ganador de RRHH esté serializado y evalúe los 5 perfiles ficticios."""
    assert HR_BEST_PIPELINE_PATH.exists()
    assert HR_ALL_PIPELINES_PATH.exists()
    assert HR_METRICS_JSON_PATH.exists()

    res = run_full_hr_pipeline(force_retrain=False)
    assert res["winner_name"] in ["DecisionTree", "RandomForest", "GradientBoosting"]
    assert 0.20 <= res["optimal_threshold"] <= 0.80
    assert res["test_evaluation"]["roc_auc"] > 0.75

    profiles = get_five_synthetic_profiles()
    assert len(profiles) == 5
    preds = predict_employee_profiles(
        res["winner_pipeline"], profiles, optimal_threshold=res["optimal_threshold"]
    )
    assert len(preds) == 5
    for p in preds:
        assert 0.0 <= p["attrition_probability"] <= 1.0
        assert p["predicted_class"] in (0, 1)


def test_ecommerce_fair_comparison_and_stability():
    """
    Audita el rigor metodológico del Módulo 1 (Trabajo 01):
    - 5,878 clientes únicos y 6 variables cuantitativas.
    - Comparación justa de las 4 alternativas sobre el espacio original escalado.
    - 4 arquetipos semánticos, 3 estrategias comerciales y estabilidad ARI > 0.90.
    """
    assert ECOMMERCE_PIPELINE_PATH.exists()
    assert ECOMMERCE_RESULTS_PATH.exists()
    assert ECOMMERCE_METRICS_JSON_PATH.exists()

    ecom = run_full_ecommerce_pipeline(force_retrain=False)
    assert ecom["preprocessing"]["n_unique_customers"] >= 5800
    assert len(ecom["preprocessing"]["feature_columns"]) == 6

    matrix = ecom["fair_comparison_matrix"]
    assert len(matrix) == 4
    model_ids = [m["model_id"] for m in matrix]
    assert model_ids == ["kmeans_original", "dbscan_original", "pca_kmeans", "pca_dbscan"]

    assert ecom["stability_analysis"]["kmeans_seed_stability"]["mean_pairwise_ari"] > 0.90
    assert len(ecom["cluster_profiles"]) == 4
    assert len(ecom["commercial_strategies"]) == 3


def test_academic_content_and_streamlit_syntax():
    """Verifica el módulo de contenido académico y la compilación de las páginas Streamlit."""
    ecom = run_full_ecommerce_pipeline(force_retrain=False)
    hr = run_full_hr_pipeline(force_retrain=False)

    e_qas = format_ecommerce_defense_answers(ecom)
    h_qas = format_hr_defense_answers(hr)
    assert len(e_qas) >= 5
    assert len(h_qas) == 5
    assert "problem_definition" in ECOMMERCE_ACADEMIC_CONTENT
    assert "ethical_and_legal_analysis" in HR_ACADEMIC_CONTENT

    app_files = [
        Path("app/main.py"),
        Path("app/pages/1_📊_Overview.py"),
        Path("app/pages/2_🛒_Customer_Segmentation.py"),
        Path("app/pages/3_👥_HR_Employee_Attrition.py"),
    ]
    for fpath in app_files:
        assert fpath.exists(), f"Falta el archivo {fpath}"
        source = fpath.read_text(encoding="utf-8")
        compile(source, str(fpath), "exec")


def test_streamlit_pages_render_without_exception():
    """Ejecuta un smoke test headless de las páginas de Streamlit con AppTest."""
    from streamlit.testing.v1 import AppTest

    for page_path in [
        "app/main.py",
        "app/pages/1_📊_Overview.py",
        "app/pages/2_🛒_Customer_Segmentation.py",
        "app/pages/3_👥_HR_Employee_Attrition.py",
    ]:
        at = AppTest.from_file(str(ROOT_DIR / page_path), default_timeout=60)
        at.run()
        assert not at.exception, f"Excepción en {page_path}: {at.exception}"

