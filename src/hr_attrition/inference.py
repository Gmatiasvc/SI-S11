"""Etapas 15 y 16 del Grupo 08: Predicción de 5 perfiles ficticios, simulador interactivo y serialización con Joblib."""
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    HR_ALL_PIPELINES_PATH,
    HR_BEST_PIPELINE_PATH,
    HR_METRICS_JSON_PATH,
)
from src.data.loaders import get_hr_eda_summary, load_raw_hr
from src.hr_attrition.evaluation import (
    analyze_interpretability_and_tenure,
    evaluate_on_test_once,
)
from src.hr_attrition.preprocessor import (
    prepare_features_and_target,
    split_train_test,
)
from src.hr_attrition.training import (
    evaluate_baseline_cv,
    get_shared_cv,
    optimize_hyperparameters,
)
from src.utils.metrics_logger import save_metrics_json


def get_five_synthetic_profiles() -> List[Dict[str, Any]]:
    """
    Etapa 15 y Sección 6 del PDF de Grupo 08:
    Define cinco registros nuevos, razonables y completamente descritos que cubren
    distintos niveles de riesgo laboral para demostrar la inferencia del Pipeline ganador.
    """
    return [
        {
            "profile_id": "PERFIL-01",
            "profile_name": "1. Analista Comercial Junior con Sobrecarga y Viajes Frecuentes",
            "narrative_description": (
                "Joven representante de ventas (26 años, soltero, 1 año en la empresa) que realiza "
                "horas extra constantes (OverTime='Yes'), viaja frecuentemente, vive a 24 km de la oficina, "
                "percibe un ingreso bajo ($2,350/mes) sin opciones sobre acciones y reporta baja satisfacción ambiental."
            ),
            "features": {
                "Age": 26,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 580,
                "Department": "Sales",
                "DistanceFromHome": 24,
                "Education": 2,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 1,
                "Gender": "Male",
                "HourlyRate": 48,
                "JobInvolvement": 2,
                "JobLevel": 1,
                "JobRole": "Sales Representative",
                "JobSatisfaction": 1,
                "MaritalStatus": "Single",
                "MonthlyIncome": 2350,
                "MonthlyRate": 12400,
                "NumCompaniesWorked": 4,
                "OverTime": "Yes",
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 3,
                "TrainingTimesLastYear": 1,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 1,
                "YearsInCurrentRole": 0,
                "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 0,
            },
        },
        {
            "profile_id": "PERFIL-02",
            "profile_name": "2. Directora de Investigación I+D Consolidada y Alto Bienestar",
            "narrative_description": (
                "Líder científica de 46 años (casada, 12 años en la empresa, nivel jerárquico 4), "
                "con ingreso competitivo ($14,800/mes), sin horas extra, corta distancia al hogar (4 km), "
                "plan de acciones nivel 2 y alta satisfacción laboral y de balance vida-trabajo."
            ),
            "features": {
                "Age": 46,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 1120,
                "Department": "Research & Development",
                "DistanceFromHome": 4,
                "Education": 4,
                "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": 4,
                "Gender": "Female",
                "HourlyRate": 88,
                "JobInvolvement": 4,
                "JobLevel": 4,
                "JobRole": "Research Director",
                "JobSatisfaction": 4,
                "MaritalStatus": "Married",
                "MonthlyIncome": 14800,
                "MonthlyRate": 21500,
                "NumCompaniesWorked": 2,
                "OverTime": "No",
                "PercentSalaryHike": 18,
                "PerformanceRating": 4,
                "RelationshipSatisfaction": 4,
                "StockOptionLevel": 2,
                "TotalWorkingYears": 21,
                "TrainingTimesLastYear": 3,
                "WorkLifeBalance": 4,
                "YearsAtCompany": 12,
                "YearsInCurrentRole": 8,
                "YearsSinceLastPromotion": 2,
                "YearsWithCurrManager": 7,
            },
        },
        {
            "profile_id": "PERFIL-03",
            "profile_name": "3. Técnico de Laboratorio Recién Ingresado con Fricción de Traslado",
            "narrative_description": (
                "Técnico de laboratorio de 29 años (2 años en la compañía, salario mensual de $2,780) "
                "que trabaja horas extra regularmente, recorre 27 km diarios hasta la planta, ha trabajado "
                "en 6 empresas previas y presenta baja satisfacción con su entorno de trabajo."
            ),
            "features": {
                "Age": 29,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 490,
                "Department": "Research & Development",
                "DistanceFromHome": 27,
                "Education": 3,
                "EducationField": "Medical",
                "EnvironmentSatisfaction": 1,
                "Gender": "Male",
                "HourlyRate": 54,
                "JobInvolvement": 2,
                "JobLevel": 1,
                "JobRole": "Laboratory Technician",
                "JobSatisfaction": 2,
                "MaritalStatus": "Single",
                "MonthlyIncome": 2780,
                "MonthlyRate": 9800,
                "NumCompaniesWorked": 6,
                "OverTime": "Yes",
                "PercentSalaryHike": 12,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 6,
                "TrainingTimesLastYear": 1,
                "WorkLifeBalance": 2,
                "YearsAtCompany": 2,
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 1,
            },
        },
        {
            "profile_id": "PERFIL-04",
            "profile_name": "4. Especialista de RRHH en Riesgo Moderado por Estancamiento de Carrera",
            "narrative_description": (
                "Profesional de Recursos Humanos de 35 años (7 años en la empresa, 7 años sin promoción) "
                "que realiza horas extra ocasionales, percibe $4,400/mes, vive a 16 km y muestra "
                "satisfacción laboral intermedia con señales de fatiga por falta de ascenso."
            ),
            "features": {
                "Age": 35,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 750,
                "Department": "Human Resources",
                "DistanceFromHome": 16,
                "Education": 3,
                "EducationField": "Human Resources",
                "EnvironmentSatisfaction": 2,
                "Gender": "Female",
                "HourlyRate": 65,
                "JobInvolvement": 2,
                "JobLevel": 2,
                "JobRole": "Human Resources",
                "JobSatisfaction": 2,
                "MaritalStatus": "Divorced",
                "MonthlyIncome": 4400,
                "MonthlyRate": 15200,
                "NumCompaniesWorked": 3,
                "OverTime": "Yes",
                "PercentSalaryHike": 12,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 1,
                "TotalWorkingYears": 10,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 2,
                "YearsAtCompany": 7,
                "YearsInCurrentRole": 6,
                "YearsSinceLastPromotion": 7,
                "YearsWithCurrManager": 5,
            },
        },
        {
            "profile_id": "PERFIL-05",
            "profile_name": "5. Ejecutivo de Ventas Senior Estable con Acciones y Sin Horas Extra",
            "narrative_description": (
                "Ejecutivo comercial de 39 años (casado, 8 años en la organización, ingreso de $7,600/mes), "
                "sin horas extra, con nivel de opciones sobre acciones 1, ascendido hace 1 año y "
                "alta satisfacción general en su equipo."
            ),
            "features": {
                "Age": 39,
                "BusinessTravel": "Non-Travel",
                "DailyRate": 980,
                "Department": "Sales",
                "DistanceFromHome": 6,
                "Education": 3,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 3,
                "Gender": "Male",
                "HourlyRate": 76,
                "JobInvolvement": 3,
                "JobLevel": 3,
                "JobRole": "Sales Executive",
                "JobSatisfaction": 4,
                "MaritalStatus": "Married",
                "MonthlyIncome": 7600,
                "MonthlyRate": 17800,
                "NumCompaniesWorked": 1,
                "OverTime": "No",
                "PercentSalaryHike": 16,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StockOptionLevel": 1,
                "TotalWorkingYears": 14,
                "TrainingTimesLastYear": 3,
                "WorkLifeBalance": 3,
                "YearsAtCompany": 8,
                "YearsInCurrentRole": 5,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 6,
            },
        },
    ]


def _extract_risk_drivers(features: Dict[str, Any]) -> List[str]:
    """Identifica las señales preventivas principales de riesgo o retención de un perfil."""
    drivers: List[str] = []
    if features.get("OverTime") == "Yes":
        drivers.append("Horas extra activas (OverTime = Yes)")
    if float(features.get("MonthlyIncome", 5000)) < 3200:
        drivers.append(f"Ingreso mensual bajo (${features.get('MonthlyIncome'):,.0f})")
    if float(features.get("YearsAtCompany", 5)) <= 2:
        drivers.append(f"Baja antigüedad ({features.get('YearsAtCompany')} años en la empresa)")
    if float(features.get("DistanceFromHome", 5)) >= 15:
        drivers.append(f"Distancia elevada al trabajo ({features.get('DistanceFromHome')} km)")
    if features.get("BusinessTravel") == "Travel_Frequently":
        drivers.append("Viajes de negocio frecuentes")
    if int(features.get("EnvironmentSatisfaction", 3)) <= 2 or int(features.get("JobSatisfaction", 3)) <= 2:
        drivers.append("Baja satisfacción laboral / ambiental (<= 2/4)")
    if int(features.get("StockOptionLevel", 1)) == 0:
        drivers.append("Sin incentivos de acciones (StockOptionLevel = 0)")
    if int(features.get("YearsSinceLastPromotion", 0)) >= 5:
        drivers.append(f"Estancamiento sin promoción ({features.get('YearsSinceLastPromotion')} años)")

    if not drivers:
        drivers.append("Factores protectores sólidos: sin horas extra, remuneración competitiva y buen clima.")
    return drivers


def predict_employee_profiles(
    pipeline: Pipeline,
    profiles: List[Dict[str, Any]],
    optimal_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Evalúa una lista de perfiles de empleados (o un único registro del simulador)
    usando el Pipeline ganador serializado, sin reentrenar ni buscar hiperparámetros.
    """
    rows = [p["features"] for p in profiles]
    df_in = pd.DataFrame(rows)
    for col in df_in.columns:
        if not pd.api.types.is_numeric_dtype(df_in[col]):
            df_in[col] = df_in[col].astype(object)

    probas = pipeline.predict_proba(df_in)[:, 1]
    outputs: List[Dict[str, Any]] = []

    for p, prob in zip(profiles, probas):
        prob_val = float(prob)
        pred_class = int(prob_val >= optimal_threshold)
        if prob_val >= max(optimal_threshold + 0.15, 0.55):
            risk_level = "Alto Riesgo de Salida"
            badge_color = "#EF4444"
            hr_recommendation = (
                "Prioridad Alta: Programar entrevista preventiva de bienestar, revisar carga de "
                "horas extra, esquema de trabajo flexible/híbrido y plan de compensación."
            )
        elif prob_val >= optimal_threshold:
            risk_level = "Riesgo Moderado-Alto (Alerta)"
            badge_color = "#F59E0B"
            hr_recommendation = (
                "Alerta Preventiva: Evaluar movilidad interna, reconocimiento de carrera y "
                "acompañamiento del líder directo en el próximo ciclo 1-a-1."
            )
        else:
            risk_level = "Riesgo Bajo (Estable)"
            badge_color = "#10B981"
            hr_recommendation = (
                "Seguimiento Regular: Mantener condiciones actuales de desarrollo profesional "
                "y participación en proyectos estratégicos."
            )

        outputs.append(
            {
                "profile_id": p.get("profile_id", "SIM"),
                "profile_name": p.get("profile_name", "Perfil Simulado"),
                "narrative_description": p.get("narrative_description", ""),
                "attrition_probability": round(prob_val, 4),
                "attrition_probability_pct": round(prob_val * 100, 2),
                "predicted_class": pred_class,
                "predicted_label": "1 (Riesgo de Rotación)" if pred_class == 1 else "0 (Permanece)",
                "threshold_applied": round(float(optimal_threshold), 2),
                "risk_level": risk_level,
                "badge_color": badge_color,
                "key_drivers": _extract_risk_drivers(p["features"]),
                "hr_action": hr_recommendation,
                "ethical_disclaimer": (
                    "Aviso Ético Obligatorio: Esta estimación probabilística es una señal temprana para "
                    "diseñar programas de bienestar y retención; bajo ninguna circunstancia constituye "
                    "una decisión laboral automática ni justifica sanciones o desvinculaciones."
                ),
                "features": p["features"],
            }
        )

    return outputs


def run_full_hr_pipeline(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Etapa 16:
    Ejecuta las 16 etapas obligatorias del Grupo 08, guarda el Pipeline ganador y todos
    los modelos optimizados con joblib en `models/`, y serializa los resultados en JSON.
    Si los artefactos ya existen y `force_retrain=False`, los carga instantáneamente.
    """
    if (
        not force_retrain
        and HR_BEST_PIPELINE_PATH.exists()
        and HR_ALL_PIPELINES_PATH.exists()
        and HR_METRICS_JSON_PATH.exists()
    ):
        import json

        with open(HR_METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        best_bundle = joblib.load(HR_BEST_PIPELINE_PATH)
        all_bundles = joblib.load(HR_ALL_PIPELINES_PATH)
        return {
            **metrics_data,
            "winner_pipeline": best_bundle["pipeline"],
            "all_pipelines": all_bundles,
        }

    # Etapa 02 y 03: Carga y EDA
    df_raw = load_raw_hr()
    eda_summary = get_hr_eda_summary(df_raw)

    # Etapa 04 y 05: Limpieza de constantes/identificadores y separación X, y
    X, y, numeric_cols, categorical_cols = prepare_features_and_target(df_raw)

    # Etapa 06: Split Train/Test estratificado 80/20 reproducible
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Mismos 5 folds compartidos de StratifiedKFold
    shared_cv = get_shared_cv()

    # Etapa 09 y 10: Validación cruzada base
    baseline_cv_results = evaluate_baseline_cv(
        X_train, y_train, numeric_cols, categorical_cols, cv=shared_cv
    )

    # Etapa 11 y 12: Optimización de hiperparámetros (RandomizedSearchCV >= 20 configs) y selección de ganador
    winner_name, winner_pipeline, all_optimized_pipelines, tuning_summaries = (
        optimize_hyperparameters(
            X_train, y_train, numeric_cols, categorical_cols, cv=shared_cv
        )
    )

    optimal_threshold = tuning_summaries[winner_name]["oof_threshold_tuning"][
        "optimal_threshold"
    ]

    # Tabla comparativa obligatoria de la Sección 5 (CV inicial vs Mejor CV vs Hiperparámetros óptimos vs Tiempo)
    comparison_table: List[Dict[str, Any]] = []
    for model_name in ["DecisionTree", "RandomForest", "GradientBoosting"]:
        base_info = baseline_cv_results[model_name]
        tune_info = tuning_summaries[model_name]
        oof_info = tune_info["oof_threshold_tuning"]
        comparison_table.append(
            {
                "model_name": model_name,
                "is_winner": model_name == winner_name,
                "cv_initial_f1_mean": base_info["f1_mean"],
                "cv_initial_f1_std": base_info["f1_std"],
                "cv_initial_roc_auc": base_info["roc_auc_mean"],
                "cv_best_f1_mean": tune_info["best_cv_f1_mean"],
                "cv_best_f1_std": tune_info["best_cv_f1_std"],
                "oof_f1_optimal_threshold": oof_info["oof_f1_optimal"],
                "optimal_threshold": oof_info["optimal_threshold"],
                "cv_best_precision": oof_info["oof_precision_optimal"],
                "cv_best_recall": oof_info["oof_recall_optimal"],
                "cv_best_roc_auc": tune_info["best_cv_roc_auc_mean"],
                "cv_best_pr_auc": tune_info["best_cv_pr_auc_mean"],
                "best_params": tune_info["best_params"],
                "search_time_sec": tune_info["search_time_sec"],
            }
        )

    # Etapa 13: Evaluación final una sola vez en Test del Pipeline ganador
    test_eval = evaluate_on_test_once(
        winner_pipeline, X_test, y_test, optimal_threshold=optimal_threshold
    )

    # Etapa 14: Interpretación de variables importantes y estudio por antigüedad
    interpretability = analyze_interpretability_and_tenure(
        winner_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        optimal_threshold=optimal_threshold,
    )

    # Etapa 15: Predicción sobre 5 perfiles nuevos razonables y completamente descritos
    five_profiles = get_five_synthetic_profiles()
    five_predictions = predict_employee_profiles(
        winner_pipeline, five_profiles, optimal_threshold=optimal_threshold
    )

    # Rangos y valores por defecto de variables para el formulario interactivo en Streamlit
    feature_ui_specs: Dict[str, Any] = {}
    for col in numeric_cols:
        feature_ui_specs[col] = {
            "type": "numeric",
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
            "mean": round(float(X[col].mean()), 2),
        }
    for col in categorical_cols:
        feature_ui_specs[col] = {
            "type": "categorical",
            "options": sorted([str(v) for v in X[col].dropna().unique().tolist()]),
            "default": str(X[col].mode().iloc[0]),
        }

    # Etapa 16: Guardar artefactos con joblib y json
    best_bundle = {
        "winner_name": winner_name,
        "pipeline": winner_pipeline,
        "optimal_threshold": optimal_threshold,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_ui_specs": feature_ui_specs,
    }
    joblib.dump(best_bundle, HR_BEST_PIPELINE_PATH)
    joblib.dump(all_optimized_pipelines, HR_ALL_PIPELINES_PATH)

    json_payload: Dict[str, Any] = {
        "winner_name": winner_name,
        "optimal_threshold": optimal_threshold,
        "split_info": {
            "total_rows": int(len(X)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_positive_rate_pct": round(float(y_train.mean() * 100), 2),
            "test_positive_rate_pct": round(float(y_test.mean() * 100), 2),
            "cv_folds": 5,
            "random_state": 42,
        },
        "eda_summary": eda_summary,
        "baseline_cv": baseline_cv_results,
        "tuning_summaries": tuning_summaries,
        "comparison_table": comparison_table,
        "test_evaluation": test_eval,
        "interpretability": interpretability,
        "five_synthetic_predictions": five_predictions,
        "feature_ui_specs": feature_ui_specs,
    }
    save_metrics_json(json_payload, HR_METRICS_JSON_PATH)

    return {
        **json_payload,
        "winner_pipeline": winner_pipeline,
        "all_pipelines": all_optimized_pipelines,
    }
