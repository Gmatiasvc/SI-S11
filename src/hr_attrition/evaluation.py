"""Etapas 13 y 14 del Grupo 08: Evaluación final única en Test, importancia de variables y análisis por antigüedad."""
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def evaluate_on_test_once(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Etapa 13:
    Evalúa una sola vez en Test el Pipeline ganador usando tanto el umbral estándar (0.50)
    como el umbral óptimo aprendido exclusivamente en validación cruzada (OOF).
    """
    y_true = y_test.to_numpy()
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    preds_default = (y_proba >= 0.50).astype(int)
    preds_tuned = (y_proba >= optimal_threshold).astype(int)

    cm_default = confusion_matrix(y_true, preds_default)
    cm_tuned = confusion_matrix(y_true, preds_tuned)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, y_proba)

    tn, fp, fn, tp = [int(v) for v in cm_tuned.ravel()]

    return {
        "n_test_samples": int(len(y_true)),
        "n_test_positive": int(np.sum(y_true == 1)),
        "n_test_negative": int(np.sum(y_true == 0)),
        "optimal_threshold_used": round(float(optimal_threshold), 2),
        # Métricas con umbral óptimo de validación
        "f1_tuned": round(float(f1_score(y_true, preds_tuned, zero_division=0)), 4),
        "precision_tuned": round(float(precision_score(y_true, preds_tuned, zero_division=0)), 4),
        "recall_tuned": round(float(recall_score(y_true, preds_tuned, zero_division=0)), 4),
        "accuracy_tuned": round(float(accuracy_score(y_true, preds_tuned)), 4),
        # Métricas con umbral por defecto 0.50
        "f1_default_050": round(float(f1_score(y_true, preds_default, zero_division=0)), 4),
        "precision_default_050": round(float(precision_score(y_true, preds_default, zero_division=0)), 4),
        "recall_default_050": round(float(recall_score(y_true, preds_default, zero_division=0)), 4),
        "accuracy_default_050": round(float(accuracy_score(y_true, preds_default)), 4),
        # Métricas independientes de umbral
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        # Matrices de confusión
        "confusion_matrix_tuned": cm_tuned.tolist(),
        "confusion_matrix_default_050": cm_default.tolist(),
        "error_breakdown": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "false_positive_rate": round(float(fp / max(tn + fp, 1)), 4),
            "false_negative_rate": round(float(fn / max(tp + fn, 1)), 4),
        },
        "roc_curve": {
            "fpr": np.round(fpr, 4).tolist(),
            "tpr": np.round(tpr, 4).tolist(),
        },
        "pr_curve": {
            "precision": np.round(prec_curve, 4).tolist(),
            "recall": np.round(rec_curve, 4).tolist(),
        },
    }


def analyze_interpretability_and_tenure(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Etapa 14 y Análisis específico obligatorio (Sección 5 del PDF):
    1. Importancia de variables por impureza (MDI) sobre las características codificadas.
    2. Importancia por permutación (Permutation Importance) sobre las variables originales de entrada.
    3. Estudio obligatorio de diferencias de rotación y desempeño (Recall, Precision, F1) por tramos de antigüedad (YearsAtCompany).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names_out = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    mdi_records: List[Dict[str, Any]] = []
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx[:20]:
            mdi_records.append(
                {
                    "feature": feature_names_out[idx],
                    "importance_mdi": round(float(importances[idx]), 4),
                }
            )

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_idx = np.argsort(perm.importances_mean)[::-1]
    perm_records: List[Dict[str, Any]] = []
    cols_list = list(X_test.columns)
    for idx in perm_idx[:15]:
        perm_records.append(
            {
                "feature": cols_list[idx],
                "importance_mean": round(float(perm.importances_mean[idx]), 4),
                "importance_std": round(float(perm.importances_std[idx]), 4),
            }
        )

    # Análisis obligatorio por tramos de antigüedad (YearsAtCompany) en Test y en el dataset global
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= optimal_threshold).astype(int)

    df_eval = X_test.copy()
    df_eval["y_true"] = y_test.values
    df_eval["y_pred"] = y_pred_test
    df_eval["y_proba"] = y_proba_test

    bins = [-1, 2, 5, 10, 100]
    labels_tenure = [
        "0-2 años (Recientes)",
        "3-5 años (Consolidación)",
        "6-10 años (Estables)",
        ">10 años (Veteranos)",
    ]
    df_eval["TenureBand"] = pd.cut(
        df_eval["YearsAtCompany"], bins=bins, labels=labels_tenure
    )

    tenure_analysis: List[Dict[str, Any]] = []
    for band in labels_tenure:
        sub = df_eval[df_eval["TenureBand"] == band]
        n_sub = int(len(sub))
        if n_sub == 0:
            continue
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        actual_rate = float(np.mean(yt) * 100)
        pred_prob_mean = float(np.mean(sub["y_proba"]) * 100)
        tenure_analysis.append(
            {
                "tenure_band": band,
                "n_employees_test": n_sub,
                "actual_attrition_count": int(np.sum(yt == 1)),
                "actual_attrition_rate_pct": round(actual_rate, 2),
                "mean_predicted_risk_pct": round(pred_prob_mean, 2),
                "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
                "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
                "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            }
        )

    # Tasa histórica de rotación por antigüedad en todo el conjunto (Train + Test)
    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)
    tenure_bands_all = pd.cut(X_all["YearsAtCompany"], bins=bins, labels=labels_tenure)
    full_tenure_rates: List[Dict[str, Any]] = []
    for band in labels_tenure:
        mask = tenure_bands_all == band
        n_tot = int(np.sum(mask))
        n_pos = int(np.sum(y_all[mask] == 1))
        full_tenure_rates.append(
            {
                "tenure_band": band,
                "total_employees": n_tot,
                "attrition_count": n_pos,
                "attrition_rate_pct": round(float(n_pos / max(n_tot, 1) * 100), 2),
            }
        )

    return {
        "mdi_top_features": mdi_records,
        "permutation_top_features": perm_records,
        "tenure_band_test_metrics": tenure_analysis,
        "tenure_band_population_rates": full_tenure_rates,
    }
