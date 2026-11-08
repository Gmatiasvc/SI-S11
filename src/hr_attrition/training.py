"""Etapas 09, 10, 11 y 12 del Grupo 08: Validación cruzada base, optimización de hiperparámetros y umbral OOF."""
import time
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, N_ITER_SEARCH, RANDOM_STATE
from src.hr_attrition.models import get_base_pipelines, get_hyperparameter_grids


def get_shared_cv(
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Devuelve el objeto StratifiedKFold compartido exactamente por los 3 modelos."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_baseline_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cv: StratifiedKFold | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Etapas 09 y 10:
    Evalúa las 3 configuraciones iniciales en los mismos 5 folds de Train.
    Compara media, desviación estándar, complejidad y tiempos de cómputo.
    """
    if cv is None:
        cv = get_shared_cv()

    pipelines = get_base_pipelines(numeric_cols, categorical_cols)
    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, pipe in pipelines.items():
        t0 = time.perf_counter()
        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        elapsed = time.perf_counter() - t0

        results[name] = {
            "model_name": name,
            "f1_mean": round(float(np.mean(cv_res["test_f1"])), 4),
            "f1_std": round(float(np.std(cv_res["test_f1"])), 4),
            "f1_folds": [round(float(v), 4) for v in cv_res["test_f1"]],
            "train_f1_mean": round(float(np.mean(cv_res["train_f1"])), 4),
            "precision_mean": round(float(np.mean(cv_res["test_precision"])), 4),
            "precision_std": round(float(np.std(cv_res["test_precision"])), 4),
            "recall_mean": round(float(np.mean(cv_res["test_recall"])), 4),
            "recall_std": round(float(np.std(cv_res["test_recall"])), 4),
            "accuracy_mean": round(float(np.mean(cv_res["test_accuracy"])), 4),
            "roc_auc_mean": round(float(np.mean(cv_res["test_roc_auc"])), 4),
            "pr_auc_mean": round(float(np.mean(cv_res["test_pr_auc"])), 4),
            "fit_time_total_sec": round(float(elapsed), 3),
        }
    return results


def find_optimal_threshold_oof(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    """
    Calcula probabilidades out-of-fold (OOF) exclusivamente sobre Train usando los mismos folds
    y encuentra el umbral tau en [0.20, 0.80] que maximiza el F1-Score de la clase positiva (Attrition=1).
    Jamás utiliza el conjunto Test.
    """
    oof_proba = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.linspace(0.20, 0.80, 61)
    curve: list[Dict[str, float]] = []
    best_tau = 0.50
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    y_true = y_train.to_numpy()
    for tau in thresholds:
        preds = (oof_proba >= tau).astype(int)
        f1_val = float(f1_score(y_true, preds, zero_division=0))
        prec_val = float(precision_score(y_true, preds, zero_division=0))
        rec_val = float(recall_score(y_true, preds, zero_division=0))
        curve.append(
            {
                "threshold": round(float(tau), 2),
                "f1": round(f1_val, 4),
                "precision": round(prec_val, 4),
                "recall": round(rec_val, 4),
            }
        )
        if f1_val > best_f1:
            best_f1 = f1_val
            best_tau = float(tau)
            best_prec = prec_val
            best_rec = rec_val

    default_preds = (oof_proba >= 0.50).astype(int)
    f1_at_050 = float(f1_score(y_true, default_preds, zero_division=0))

    return {
        "optimal_threshold": round(best_tau, 2),
        "oof_f1_optimal": round(best_f1, 4),
        "oof_precision_optimal": round(best_prec, 4),
        "oof_recall_optimal": round(best_rec, 4),
        "oof_f1_default_050": round(f1_at_050, 4),
        "threshold_curve": curve,
    }


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cv: StratifiedKFold | None = None,
    n_iter: int = N_ITER_SEARCH,
) -> Tuple[str, Pipeline, Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """
    Etapas 11 y 12:
    Optimiza hiperparámetros con RandomizedSearchCV (>= 20 configuraciones por modelo)
    dentro de Train usando la misma métrica principal (F1) y los mismos 5 folds.
    Selecciona el modelo ganador por métrica de validación cruzada y realiza refit en todo Train.
    """
    if cv is None:
        cv = get_shared_cv()

    base_pipelines = get_base_pipelines(numeric_cols, categorical_cols)
    param_grids = get_hyperparameter_grids()

    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    best_estimators: Dict[str, Pipeline] = {}
    tuning_summaries: Dict[str, Dict[str, Any]] = {}

    for name, pipe in base_pipelines.items():
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            scoring=scoring,
            refit="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            return_train_score=True,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        best_idx = search.best_index_
        cv_res = search.cv_results_

        best_pipe = search.best_estimator_
        best_estimators[name] = best_pipe

        oof_threshold_info = find_optimal_threshold_oof(best_pipe, X_train, y_train, cv)

        fold_f1s = [
            round(float(cv_res[f"split{k}_test_f1"][best_idx]), 4)
            for k in range(cv.get_n_splits())
        ]

        top_indices = np.argsort(cv_res["rank_test_f1"])[:10]
        top_candidates = []
        for idx in top_indices:
            clean_params = {
                k.replace("classifier__", ""): v
                for k, v in cv_res["params"][idx].items()
            }
            top_candidates.append(
                {
                    "rank": int(cv_res["rank_test_f1"][idx]),
                    "f1_mean": round(float(cv_res["mean_test_f1"][idx]), 4),
                    "f1_std": round(float(cv_res["std_test_f1"][idx]), 4),
                    "precision_mean": round(float(cv_res["mean_test_precision"][idx]), 4),
                    "recall_mean": round(float(cv_res["mean_test_recall"][idx]), 4),
                    "roc_auc_mean": round(float(cv_res["mean_test_roc_auc"][idx]), 4),
                    "params": clean_params,
                }
            )

        clean_best_params = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }

        tuning_summaries[name] = {
            "model_name": name,
            "n_configs_explored": int(len(cv_res["params"])),
            "best_cv_f1_mean": round(float(search.best_score_), 4),
            "best_cv_f1_std": round(float(cv_res["std_test_f1"][best_idx]), 4),
            "best_cv_f1_folds": fold_f1s,
            "best_cv_train_f1_mean": round(float(cv_res["mean_train_f1"][best_idx]), 4),
            "best_cv_precision_mean": round(float(cv_res["mean_test_precision"][best_idx]), 4),
            "best_cv_recall_mean": round(float(cv_res["mean_test_recall"][best_idx]), 4),
            "best_cv_accuracy_mean": round(float(cv_res["mean_test_accuracy"][best_idx]), 4),
            "best_cv_roc_auc_mean": round(float(cv_res["mean_test_roc_auc"][best_idx]), 4),
            "best_cv_pr_auc_mean": round(float(cv_res["mean_test_pr_auc"][best_idx]), 4),
            "best_params": clean_best_params,
            "search_time_sec": round(float(elapsed), 3),
            "oof_threshold_tuning": oof_threshold_info,
            "top_candidates": top_candidates,
        }

    # Selección del ganador según F1 de validación cruzada (considerando F1 OOF con umbral optimizado en CV y F1 CV estándar)
    winner_name = max(
        tuning_summaries.keys(),
        key=lambda k: (
            tuning_summaries[k]["oof_threshold_tuning"]["oof_f1_optimal"],
            tuning_summaries[k]["best_cv_f1_mean"],
        ),
    )
    winner_pipeline = best_estimators[winner_name]

    return winner_name, winner_pipeline, best_estimators, tuning_summaries
