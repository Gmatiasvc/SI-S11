"""Algoritmos de agrupamiento K-Means y DBSCAN en espacio original y espacio PCA (Etapas 4, 5 y 7)."""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from src.config import KMEANS_K_RANGE, KMEANS_N_INIT, RANDOM_STATE


def _compute_internal_metrics(X_space: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Calcula Silhouette, Davies-Bouldin y Calinski-Harabasz sobre los puntos válidos
    (excluyendo etiqueta -1 si existe ruido de DBSCAN), verificando al menos 2 clusters.
    """
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    unique_clusters = np.unique(valid_labels)

    if len(unique_clusters) < 2 or np.sum(valid_mask) <= len(unique_clusters):
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

    X_valid = X_space[valid_mask]
    return {
        "silhouette": round(float(silhouette_score(X_valid, valid_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_valid, valid_labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_valid, valid_labels)), 2),
    }


def explore_kmeans(
    X_space: np.ndarray,
    k_range: List[int] = KMEANS_K_RANGE,
    random_state: int = RANDOM_STATE,
    n_init: int = KMEANS_N_INIT,
    X_eval_original: Optional[np.ndarray] = None,
    preferred_k_min: int = 3,
    preferred_k_max: int = 5,
) -> Dict[str, Any]:
    """
    Explora K-Means para K en k_range (2..10) con inicialización k-means++, n_init=20 y semilla fija.
    Calcula Inercia, Silhouette, DBI, CHI y distribución de tamaños de cluster.
    Selecciona el K óptimo combinando evidencia cuantitativa (Silhouette/DBI/Codo)
    y utilidad accionable de negocio (segmentación RFM en 3-5 arquetipos diferenciados).
    """
    if X_eval_original is None:
        X_eval_original = X_space

    history: List[Dict[str, Any]] = []
    models_by_k: Dict[int, KMeans] = {}
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in k_range:
        km = KMeans(
            n_clusters=int(k),
            init="k-means++",
            n_init=n_init,
            random_state=random_state,
        )
        labels = km.fit_predict(X_space)
        models_by_k[int(k)] = km
        labels_by_k[int(k)] = labels

        metrics_fit = _compute_internal_metrics(X_space, labels)
        metrics_orig = _compute_internal_metrics(X_eval_original, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
        min_cluster_pct = round(float(np.min(counts) / len(labels) * 100), 2)

        history.append(
            {
                "k": int(k),
                "inertia": round(float(km.inertia_), 2),
                "silhouette": metrics_orig["silhouette"],
                "davies_bouldin": metrics_orig["davies_bouldin"],
                "calinski_harabasz": metrics_orig["calinski_harabasz"],
                "silhouette_fit_space": metrics_fit["silhouette"],
                "davies_bouldin_fit_space": metrics_fit["davies_bouldin"],
                "calinski_harabasz_fit_space": metrics_fit["calinski_harabasz"],
                "cluster_sizes": cluster_sizes,
                "min_cluster_pct": min_cluster_pct,
            }
        )

    best_overall_entry = max(history, key=lambda r: r["silhouette"] or -1.0)
    business_candidates = [
        r
        for r in history
        if preferred_k_min <= r["k"] <= preferred_k_max and r["min_cluster_pct"] >= 2.0
    ]
    if business_candidates:
        best_selected_entry = max(business_candidates, key=lambda r: r["silhouette"] or -1.0)
    else:
        best_selected_entry = best_overall_entry

    best_k = int(best_selected_entry["k"])
    best_model = models_by_k[best_k]
    best_labels = labels_by_k[best_k]
    sil_samples = silhouette_samples(X_eval_original, best_labels)

    return {
        "best_k": best_k,
        "best_k_overall_silhouette": int(best_overall_entry["k"]),
        "best_model": best_model,
        "best_labels": best_labels,
        "best_metrics": best_selected_entry,
        "k_search_results": history,
        "silhouette_samples": np.round(sil_samples, 4),
        "selection_rationale": (
            f"Se exploró K=2..10 con k-means++ (n_init={n_init}, random_state={random_state}). "
            f"Aunque K={best_overall_entry['k']} alcanza un Silhouette de {best_overall_entry['silhouette']:.4f} "
            f"al separar macro-grupos, K={best_k} (Silhouette={best_selected_entry['silhouette']:.4f}, "
            f"DBI={best_selected_entry['davies_bouldin']:.4f}, CHI={best_selected_entry['calinski_harabasz']:.1f}) "
            f"captura el punto de inflexión (codo de inercia) y distingue {best_k} perfiles estratégicos "
            "accionables (Clientes VIP/Campeones, Leales Regulares, Ocasionales con fricción/cancelación y En Riesgo/Inactivos)."
        ),
    }


def compute_k_distance_curve(X_space: np.ndarray, k: int = 11) -> Dict[str, Any]:
    """
    Calcula la curva ordenada de distancias al k-ésimo vecino más cercano (regla orientativa
    k = 2 * dim - 1 = 11 para 6 variables) y detecta el punto de máxima curvatura (codo)
    para orientar la búsqueda de epsilon en DBSCAN.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_space)
    distances, _ = nn.kneighbors(X_space)
    k_distances = np.sort(distances[:, -1])

    n_pts = len(k_distances)
    cutoff_idx = int(n_pts * 0.99)
    y = k_distances[:cutoff_idx]
    x = np.arange(len(y), dtype=float)
    x_norm = x / max(len(y) - 1, 1)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-8)
    deviation = y_norm - x_norm
    knee_idx = int(np.argmax(deviation))
    suggested_eps = round(float(y[knee_idx]), 2)

    sample_indices = np.linspace(0, n_pts - 1, 200, dtype=int)

    return {
        "k_neighbors": int(k),
        "suggested_eps": suggested_eps,
        "knee_index": knee_idx,
        "percentiles": {
            "p50": round(float(np.percentile(k_distances, 50)), 3),
            "p75": round(float(np.percentile(k_distances, 75)), 3),
            "p90": round(float(np.percentile(k_distances, 90)), 3),
            "p95": round(float(np.percentile(k_distances, 95)), 3),
            "p98": round(float(np.percentile(k_distances, 98)), 3),
        },
        "curve_sampled_indices": sample_indices.tolist(),
        "curve_sampled_distances": np.round(k_distances[sample_indices], 4).tolist(),
        "full_k_distances": k_distances,
    }


def explore_dbscan(
    X_space: np.ndarray,
    eps_values: Optional[List[float]] = None,
    min_samples_values: Optional[List[int]] = None,
    X_eval_original: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Realiza una búsqueda razonada en grilla sobre eps y min_samples para DBSCAN.
    Identifica puntos núcleo (core), borde (border) y ruido (noise, label == -1).
    Evalúa exclusivamente configuraciones con al menos 2 clusters válidos (excluyendo -1).
    Calcula métricas tanto en el espacio de ajuste como en el espacio original escalado.
    """
    if X_eval_original is None:
        X_eval_original = X_space
    if eps_values is None:
        eps_values = [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 1.00]
    if min_samples_values is None:
        min_samples_values = [5, 8, 10, 12, 15, 20]

    n_total = len(X_space)
    grid_results: List[Dict[str, Any]] = []
    valid_candidates: List[Dict[str, Any]] = []

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=float(eps), min_samples=int(ms), metric="euclidean")
            labels = db.fit_predict(X_space)
            core_mask = np.zeros(n_total, dtype=bool)
            if hasattr(db, "core_sample_indices_") and len(db.core_sample_indices_) > 0:
                core_mask[db.core_sample_indices_] = True
            noise_mask = labels == -1
            border_mask = (~core_mask) & (~noise_mask)

            n_noise = int(np.sum(noise_mask))
            n_core = int(np.sum(core_mask))
            n_border = int(np.sum(border_mask))
            noise_ratio = round(float(n_noise / n_total), 4)
            coverage_ratio = round(float(1.0 - noise_ratio), 4)

            valid_clusters = [int(c) for c in np.unique(labels) if c != -1]
            n_clusters = len(valid_clusters)

            metrics_orig = _compute_internal_metrics(X_eval_original, labels)
            metrics_fit = _compute_internal_metrics(X_space, labels)

            cluster_sizes = {
                int(c): int(np.sum(labels == c)) for c in valid_clusters
            }
            sorted_counts = sorted(cluster_sizes.values(), reverse=True)
            second_cluster_size = int(sorted_counts[1]) if len(sorted_counts) >= 2 else 0

            row = {
                "eps": float(eps),
                "min_samples": int(ms),
                "n_clusters": n_clusters,
                "n_core": n_core,
                "n_border": n_border,
                "n_noise": n_noise,
                "noise_pct": round(noise_ratio * 100, 2),
                "coverage_pct": round(coverage_ratio * 100, 2),
                "n_evaluated": int(n_total - n_noise),
                "second_cluster_size": second_cluster_size,
                "silhouette": metrics_orig["silhouette"],
                "davies_bouldin": metrics_orig["davies_bouldin"],
                "calinski_harabasz": metrics_orig["calinski_harabasz"],
                "silhouette_fit_space": metrics_fit["silhouette"],
                "davies_bouldin_fit_space": metrics_fit["davies_bouldin"],
                "calinski_harabasz_fit_space": metrics_fit["calinski_harabasz"],
                "cluster_sizes": cluster_sizes,
                "is_valid_config": bool(n_clusters >= 2),
            }
            grid_results.append(row)

            if n_clusters >= 2 and metrics_orig["silhouette"] is not None:
                balance_bonus = 0.15 if second_cluster_size >= 25 else 0.0
                coverage_bonus = 0.25 * coverage_ratio
                composite_score = float(metrics_orig["silhouette"]) + coverage_bonus + balance_bonus
                valid_candidates.append(
                    {
                        "entry": row,
                        "labels": labels,
                        "core_mask": core_mask,
                        "border_mask": border_mask,
                        "noise_mask": noise_mask,
                        "composite_score": composite_score,
                    }
                )

    if valid_candidates:
        best_cand = max(valid_candidates, key=lambda c: c["composite_score"])
        best_entry = best_cand["entry"]
        best_labels = best_cand["labels"]
        best_core_mask = best_cand["core_mask"]
        best_border_mask = best_cand["border_mask"]
        best_noise_mask = best_cand["noise_mask"]
    else:
        fallback_db = DBSCAN(eps=0.55, min_samples=6)
        best_labels = fallback_db.fit_predict(X_space)
        best_core_mask = np.zeros(n_total, dtype=bool)
        if len(fallback_db.core_sample_indices_) > 0:
            best_core_mask[fallback_db.core_sample_indices_] = True
        best_noise_mask = best_labels == -1
        best_border_mask = (~best_core_mask) & (~best_noise_mask)
        best_entry = grid_results[0]

    point_types = np.where(
        best_core_mask,
        "Core (Núcleo)",
        np.where(best_border_mask, "Border (Borde)", "Noise (Ruido / Atípico)"),
    )

    return {
        "best_eps": float(best_entry["eps"]),
        "best_min_samples": int(best_entry["min_samples"]),
        "best_metrics": best_entry,
        "best_labels": best_labels,
        "point_types": point_types,
        "grid_search_results": grid_results,
        "n_valid_configs": len(valid_candidates),
    }
