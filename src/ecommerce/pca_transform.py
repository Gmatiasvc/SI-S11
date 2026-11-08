"""Reducción de dimensionalidad con PCA: Scree Plot, varianza 90-95%, cargas y error de reconstrucción (Etapa 6)."""
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from src.config import PCA_VARIANCE_THRESHOLD, RANDOM_STATE


def run_pca_analysis(
    X_scaled: pd.DataFrame,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Ajusta PCA completo sobre X_scaled, determina el número óptimo de componentes d
    para alcanzar el umbral de varianza explicada acumulada (90%-95%), calcula las
    cargas (loadings), las coordenadas proyectadas y el error cuadrático medio de
    reconstrucción por cantidad de componentes.
    """
    feature_names: List[str] = list(X_scaled.columns)
    X_arr = X_scaled.to_numpy(dtype=float)
    n_samples, n_features = X_arr.shape

    pca_full = PCA(n_components=n_features, random_state=random_state)
    X_pca_full = pca_full.fit_transform(X_arr)

    explained_ratio = pca_full.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)
    eigenvalues = pca_full.explained_variance_

    # Seleccionar menor cantidad de componentes que alcance variance_threshold (ej. >= 90%)
    idx_candidates = np.where(cumulative_ratio >= variance_threshold)[0]
    n_selected = int(idx_candidates[0] + 1) if len(idx_candidates) > 0 else n_features
    n_selected = max(2, min(n_selected, n_features - 1))

    # Ajustar PCA con n_selected componentes para obtener proyección y reconstrucción
    pca_selected = PCA(n_components=n_selected, random_state=random_state)
    X_pca_selected = pca_selected.fit_transform(X_arr)
    X_reconstructed = pca_selected.inverse_transform(X_pca_selected)
    reconstruction_mse_selected = float(mean_squared_error(X_arr, X_reconstructed))
    reconstruction_rmse_selected = float(np.sqrt(reconstruction_mse_selected))

    # Curva de error de reconstrucción para d = 1..n_features
    reconstruction_curve: List[Dict[str, Any]] = []
    for d in range(1, n_features + 1):
        pca_d = PCA(n_components=d, random_state=random_state)
        proj_d = pca_d.fit_transform(X_arr)
        rec_d = pca_d.inverse_transform(proj_d)
        mse_d = float(mean_squared_error(X_arr, rec_d))
        reconstruction_curve.append(
            {
                "n_components": d,
                "PC": f"PC{d}",
                "eigenvalue": round(float(eigenvalues[d - 1]), 4),
                "explained_variance_ratio": round(float(explained_ratio[d - 1]), 4),
                "explained_variance_pct": round(float(explained_ratio[d - 1] * 100), 2),
                "cumulative_variance_ratio": round(float(cumulative_ratio[d - 1]), 4),
                "cumulative_variance_pct": round(float(cumulative_ratio[d - 1] * 100), 2),
                "reconstruction_mse": round(mse_d, 5),
                "information_loss_pct": round(float((1.0 - cumulative_ratio[d - 1]) * 100), 2),
            }
        )

    pc_names = [f"PC{i+1}" for i in range(n_features)]
    loadings_df = pd.DataFrame(
        pca_full.components_.T,
        index=feature_names,
        columns=pc_names,
    ).round(4)

    X_pca_df = pd.DataFrame(
        X_pca_selected,
        index=X_scaled.index,
        columns=pc_names[:n_selected],
    )

    # Interpretación automática de los primeros componentes según cargas mayores
    component_interpretations: Dict[str, str] = {}
    for i in range(min(n_selected, 4)):
        pc_col = f"PC{i+1}"
        sorted_loads = loadings_df[pc_col].reindex(
            loadings_df[pc_col].abs().sort_values(ascending=False).index
        )
        top_vars = [
            f"{var} ({val:+.2f})" for var, val in sorted_loads.head(3).items()
        ]
        var_pct = explained_ratio[i] * 100
        component_interpretations[pc_col] = (
            f"{pc_col} ({var_pct:.1f}% var.): Dominado por {', '.join(top_vars)}."
        )

    return {
        "pca_model": pca_selected,
        "pca_full_model": pca_full,
        "n_features_original": n_features,
        "n_components_selected": n_selected,
        "variance_threshold": variance_threshold,
        "cumulative_variance_selected": round(float(cumulative_ratio[n_selected - 1]), 4),
        "cumulative_variance_selected_pct": round(float(cumulative_ratio[n_selected - 1] * 100), 2),
        "reconstruction_mse": round(reconstruction_mse_selected, 5),
        "reconstruction_rmse": round(reconstruction_rmse_selected, 5),
        "scree_and_reconstruction": reconstruction_curve,
        "loadings": loadings_df.to_dict(),
        "loadings_records": loadings_df.reset_index().rename(columns={"index": "Feature"}).to_dict(orient="records"),
        "component_interpretations": component_interpretations,
        "X_pca_df": X_pca_df,
        "X_pca_2d": pd.DataFrame(
            X_pca_full[:, :2], index=X_scaled.index, columns=["PC1", "PC2"]
        ),
        "X_pca_3d": pd.DataFrame(
            X_pca_full[:, :3], index=X_scaled.index, columns=["PC1", "PC2", "PC3"]
        ),
    }
