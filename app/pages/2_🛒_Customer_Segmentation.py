"""Página 2: Módulo 1 — Segmentación Estratégica de Clientes de Comercio Electrónico (Trabajo 01)."""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ECOMMERCE_PIPELINE_PATH
from src.ecommerce.business_insights import run_full_ecommerce_pipeline
from src.ecommerce.feature_engineering import FEATURE_COLUMNS
from src.utils.academic_content import (
    ECOMMERCE_ACADEMIC_CONTENT,
    format_ecommerce_defense_answers,
)

_css_file = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
if _css_file.exists():
    st.markdown(f"<style>{_css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)



@st.cache_resource(show_spinner="Cargando motor no supervisado de Comercio Electrónico...")
def load_ecommerce_bundle():
    results = run_full_ecommerce_pipeline(force_retrain=False)
    artifact = joblib.load(ECOMMERCE_PIPELINE_PATH)
    return results, artifact


res, artifact = load_ecommerce_bundle()
df_cust: pd.DataFrame = res["df_customers"]
prep = res["preprocessing"]
eda = res["eda_summary"]
pca_info = res["pca_summary"]
km_info = res["kmeans_original"]
db_info = res["dbscan_original"]
pca_km_info = res["pca_kmeans"]
pca_db_info = res["pca_dbscan"]
fair_matrix = res["fair_comparison_matrix"]
stability = res["stability_analysis"]
profiles = res["cluster_profiles"]
anomalies = res["anomalies_analysis"]
strategies = res["commercial_strategies"]

# Cabecera Hero
st.markdown(
    """
    <div class="hero-banner">
        <span class="badge-pill">Trabajo Aplicativo 01 · Aprendizaje No Supervisado</span>
        <h1>🛒 Segmentación de Clientes de Comercio Electrónico (RFM + K-Means + DBSCAN + PCA)</h1>
        <p>
            Descubrimiento de estructuras latentes, arquetipos de consumo y detección de clientes atípicos sobre
            <b>5,878 clientes únicos</b> (agregados a partir de 1,067,371 transacciones de <i>UCI Online Retail II</i>),
            evaluando rigurosamente las 10 etapas obligatorias de la guía experimental.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pestañas organizadas siguiendo exactamente las 10 etapas y evidencias de la rúbrica
tabs = st.tabs(
    [
        "1️⃣ Problema, Datos y EDA (Etapas 1–3)",
        "2️⃣ K-Means y DBSCAN (Etapas 4–5)",
        "3️⃣ PCA y Modelos Combinados (Etapas 6–7)",
        "4️⃣ Comparación Justa y Estabilidad (Etapas 8–10)",
        "5️⃣ Perfiles, Atípicos y Estrategias Comerciales",
        "6️⃣ Simulador RFM y Sustentación (8 Preguntas)",
    ]
)

# ==============================================================================
# TAB 1: ETAPAS 1, 2 Y 3 (PROBLEMA, EDA Y PREPROCESAMIENTO)
# ==============================================================================
with tabs[0]:
    st.subheader("Etapa 1: Comprensión del Problema y Unidad de Análisis")
    prob = ECOMMERCE_ACADEMIC_CONTENT["problem_definition"]
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>🎯 Objetivo de Análisis vs. Predicción Futura</h4>
                <p>{prob['descriptive_vs_predictive']['objetivo_descriptivo_no_supervisado']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>💡 Diferencia con Aprendizaje Supervisado y Utilidad CRM</h4>
                <p>{prob['descriptive_vs_predictive']['diferencia_con_prediccion_supervisada']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📖 Ver Registro Completo de Procedencia del Dataset, Diccionario y Fórmulas Matemáticas (LaTeX)", expanded=True):
        prov = ECOMMERCE_ACADEMIC_CONTENT["dataset_provenance"]
        st.markdown(
            f"- **Dataset:** `{prov['name']}` | **Enlace Exacto:** [{prov['kaggle_url']}]({prov['kaggle_url']}) / [{prov['uci_url']}]({prov['uci_url']})\n"
            f"- **Autor y Licencia:** {prov['author']} — **{prov['license']}** (Fecha de acceso: {prov['access_date']})\n"
            f"- **Unidad de Análisis:** {prov['unit_of_analysis']} — **{prep['n_unique_customers']:,} clientes únicos** con compras válidas (de {prep['n_raw_rows']:,} filas brutas)."
        )
        st.markdown("##### Fórmulas Matemáticas de las 6 Variables Cuantitativas Estudiadas")
        st.latex(
            r"R_i = (T_{\text{ref}} - \max_{t \in \mathcal{V}_i} t)_{\text{días}}, \quad "
            r"F_i = |\mathcal{I}_i^{\text{válidas}}|, \quad "
            r"M_i = \sum_{j \in \mathcal{V}_i} Q_{ij} \times P_{ij}"
        )
        st.latex(
            r"\text{AvgTicket}_i = \frac{M_i}{F_i}, \quad "
            r"\text{UniqueCategories}_i = |\text{SKUs}_i|, \quad "
            r"\text{CancellationRate}_i = \max\left(\frac{|\mathcal{I}_i^{\text{cancel}}|}{|\mathcal{I}_i^{\text{total}}|}, \frac{L_i^{\text{cancel}}}{L_i^{\text{total}}}\right) \in [0, 1]"
        )
        feat_dict_rows = [
            {
                "Variable": k,
                "Nombre Comercial": v["nombre_esp"],
                "Fórmula de Agregación": v["formula"],
                "Justificación en el Problema": v["justificacion"],
            }
            for k, v in prep["feature_dictionary"].items()
        ]
        st.dataframe(pd.DataFrame(feat_dict_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Etapa 2: Exploración de Datos (EDA) y Cuatro Hallazgos Clave")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Filas Brutas CSV", f"{eda['shape']['rows']:,}", f"{eda['shape']['cols']} columnas")
    k2.metric("Faltantes en Customer ID", f"{eda['missing_counts']['Customer ID']:,}", f"{eda['missing_pct']['Customer ID']:.2f}% (Excluidos)")
    k3.metric("Facturas Canceladas ('C')", f"{eda['n_cancelled']:,}", "Usadas en CancellationRate")
    k4.metric("Duplicados Exactos", f"{eda['n_duplicates']:,}", "Depurados en agregación")
    k5.metric("Clientes Válidos Finales", f"{prep['n_unique_customers']:,}", f"{eda['n_countries']} países")

    st.markdown("#### 🔍 Interpretación Obligatoria de Hallazgos Exploratorios")
    f_col1, f_col2 = st.columns(2)
    for idx, finding in enumerate(eda["findings"]):
        target_col = f_col1 if idx % 2 == 0 else f_col2
        with target_col:
            st.info(finding)

    st.markdown("---")
    st.subheader("Etapa 3: Preprocesamiento, Corrección de Asimetría (`log1p`) y Escalado Compartido")
    st.markdown(
        "Para garantizar que un puñado de compradores mayoristas (*whales*) no distorsione las distancias euclídeas, "
        "se aplicó **winsorización suave al percentil 99.5%** seguida de transformación **`np.log1p`** y **`StandardScaler`**, "
        "manteniendo una única matriz escalada compartida (`X_scaled`) para evaluar las 4 alternativas justamente."
    )

    skew_df = pd.DataFrame(
        [
            {
                "Variable": col,
                "Media Original": round(df_cust[col].mean(), 2),
                "Mediana Original": round(df_cust[col].median(), 2),
                "Tope Winsor (p99.5)": prep["winsor_caps"][col],
                "Asimetría (Skewness) Antes": prep["skewness_before_log1p"][col],
                "Asimetría (Skewness) Después de log1p": prep["skewness_after_log1p"][col],
                "Media StandardScaler": 0.0,
                "DE StandardScaler": 1.0,
            }
            for col in FEATURE_COLUMNS
        ]
    )
    st.dataframe(skew_df, use_container_width=True, hide_index=True)

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        selected_hist_var = st.selectbox(
            "Seleccionar variable para inspeccionar distribución (Original vs Transformada Escalada):",
            FEATURE_COLUMNS,
            index=2,
        )
        fig_dist = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"{selected_hist_var} — Escala Original (Asimétrica)",
                f"{selected_hist_var} — Tras log1p + StandardScaler",
            ),
        )
        fig_dist.add_trace(
            go.Histogram(x=df_cust[selected_hist_var], nbinsx=45, marker_color="#F59E0B", name="Original"),
            row=1,
            col=1,
        )
        fig_dist.add_trace(
            go.Histogram(
                x=res["X_scaled_df"][selected_hist_var],
                nbinsx=45,
                marker_color="#6366F1",
                name="Escalada",
            ),
            row=1,
            col=2,
        )
        fig_dist.update_layout(
            template="plotly_white",
            height=360,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with viz_col2:
        corr_mat = pd.DataFrame(prep["corr_scaled"])
        fig_corr = px.imshow(
            corr_mat,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de Correlación de Pearson (Variables Transformadas y Escaladas)",
            template="plotly_white",
        )
        fig_corr.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# TAB 2: ETAPAS 4 Y 5 (K-MEANS Y DBSCAN EN ESPACIO ORIGINAL)
# ==============================================================================
with tabs[1]:
    st.subheader("Etapa 4: Exploración de K-Means ($K = 2..10$, `k-means++`, `n_init=20`, `random_state=42`)")
    st.info(km_info["selection_rationale"])

    km_hist_df = pd.DataFrame(km_info["k_search_results"])

    fig_km = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Gráfico del Codo (Inercia Intra-Cluster ↓)",
            "Coeficiente Silhouette (↑ mejor)",
            "Índice Davies-Bouldin (↓ mejor)",
            "Índice Calinski-Harabasz (↑ mejor)",
        ),
    )
    fig_km.add_trace(
        go.Scatter(x=km_hist_df["k"], y=km_hist_df["inertia"], mode="lines+markers", marker=dict(size=8, color="#2563EB"), name="Inercia"),
        row=1,
        col=1,
    )
    fig_km.add_trace(
        go.Scatter(x=km_hist_df["k"], y=km_hist_df["silhouette"], mode="lines+markers", marker=dict(size=8, color="#10B981"), name="Silhouette"),
        row=1,
        col=2,
    )
    fig_km.add_trace(
        go.Scatter(x=km_hist_df["k"], y=km_hist_df["davies_bouldin"], mode="lines+markers", marker=dict(size=8, color="#F59E0B"), name="DBI"),
        row=2,
        col=1,
    )
    fig_km.add_trace(
        go.Scatter(x=km_hist_df["k"], y=km_hist_df["calinski_harabasz"], mode="lines+markers", marker=dict(size=8, color="#9333EA"), name="CHI"),
        row=2,
        col=2,
    )
    for r_idx in [1, 2]:
        for c_idx in [1, 2]:
            fig_km.add_vline(
                x=km_info["best_k"],
                line_dash="dash",
                line_color="#EF4444",
                row=r_idx,
                col=c_idx,
            )
    fig_km.update_layout(
        template="plotly_white",
        height=520,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A"),
    )
    st.plotly_chart(fig_km, use_container_width=True)

    km_display_df = km_hist_df[["k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz", "min_cluster_pct", "cluster_sizes"]].copy()
    km_display_df["cluster_sizes"] = km_display_df["cluster_sizes"].apply(
        lambda d: ", ".join(f"C{k}: {v}" for k, v in d.items()) if isinstance(d, dict) else str(d)
    )
    st.dataframe(
        km_display_df.rename(
            columns={
                "k": "K (Clusters)",
                "inertia": "Inercia",
                "silhouette": "Silhouette",
                "davies_bouldin": "Davies-Bouldin (DBI)",
                "calinski_harabasz": "Calinski-Harabasz (CHI)",
                "min_cluster_pct": "% Cluster Mínimo",
                "cluster_sizes": "Tamaños por Cluster",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Etapa 5: DBSCAN — Curva $k$-Distances, Búsqueda Razonada y Puntos Núcleo/Borde/Ruido")

    db_col1, db_col2 = st.columns([1.05, 1.25])
    with db_col1:
        kdist = db_info["k_distance_summary"]
        fig_kdist = go.Figure()
        fig_kdist.add_trace(
            go.Scatter(
                x=kdist["curve_sampled_indices"],
                y=kdist["curve_sampled_distances"],
                mode="lines",
                line=dict(color="#2563EB", width=2.5),
                name=f"{kdist['k_neighbors']}-distance",
            )
        )
        fig_kdist.add_hline(
            y=db_info["best_eps"],
            line_dash="dash",
            line_color="#10B981",
            annotation_text=f"ε seleccionado = {db_info['best_eps']} (codo de densidad)",
        )
        fig_kdist.update_layout(
            title=f"Curva {kdist['k_neighbors']}-Distances Ordenada (Orientación de ε)",
            xaxis_title="Clientes ordenados por distancia al 11° vecino",
            yaxis_title="Distancia Euclídea (Espacio Escalado)",
            template="plotly_white",
            height=380,
            margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_kdist, use_container_width=True)

    with db_col2:
        bm = db_info["best_metrics"]
        pt_counts = pd.DataFrame(
            [
                {"Tipo de Punto": "Core (Núcleo Denso)", "Cantidad": bm["n_core"], "Porcentaje": round(bm["n_core"] / len(df_cust) * 100, 2)},
                {"Tipo de Punto": "Border (Borde de Vecindad)", "Cantidad": bm["n_border"], "Porcentaje": round(bm["n_border"] / len(df_cust) * 100, 2)},
                {"Tipo de Punto": "Noise (Ruido / Atípicos = -1)", "Cantidad": bm["n_noise"], "Porcentaje": bm["noise_pct"]},
            ]
        )
        fig_pts = px.bar(
            pt_counts,
            x="Tipo de Punto",
            y="Cantidad",
            color="Tipo de Punto",
            text=pt_counts["Porcentaje"].apply(lambda v: f"{v:.2f}%"),
            color_discrete_sequence=["#10B981", "#3B82F6", "#EF4444"],
            title=f"Desglose Topológico DBSCAN (ε={db_info['best_eps']}, min_samples={db_info['best_min_samples']})",
            template="plotly_white",
        )
        fig_pts.update_traces(textposition="outside", textfont=dict(color="#0F172A"))
        fig_pts.update_layout(
            height=380,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            yaxis=dict(range=[0, max(pt_counts["Cantidad"]) * 1.18]),
        )
        st.plotly_chart(fig_pts, use_container_width=True)

    db_grid_df = pd.DataFrame(db_info["grid_search_results"])
    valid_db_df = db_grid_df[db_grid_df["is_valid_config"]].copy()
    if "cluster_sizes" in valid_db_df.columns:
        valid_db_df["cluster_sizes"] = valid_db_df["cluster_sizes"].apply(
            lambda d: ", ".join(f"C{k}: {v}" for k, v in d.items()) if isinstance(d, dict) else str(d)
        )
    st.markdown(
        rf"**Configuraciones válidas ($\ge 2$ grupos excluyendo ruido):** Se encontraron **{len(valid_db_df)}** combinaciones válidas en la grilla explorada."
    )
    st.dataframe(
        valid_db_df[
            [
                "eps",
                "min_samples",
                "n_clusters",
                "n_core",
                "n_border",
                "n_noise",
                "noise_pct",
                "coverage_pct",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "cluster_sizes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ==============================================================================
# TAB 3: ETAPAS 6 Y 7 (PCA Y MODELOS COMBINADOS)
# ==============================================================================
with tabs[2]:
    st.subheader(
        f"Etapa 6: Análisis de Componentes Principales (PCA) — Selección de d = {pca_info['n_components_selected']} Componentes ({pca_info['cumulative_variance_selected_pct']:.2f}% Varianza)"
    )
    st.markdown(
        "Siguiendo la nota metodológica obligatoria, **PCA se evalúa exclusivamente como técnica de reducción de dimensionalidad y descorrelación ortogonal**, "
        "nunca como un algoritmo de clustering."
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Variables Originales", f"{pca_info['n_features_original']}", "Espacio R^6")
    p2.metric("Componentes Elegidos (d)", f"{pca_info['n_components_selected']}", "Objetivo >= 90% varianza")
    p3.metric("Varianza Acumulada", f"{pca_info['cumulative_variance_selected_pct']:.2f}%", f"Pérdida: {100 - pca_info['cumulative_variance_selected_pct']:.2f}%")
    p4.metric("Error de Reconstrucción (MSE)", f"{pca_info['reconstruction_mse']:.4f}", f"RMSE = {pca_info['reconstruction_rmse']:.4f}")

    scree_df = pd.DataFrame(pca_info["scree_and_reconstruction"])
    col_scree, col_load = st.columns(2)

    with col_scree:
        fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scree.add_trace(
            go.Bar(
                x=scree_df["PC"],
                y=scree_df["explained_variance_pct"],
                name="Varianza Individual (%)",
                marker_color="#4F46E5",
            ),
            secondary_y=False,
        )
        fig_scree.add_trace(
            go.Scatter(
                x=scree_df["PC"],
                y=scree_df["cumulative_variance_pct"],
                name="Varianza Acumulada (%)",
                mode="lines+markers+text",
                text=scree_df["cumulative_variance_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="top center",
                textfont=dict(color="#0F172A"),
                line=dict(color="#10B981", width=3),
            ),
            secondary_y=False,
        )
        fig_scree.add_trace(
            go.Scatter(
                x=scree_df["PC"],
                y=scree_df["reconstruction_mse"],
                name="MSE Reconstrucción",
                mode="lines+markers",
                line=dict(color="#D97706", width=2.5, dash="dot"),
            ),
            secondary_y=True,
        )
        fig_scree.add_hline(y=90.0, line_dash="dash", line_color="#EF4444", secondary_y=False)
        fig_scree.update_layout(
            title="Scree Plot, Varianza Acumulada (%) y Error de Reconstrucción (MSE)",
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_scree.update_yaxes(title_text="Varianza Explicada (%)", range=[0, 108], secondary_y=False)
        fig_scree.update_yaxes(title_text="MSE de Reconstrucción", secondary_y=True)
        st.plotly_chart(fig_scree, use_container_width=True)

    with col_load:
        load_df = pd.DataFrame(pca_info["loadings"])
        fig_load = px.imshow(
            load_df.iloc[:, :4],
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de Cargas (Loadings) de los Primeros 4 Componentes Principales",
            template="plotly_white",
        )
        fig_load.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_load, use_container_width=True)

    for pc_key, desc in pca_info["component_interpretations"].items():
        st.markdown(f"- **{desc}**")

    st.markdown("---")
    st.subheader("Etapa 7: Exploración Interactiva 2D y 3D en el Subespacio Proyectado (PC1, PC2, PC3)")

    vcol1, vcol2 = st.columns([1, 3])
    with vcol1:
        color_choice = st.radio(
            "Colorear proyección PCA según partición:",
            [
                "K-Means (Arquetipos Semánticos)",
                "PCA + K-Means (K=4)",
                "DBSCAN (Tipo de Punto: Núcleo/Borde/Ruido)",
                "PCA + DBSCAN",
            ],
        )
        dim_choice = st.radio("Modo de visualización:", ["3D Interactivo (PC1, PC2, PC3)", "2D Biplot (PC1 vs PC2)"])

    plot_df = df_cust.reset_index().copy()
    if color_choice == "K-Means (Arquetipos Semánticos)":
        color_col = "Cluster_Name"
    elif color_choice == "PCA + K-Means (K=4)":
        plot_df["PCA_KMeans_Label"] = "Cluster " + plot_df["PCA_KMeans_Cluster"].astype(str)
        color_col = "PCA_KMeans_Label"
    elif color_choice == "DBSCAN (Tipo de Punto: Núcleo/Borde/Ruido)":
        color_col = "DBSCAN_PointType"
    else:
        plot_df["PCA_DBSCAN_Label"] = np.where(
            plot_df["PCA_DBSCAN_Cluster"] == -1,
            "Ruido (-1)",
            "Cluster " + plot_df["PCA_DBSCAN_Cluster"].astype(str),
        )
        color_col = "PCA_DBSCAN_Label"

    with vcol2:
        if dim_choice.startswith("3D"):
            fig_pca_scatter = px.scatter_3d(
                plot_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color=color_col,
                hover_data=["Customer ID", "Recency", "Frequency", "Monetary", "AvgTicket", "CancellationRate"],
                opacity=0.78,
                title=f"Proyección 3D de Clientes en Componentes Principales ({color_choice})",
                template="plotly_white",
            )
            fig_pca_scatter.update_traces(marker=dict(size=3.5))
            fig_pca_scatter.update_layout(
                height=540,
                margin=dict(l=0, r=0, t=45, b=0),
                paper_bgcolor="#FFFFFF",
                font=dict(color="#0F172A"),
            )
        else:
            fig_pca_scatter = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color=color_col,
                hover_data=["Customer ID", "Recency", "Frequency", "Monetary", "AvgTicket", "CancellationRate"],
                opacity=0.75,
                title=f"Proyección 2D en Componentes Principales PC1 vs PC2 ({color_choice})",
                template="plotly_white",
            )
            fig_pca_scatter.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=45, b=20),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font=dict(color="#0F172A"),
            )
        st.plotly_chart(fig_pca_scatter, use_container_width=True)

# ==============================================================================
# TAB 4: ETAPAS 8, 9 Y 10 (COMPARACIÓN JUSTA, ESTABILIDAD Y DECISIÓN)
# ==============================================================================
with tabs[3]:
    st.subheader("Etapa 8: Matriz de Comparación Metodológica Justa de las 4 Alternativas")
    st.markdown(
        """
        > **Nota metodológica verificada:** Las cuatro particiones se comparan utilizando **las mismas observaciones y la representación escalada original común (`X_scaled`)** como referencia primaria justa. Las métricas en el espacio PCA se reportan únicamente como diagnóstico secundario. En DBSCAN y PCA+DBSCAN las métricas internas excluyen la etiqueta de ruido (`-1`), declarando explícitamente la **tasa de cobertura (%)** y la **cantidad de clientes evaluados ($N$)**.
        """
    )

    fair_df = pd.DataFrame(fair_matrix)
    display_fair = fair_df[
        [
            "algorithm",
            "hyperparameters",
            "n_clusters",
            "coverage_pct",
            "noise_pct",
            "n_evaluated",
            "silhouette_original",
            "davies_bouldin_original",
            "calinski_harabasz_original",
            "silhouette_pca_diagnostic",
            "inertia_original_comparable",
            "stability",
            "interpretability_and_utility",
        ]
    ].rename(
        columns={
            "algorithm": "Alternativa",
            "hyperparameters": "Hiperparámetros",
            "n_clusters": "Clusters Válidos",
            "coverage_pct": "Cobertura (%)",
            "noise_pct": "Ruido (%)",
            "n_evaluated": "N Evaluado",
            "silhouette_original": "Silhouette (Espacio Original)",
            "davies_bouldin_original": "Davies-Bouldin (Espacio Original)",
            "calinski_harabasz_original": "Calinski-Harabasz (Original)",
            "silhouette_pca_diagnostic": "Silhouette (Diag. PCA)",
            "inertia_original_comparable": "Inercia Comparable",
            "stability": "Estabilidad",
            "interpretability_and_utility": "Interpretabilidad y Utilidad Comercial",
        }
    )
    st.dataframe(display_fair, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Etapa 9: Pruebas de Estabilidad por Semillas y Sensibilidad de Parámetros")

    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown("##### Estabilidad de K-Means ante Perturbación de Semillas (`[10, 42, 99, 123, 2024]`)")
        seed_df = pd.DataFrame(stability["kmeans_seed_stability"]["runs"]).copy()
        if "cluster_sizes" in seed_df.columns:
            seed_df["cluster_sizes"] = seed_df["cluster_sizes"].apply(
                lambda d: ", ".join(f"C{k}: {v}" for k, v in d.items()) if isinstance(d, dict) else str(d)
            )
        st.dataframe(seed_df, use_container_width=True, hide_index=True)
        st.success(
            f"**Concordancia entre semillas (Adjusted Rand Index):** Media = **{stability['kmeans_seed_stability']['mean_pairwise_ari']:.4f}** "
            f"(Mínimo = {stability['kmeans_seed_stability']['min_pairwise_ari']:.4f}) | "
            f"**Desviación Estándar de Silhouette:** `{stability['kmeans_seed_stability']['silhouette_std']:.5f}` | "
            f"**Concordancia ARI K-Means vs PCA+K-Means:** `{stability['kmeans_vs_pca_kmeans_ari']:.4f}`."
        )

    with scol2:
        st.markdown("##### Sensibilidad de DBSCAN ante Variaciones de $\\varepsilon$ y `min_samples`")
        sens_df = pd.DataFrame(stability["dbscan_sensitivity"]["runs"])
        st.dataframe(sens_df, use_container_width=True, hide_index=True)
        st.warning(
            f"Pequeñas variaciones de $\\pm 0.10$ en $\\varepsilon$ hacen oscilar el porcentaje de ruido entre "
            f"**{stability['dbscan_sensitivity']['min_noise_pct']:.2f}%** y **{stability['dbscan_sensitivity']['max_noise_pct']:.2f}%**, "
            "y el número de grupos entre 1 y 12, confirmando alta sensibilidad al gradiente de densidad."
        )

    st.markdown("---")
    st.subheader("Etapa 10: Decisión Técnica Sustentada")
    dec = res["narrative_and_defense"]["final_decision_stage10"]
    st.markdown(
        f"""
        <div class="insight-card">
            <h4>🏆 Alternativa Seleccionada: {dec['chosen_alternative']}</h4>
            <p>{dec['justification']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for alt_name, reason in dec["why_others_not_selected_as_primary"].items():
        st.markdown(f"- **Por qué no se eligió `{alt_name}` como partición primaria única:** {reason}")

# ==============================================================================
# TAB 5: PERFILES EN UNIDADES ORIGINALES, ATÍPICOS Y 3 ESTRATEGIAS COMERCIALES
# ==============================================================================
with tabs[4]:
    st.subheader("Caracterización de Perfiles en Unidades Originales (Media, Mediana, Cantidad y Dispersión)")

    prof_df = pd.DataFrame(profiles)
    summary_table = prof_df[
        [
            "cluster_name",
            "count",
            "percentage",
            "Recency_mean",
            "Recency_median",
            "Recency_iqr",
            "Frequency_mean",
            "Frequency_median",
            "Frequency_iqr",
            "Monetary_mean",
            "Monetary_median",
            "Monetary_iqr",
            "AvgTicket_mean",
            "UniqueCategories_mean",
            "CancellationRate_mean",
        ]
    ].rename(
        columns={
            "cluster_name": "Arquetipo (Etiqueta Semántica)",
            "count": "Clientes (N)",
            "percentage": "% Cartera",
            "Recency_mean": "Recency Media (días)",
            "Recency_median": "Recency Mediana",
            "Recency_iqr": "Recency IQR",
            "Frequency_mean": "Frequency Media",
            "Frequency_median": "Frequency Mediana",
            "Frequency_iqr": "Frequency IQR",
            "Monetary_mean": "Monetary Media (£)",
            "Monetary_median": "Monetary Mediana (£)",
            "Monetary_iqr": "Monetary IQR (£)",
            "AvgTicket_mean": "Ticket Medio (£)",
            "UniqueCategories_mean": "SKUs Medios",
            "CancellationRate_mean": "Tasa Cancelación Media",
        }
    )
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    # Radar Chart de Perfiles Escalados + Boxplots
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        fig_radar = go.Figure()
        X_sc_with_cluster = res["X_scaled_df"].copy()
        X_sc_with_cluster["Cluster_Name"] = df_cust["Cluster_Name"]
        radar_means = X_sc_with_cluster.groupby("Cluster_Name")[FEATURE_COLUMNS].mean()

        for prof in profiles:
            cname = prof["cluster_name"]
            if cname in radar_means.index:
                vals = radar_means.loc[cname, FEATURE_COLUMNS].tolist()
                vals.append(vals[0])
                theta = FEATURE_COLUMNS + [FEATURE_COLUMNS[0]]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals,
                        theta=theta,
                        fill="toself",
                        name=prof["short_tag"],
                        line=dict(color=prof["color"], width=2.5),
                    )
                )
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#F8FAFC",
                radialaxis=dict(visible=True, range=[-1.5, 2.0], gridcolor="#CBD5E1", linecolor="#94A3B8"),
                angularaxis=dict(gridcolor="#CBD5E1", linecolor="#94A3B8"),
            ),
            title="Diagrama de Radar Multivariable por Perfil (Z-Scores Escalados)",
            template="plotly_white",
            height=430,
            margin=dict(l=40, r=40, t=55, b=30),
            paper_bgcolor="#FFFFFF",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with rcol2:
        box_var = st.selectbox(
            "Comparar distribución por segmento en unidades originales:",
            ["Monetary", "Recency", "Frequency", "AvgTicket", "UniqueCategories", "CancellationRate"],
            index=1,
        )
        fig_box = px.box(
            df_cust.reset_index(),
            x="Cluster_Name",
            y=box_var,
            color="Cluster_Name",
            log_y=box_var in ["Monetary", "Frequency", "AvgTicket", "UniqueCategories"],
            title=f"Distribución y Dispersión de {box_var} por Arquetipo de Cliente",
            template="plotly_white",
        )
        fig_box.update_layout(
            showlegend=False,
            height=380,
            margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("🚀 Tres Estrategias Comerciales Diferenciadas (Vinculadas a Evidencia Empírica)")
    st_cols = st.columns(3)
    for idx, strat in enumerate(strategies):
        with st_cols[idx]:
            st.markdown(
                f"""
                <div class="insight-card" style="height: 100%;">
                    <span class="status-badge badge-blue">Estrategia #{strat['strategy_id']}</span>
                    <h4 style="margin-top:0.5rem;">{strat['title']}</h4>
                    <p><b>Perfil Objetivo:</b> {strat['target_cluster']}</p>
                    <p style="margin-top:0.4rem; font-size:0.84rem; color:#334155;">{strat['empirical_evidence']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for act in strat["actions"]:
                st.markdown(f"- {act}")
            st.caption(f"🎯 **KPI Esperado:** {strat['expected_kpi']}")

    st.markdown("---")
    st.subheader("🔎 Análisis de Clientes Atípicos por Consumo (DBSCAN Noise = -1 & Percentil 99.5%)")
    st.markdown(
        f"DBSCAN identificó **{anomalies['n_dbscan_noise']} clientes atípicos ({anomalies['pct_dbscan_noise']:.2f}%)** cuyo gasto medio es de "
        f"**£{anomalies['noise_monetary_mean']:,.2f}** (frente a **£{anomalies['core_monetary_mean']:,.2f}** en los clientes normales) y una frecuencia media "
        f"de **{anomalies['noise_frequency_mean']:.1f} facturas**."
    )
    st.dataframe(pd.DataFrame(anomalies["top_anomalous_customers"]), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 6: SIMULADOR INTERACTIVO DE SEGMENTACIÓN Y PREGUNTAS DE SUSTENTACIÓN
# ==============================================================================
with tabs[5]:
    st.subheader("🎛️ Simulador en Tiempo Real de Asignación de Segmento para un Cliente")
    st.markdown(
        "Ingresa el comportamiento transaccional de un cliente para proyectarlo con el pipeline entrenado "
        "(`winsor` + `log1p` + `StandardScaler` + `PCA` + `KMeans`) y obtener su arquetipo comercial:"
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_recency = st.slider("Recency (días desde última compra)", 1, 730, 25)
        sim_freq = st.slider("Frequency (facturas válidas)", 1, 120, 12)
    with sc2:
        sim_monetary = st.number_input("Monetary — Gasto Total Acumulado (£)", min_value=10.0, max_value=250000.0, value=4500.0, step=100.0)
        sim_cats = st.slider("UniqueCategories (SKUs distintos)", 1, 500, 95)
    with sc3:
        sim_cancel = st.slider("CancellationRate (Tasa de Cancelación/Devolución)", 0.0, 1.0, 0.05, step=0.01)
        sim_avg_ticket = round(sim_monetary / max(sim_freq, 1), 2)
        st.metric("Ticket Promedio Calculado (AvgTicket)", f"£{sim_avg_ticket:,.2f}")

    # Inferencia con artefacto serializado
    raw_vec = pd.DataFrame(
        [
            {
                "Recency": float(sim_recency),
                "Frequency": float(sim_freq),
                "Monetary": float(sim_monetary),
                "AvgTicket": float(sim_avg_ticket),
                "UniqueCategories": float(sim_cats),
                "CancellationRate": float(sim_cancel),
            }
        ]
    )
    winsor_vec = raw_vec.copy()
    for c in FEATURE_COLUMNS:
        winsor_vec[c] = np.clip(winsor_vec[c], 0.0, artifact["winsor_caps"][c])
    log_vec = np.log1p(winsor_vec)
    scaled_vec = artifact["scaler"].transform(log_vec)
    pred_cid = int(artifact["kmeans_model"].predict(scaled_vec)[0])
    sem_info = artifact["semantic_mapping"][pred_cid]

    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: {sem_info['color']};">
            <h4>Segmento Asignado: <span style="color:{sem_info['color']};">{sem_info['name']}</span> (Cluster #{pred_cid})</h4>
            <p>{sem_info['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("🎓 Respuestas Rigurosas a las Preguntas de Investigación, Validez y Sustentación (Secciones 1, 4 y 8)")

    formatted_qas = format_ecommerce_defense_answers(res)
    for qa in formatted_qas:
        with st.expander(f"📌 {qa['question']}", expanded=True):
            st.markdown(f"**Foco de Rúbrica:** `{qa.get('rubric_focus', 'Sustentación obligatoria')}`")
            st.markdown(qa["answer"])

    with st.expander("⚠️ Amenazas a la Validez de la Solución y Protocolo de Verificación (Sección 4)", expanded=False):
        for threat in ECOMMERCE_ACADEMIC_CONTENT["validity_threats"]:
            st.markdown(f"#### {threat['threat']} (*{threat.get('category', '')}*)")
            st.markdown(f"- **Impacto Metodológico:** {threat.get('description', threat.get('impact', ''))}")
            st.markdown(f"- **Cómo Verificarla y Mitigarla:** {threat.get('verification_and_mitigation', threat.get('verification_protocol', ''))}")
