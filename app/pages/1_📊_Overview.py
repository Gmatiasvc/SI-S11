"""Página 1: Dashboard Ejecutivo Consolidado (Visión Integradora SI-S11)."""
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ecommerce.business_insights import run_full_ecommerce_pipeline
from src.hr_attrition.inference import run_full_hr_pipeline
from src.utils.academic_content import ECOMMERCE_ACADEMIC_CONTENT, HR_ACADEMIC_CONTENT

_css_file = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
if _css_file.exists():
    st.markdown(f"<style>{_css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)



@st.cache_resource(show_spinner="Cargando resultados y artefactos de Comercio Electrónico...")
def get_ecommerce_data():
    return run_full_ecommerce_pipeline(force_retrain=False)


@st.cache_resource(show_spinner="Cargando resultados y artefactos de RRHH Attrition...")
def get_hr_data():
    return run_full_hr_pipeline(force_retrain=False)


ecom = get_ecommerce_data()
hr = get_hr_data()

# Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <span class="badge-pill">Plataforma Unificada · Sistemas Inteligentes y Machine Learning</span>
        <h1>Centro Ejecutivo de Analítica Avanzada: E-Commerce & People Analytics</h1>
        <p>
            Solución integral que articula <b>Aprendizaje No Supervisado</b> (Segmentación RFM, K-Means, DBSCAN y PCA sobre
            1.06M de transacciones de comercio electrónico) y <b>Aprendizaje Supervisado</b> (Pipelines sin fuga de datos,
            optimización en 5 folds estratificados y predicción de rotación de personal con Árboles, Random Forest y Gradient Boosting).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fila de KPIs Consolidados (6 tarjetas)
c1, c2, c3, c4, c5, c6 = st.columns(6)

n_tx = ecom["preprocessing"]["n_raw_rows"]
n_cust = ecom["preprocessing"]["n_unique_customers"]
best_k = ecom["kmeans_original"]["best_k"]
ari_mean = ecom["stability_analysis"]["kmeans_seed_stability"]["mean_pairwise_ari"]

hr_total = hr["split_info"]["total_rows"]
hr_winner = hr["winner_name"]
hr_roc = hr["test_evaluation"]["roc_auc"]
hr_f1 = hr["test_evaluation"]["f1_tuned"]

with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Transacciones E-Commerce</div>
            <div class="kpi-value">{n_tx:,}</div>
            <div class="kpi-sub">UCI Online Retail II (2009–2011)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Clientes Únicos RFM</div>
            <div class="kpi-value">{n_cust:,}</div>
            <div class="kpi-sub">6 Variables Cuantitativas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Segmentación Ganadora</div>
            <div class="kpi-value">K-Means (K={best_k})</div>
            <div class="kpi-sub">Estabilidad ARI = {ari_mean:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Plantilla RRHH Evaluada</div>
            <div class="kpi-value">{hr_total:,}</div>
            <div class="kpi-sub">Tasa Rotación: {hr['eda_summary']['target_pct']['Yes']:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Clasificador Ganador</div>
            <div class="kpi-value">{hr_winner}</div>
            <div class="kpi-sub">Umbral OOF τ* = {hr['optimal_threshold']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c6:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Desempeño en Test RRHH</div>
            <div class="kpi-value">ROC {hr_roc:.3f}</div>
            <div class="kpi-sub">F1 Positivo = {hr_f1:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Síntesis ejecutiva lado a lado de ambos módulos
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown(
        """
        <div class="insight-card">
            <h4>🛒 Módulo 1: Segmentación de Clientes de Comercio Electrónico (Trabajo 01)</h4>
            <p>
                <b>Paradigma:</b> Aprendizaje No Supervisado (Objetivo Descriptivo y Descubrimiento de Estructuras).<br>
                <b>Flujo Metodológico (10 Etapas):</b> Agregación por <code>Customer ID</code> sobre compras válidas y cálculo de
                <b>Recency, Frequency, Monetary, AvgTicket, UniqueCategories y CancellationRate</b>. Tratamiento de asimetría con
                winsorización suave al 99.5% + <code>log1p</code> + <code>StandardScaler</code> compartido. Comparación justa de
                <b>K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN</b> sobre el espacio original escalado.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profiles_df = pd.DataFrame(ecom["cluster_profiles"])
    fig_ecom = px.bar(
        profiles_df,
        x="cluster_name",
        y="Monetary_mean",
        color="cluster_name",
        text=profiles_df["percentage"].apply(lambda p: f"{p:.1f}% cartera"),
        color_discrete_sequence=profiles_df["color"].tolist(),
        title="Gasto Promedio (£) y Peso de Cartera por Arquetipo de Cliente",
        labels={"cluster_name": "Arquetipo Semántico", "Monetary_mean": "Gasto Medio (£)"},
        template="plotly_white",
    )
    fig_ecom.update_traces(textposition="outside", textfont=dict(color="#0F172A"))
    fig_ecom.update_layout(
        showlegend=False,
        height=370,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A"),
        yaxis=dict(range=[0, profiles_df["Monetary_mean"].max() * 1.18]),
    )
    st.plotly_chart(fig_ecom, use_container_width=True)

with col_right:
    st.markdown(
        """
        <div class="insight-card">
            <h4>👥 Módulo 2: Predicción de Rotación de Empleados — HR Analytics (Grupo 08)</h4>
            <p>
                <b>Paradigma:</b> Aprendizaje Supervisado — Clasificación Binaria (<code>Attrition</code>: 1=Renuncia, 0=Permanece).<br>
                <b>Flujo Metodológico (16 Etapas):</b> Depuración de identificadores/constantes sin fuga, partición estratificada
                80/20 con Test aislado, <code>Pipeline</code> + <code>ColumnTransformer</code>, evaluación en los mismos 5 folds de
                <code>StratifiedKFold</code> y optimización de <b>Decision Tree, Random Forest y Gradient Boosting</b> con F1 positivo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_df = pd.DataFrame(hr["comparison_table"])
    fig_hr = go.Figure()
    fig_hr.add_trace(
        go.Bar(
            name="CV Inicial (Base)",
            x=comp_df["model_name"],
            y=comp_df["cv_initial_f1_mean"],
            marker_color="#64748B",
            text=comp_df["cv_initial_f1_mean"].round(3),
            textposition="outside",
            textfont=dict(color="#0F172A"),
        )
    )
    fig_hr.add_trace(
        go.Bar(
            name="Mejor CV (Tuned 0.50)",
            x=comp_df["model_name"],
            y=comp_df["cv_best_f1_mean"],
            marker_color="#4F46E5",
            text=comp_df["cv_best_f1_mean"].round(3),
            textposition="outside",
            textfont=dict(color="#0F172A"),
        )
    )
    fig_hr.add_trace(
        go.Bar(
            name="Mejor CV + Umbral OOF τ*",
            x=comp_df["model_name"],
            y=comp_df["oof_f1_optimal_threshold"],
            marker_color="#10B981",
            text=comp_df["oof_f1_optimal_threshold"].round(3),
            textposition="outside",
            textfont=dict(color="#0F172A"),
        )
    )
    fig_hr.update_layout(
        title="Progresión del F1-Score (Clase Positiva) en Validación Cruzada (5 Folds)",
        barmode="group",
        template="plotly_white",
        height=370,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A"),
        yaxis=dict(title="F1-Score (Attrition = 1)", range=[0, 0.68]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hr, use_container_width=True)

st.markdown("---")

# Tabla comparativa epistemológica y de gobernanza de ambos casos
st.subheader("🧭 Síntesis Metodológica y Rigor Experimental entre Ambos Enunciados")

comparison_overview = pd.DataFrame(
    [
        {
            "Dimensión": "Naturaleza del Problema",
            "Módulo 1: E-Commerce (Trabajo 01)": "Aprendizaje No Supervisado (Clustering + Reducción Dimensional PCA)",
            "Módulo 2: Rotación RRHH (Grupo 08)": "Aprendizaje Supervisado (Clasificación Binaria Desbalanceada 16.1% vs 83.9%)",
        },
        {
            "Dimensión": "Unidad de Análisis y Dataset",
            "Módulo 1: E-Commerce (Trabajo 01)": f"{n_cust:,} clientes únicos agregados desde {n_tx:,} facturas (UCI Online Retail II)",
            "Módulo 2: Rotación RRHH (Grupo 08)": f"{hr_total:,} colaboradores con 30 variables pre-evento limpias (IBM HR Attrition)",
        },
        {
            "Dimensión": "Preprocesamiento y Prevención de Fuga",
            "Módulo 1: E-Commerce (Trabajo 01)": "Winsorización p99.5 + log1p + único StandardScaler compartido (X_scaled)",
            "Módulo 2: Rotación RRHH (Grupo 08)": "Split 80/20 estratificado; ColumnTransformer (Mediana+Scaler / Moda+OHE) ajustado solo en Train",
        },
        {
            "Dimensión": "Modelos Comparados",
            "Módulo 1: E-Commerce (Trabajo 01)": "1) K-Means, 2) DBSCAN, 3) PCA + K-Means, 4) PCA + DBSCAN",
            "Módulo 2: Rotación RRHH (Grupo 08)": "1) Decision Tree, 2) Random Forest, 3) Gradient Boosting (mismos 5 folds)",
        },
        {
            "Dimensión": "Métrica Principal y Selección",
            "Módulo 1: E-Commerce (Trabajo 01)": "Silhouette, Davies-Bouldin, Calinski-Harabasz en espacio original + Estabilidad ARI",
            "Módulo 2: Rotación RRHH (Grupo 08)": "F1-Score de clase positiva (Attrition=1) con umbral calibrado en OOF CV",
        },
        {
            "Dimensión": "Decisión / Modelo Ganador",
            "Módulo 1: E-Commerce (Trabajo 01)": f"K-Means (K={best_k}, Silhouette={ecom['kmeans_original']['best_metrics']['silhouette']:.4f}, ARI={ari_mean:.4f}) + DBSCAN para anomalías",
            "Módulo 2: Rotación RRHH (Grupo 08)": f"{hr_winner} (F1 OOF={hr['tuning_summaries'][hr_winner]['oof_threshold_tuning']['oof_f1_optimal']:.4f}, Test ROC-AUC={hr_roc:.4f})",
        },
    ]
)
st.dataframe(comparison_overview, use_container_width=True, hide_index=True)

with st.expander("📋 Ver Certificación de Procedencia, Licencias y Trazabilidad de Datasets", expanded=False):
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        prov_e = ECOMMERCE_ACADEMIC_CONTENT["dataset_provenance"]
        st.markdown(f"#### 🛒 {prov_e['name']}")
        st.markdown(
            f"- **URL Kaggle:** [{prov_e['kaggle_url']}]({prov_e['kaggle_url']})\n"
            f"- **URL UCI Oficial:** [{prov_e['uci_url']}]({prov_e['uci_url']})\n"
            f"- **Autor:** {prov_e['author']}\n"
            f"- **Licencia:** `{prov_e['license']}`\n"
            f"- **Cita APA 7:** *{prov_e['citation_apa7']}*\n"
            f"- **Unidad de Análisis:** {prov_e['unit_of_analysis']}"
        )
    with dcol2:
        prov_h = HR_ACADEMIC_CONTENT["dataset_provenance"]
        st.markdown(f"#### 👥 {prov_h['name']}")
        st.markdown(
            f"- **URL Kaggle:** [{prov_h['kaggle_url']}]({prov_h['kaggle_url']})\n"
            f"- **Autor:** {prov_h['author']}\n"
            f"- **Licencia:** `{prov_h['license']}`\n"
            f"- **Observaciones:** {prov_h['observations_count']:,} empleados ({prov_h['features_total_count']} atributos totales, {prov_h['predictors_clean_count']} predictores válidos)\n"
            f"- **Variable Objetivo:** {prov_h['target_variable']['mapping']}"
        )
