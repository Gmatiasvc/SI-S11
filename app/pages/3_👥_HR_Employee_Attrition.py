"""Página 3: Módulo 2 — Predicción de Rotación de Empleados (Grupo 08 · Aprendizaje Supervisado)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.datadownload.loaders import load_raw_hr
from src.hr_attrition.inference import (
    predict_employee_profiles,
    run_full_hr_pipeline,
)
from src.utils.academic_content import (
    HR_ACADEMIC_CONTENT,
    format_hr_defense_answers,
)

_css_file = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
if _css_file.exists():
    st.markdown(f"<style>{_css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)



@st.cache_resource(show_spinner="Cargando Pipeline ganador serializado (.joblib) y resultados de RRHH...")
def load_hr_bundle():
    return run_full_hr_pipeline(force_retrain=False)


@st.cache_data(show_spinner=False)
def get_raw_hr_df():
    return load_raw_hr()


hr = load_hr_bundle()
df_raw_hr = get_raw_hr_df()

winner_name = hr["winner_name"]
opt_tau = hr["optimal_threshold"]
split_info = hr["split_info"]
eda = hr["eda_summary"]
comp_table = hr["comparison_table"]
tuning = hr["tuning_summaries"]
test_eval = hr["test_evaluation"]
interp = hr["interpretability"]
five_preds = hr["five_synthetic_predictions"]
ui_specs = hr["feature_ui_specs"]

# Hero Banner
st.markdown(
    f"""
    <div class="hero-banner">
        <span class="badge-pill">Grupo 08 · Aprendizaje Supervisado · 16 Etapas Obligatorias</span>
        <h1>👥 Predicción de Rotación de Empleados (Decision Tree, Random Forest & Gradient Boosting)</h1>
        <p>
            Sistema preventivo de retención de talento sobre <b>{split_info['total_rows']:,} colaboradores</b> (<i>IBM HR Analytics</i>)
            con <code>Pipeline</code> + <code>ColumnTransformer</code> sin fuga de datos, optimización sobre los mismos
            <b>5 folds estratificados</b> usando <b>F1 de clase positiva</b>, umbral óptimo calibrado en validación
            (<b>τ* = {opt_tau:.2f}</b>) y artefacto ganador (<b>{winner_name}</b>) cargado vía <code>joblib</code>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pestañas que cubren exactamente los requisitos de la Sección 6 del PDF de Grupo 08
tabs = st.tabs(
    [
        "1️⃣ Problema, Dataset y EDA (Etapas 01–07)",
        "2️⃣ Comparación Base vs. Optimizada y Parámetros (Etapas 08–12)",
        "3️⃣ Evaluación Final en Test e Interpretación (Etapas 13–14)",
        "4️⃣ 5 Perfiles Ficticios y Simulador Predictivo (Etapas 15–16)",
        "5️⃣ Ética en RRHH y Respuestas de Sustentación (Sección 9)",
    ]
)

# ==============================================================================
# TAB 1: PROBLEMA, DATASET, EDA Y PREPROCESAMIENTO SIN FUGA (ETAPAS 01-07)
# ==============================================================================
with tabs[0]:
    st.subheader("Etapas 01 y 02: Definición del Problema, Pregunta de Investigación y Carga de Datos")
    prob = HR_ACADEMIC_CONTENT["problem_definition"]
    prov = HR_ACADEMIC_CONTENT["dataset_provenance"]
    m_just = HR_ACADEMIC_CONTENT["metric_justification"]

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>❓ Pregunta de Investigación</h4>
                <p>{prob['research_question']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(prob["binary_classification_formulation"])
    with c_p2:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>⚖️ Métrica Principal Elegida ANTES del Ajuste: {m_just['primary_metric']}</h4>
                <p>{m_just['why_chosen_before_tuning']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(m_just["formula_latex"])

    with st.expander("📘 Justificación Completa de la Métrica F1 sobre Accuracy y Procedencia del Dataset", expanded=False):
        st.markdown(m_just["technical_reason_class_imbalance"])
        st.markdown(m_just["business_reason_asymmetric_costs"])
        st.markdown("---")
        st.markdown(
            f"- **Dataset:** `{prov['name']}` | **URL Exacta:** [{prov['kaggle_url']}]({prov['kaggle_url']})\n"
            f"- **Autor / Licencia:** {prov['author']} — `{prov['license']}` (Acceso: {prov['access_date']})\n"
            f"- **Significado de Objetivo (`Attrition`):** {prov['target_variable']['mapping']}"
        )

    st.markdown("---")
    st.subheader("Etapas 03, 04, 05, 06 y 07: Exploración (EDA), Limpieza de Fuga, Split 80/20 y ColumnTransformer")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Observaciones Totales", f"{split_info['total_rows']:,}", "35 columnas originales")
    k2.metric("Faltantes / Duplicados", f"{eda['missing_total']} / {eda['n_duplicates']}", "Integridad 100% verificada")
    k3.metric("Clase Positiva (Attrition=Yes)", f"{eda['target_counts']['Yes']} ({eda['target_pct']['Yes']}%)", "Desbalance 1 : 5.2")
    k4.metric("Partición Train (80%)", f"{split_info['train_rows']:,} filas", f"{split_info['train_positive_rate_pct']}% positivos")
    k5.metric("Partición Test Reservada (20%)", f"{split_info['test_rows']:,} filas", f"{split_info['test_positive_rate_pct']}% positivos")

    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        ot_ct = (
            df_raw_hr.groupby(["OverTime", "Attrition"])
            .size()
            .reset_index(name="Empleados")
        )
        fig_ot = px.bar(
            ot_ct,
            x="OverTime",
            y="Empleados",
            color="Attrition",
            barmode="group",
            text="Empleados",
            color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
            title="Impacto de las Horas Extra (OverTime) en la Rotación de Empleados",
            template="plotly_white",
        )
        fig_ot.update_traces(textposition="outside", textfont=dict(color="#0F172A"))
        fig_ot.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            yaxis=dict(range=[0, ot_ct["Empleados"].max() * 1.18]),
        )
        st.plotly_chart(fig_ot, use_container_width=True)

    with eda_c2:
        corr_series = pd.Series(eda["top_numeric_correlations_with_attrition"]).head(12)
        corr_df = pd.DataFrame({"Variable": corr_series.index, "Correlación con Attrition": corr_series.values})
        fig_corr_hr = px.bar(
            corr_df.sort_values("Correlación con Attrition"),
            x="Correlación con Attrition",
            y="Variable",
            orientation="h",
            color="Correlación con Attrition",
            color_continuous_scale="RdBu_r",
            range_color=[-0.25, 0.25],
            text=corr_df.sort_values("Correlación con Attrition")["Correlación con Attrition"].round(3),
            title="Top 12 Correlaciones Numéricas con la Rotación (Attrition = 1)",
            template="plotly_white",
        )
        fig_corr_hr.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_corr_hr, use_container_width=True)

    st.markdown("##### 🛡️ Auditoría de Limpieza Inicial y Disponibilidad Pre-Evento (Etapas 04, 05 y 07)")
    st.info(prov["pre_event_availability_and_leakage_control"]["justification"])
    drop_df = pd.DataFrame(prov["pre_event_availability_and_leakage_control"]["dropped_columns"])
    st.dataframe(
        drop_df.rename(columns={"column": "Columna Eliminada", "reason": "Justificación Técnica y Metodológica"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("📋 Ver Resumen Completo de las 16 Etapas Obligatorias (Paso 01 a Paso 16)", expanded=False):
        st.dataframe(pd.DataFrame(HR_ACADEMIC_CONTENT["sixteen_stages_summary"]), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: COMPARACIÓN BASE VS OPTIMIZADA Y PARÁMETROS (ETAPAS 08-12)
# ==============================================================================
with tabs[1]:
    st.subheader("Etapas 08 a 12: Comparación Experimental en los Mismos 5 Folds (`StratifiedKFold`)")
    st.markdown(
        "Los tres algoritmos (**Decision Tree**, **Random Forest** y **Gradient Boosting**) se ajustaron sobre la **misma partición Train (1,176 filas)** "
        "y exactamente los **mismos 5 folds** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), explorando **24 configuraciones aleatorias por modelo** (`RandomizedSearchCV`)."
    )

    # Tabla obligatoria de la Sección 5 del PDF
    comp_rows = []
    for row in comp_table:
        comp_rows.append(
            {
                "Modelo": f"🏆 {row['model_name']}" if row["is_winner"] else row["model_name"],
                "CV Inicial (F1 ± SD)": f"{row['cv_initial_f1_mean']:.4f} ± {row['cv_initial_f1_std']:.4f}",
                "Mejor CV 0.50 (F1 ± SD)": f"{row['cv_best_f1_mean']:.4f} ± {row['cv_best_f1_std']:.4f}",
                "Mejor CV OOF τ* (F1)": f"{row['oof_f1_optimal_threshold']:.4f} (τ*={row['optimal_threshold']:.2f})",
                "Precision OOF": round(row["cv_best_precision"], 4),
                "Recall OOF": round(row["cv_best_recall"], 4),
                "ROC-AUC CV": round(row["cv_best_roc_auc"], 4),
                "PR-AUC CV": round(row["cv_best_pr_auc"], 4),
                "Hiperparámetros Óptimos (best_params_)": str(row["best_params"]),
                "Tiempo (s)": f"{row['search_time_sec']:.2f} s",
            }
        )
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Gráfico de F1 por Fold y Curva de Umbral OOF
    cv_col1, cv_col2 = st.columns(2)
    with cv_col1:
        fold_records = []
        for mname, tinfo in tuning.items():
            for fold_idx, val in enumerate(tinfo["best_cv_f1_folds"], start=1):
                fold_records.append({"Modelo": mname, "Fold": f"Fold {fold_idx}", "F1 Validación": val})
        fold_df = pd.DataFrame(fold_records)
        fig_folds = px.line(
            fold_df,
            x="Fold",
            y="F1 Validación",
            color="Modelo",
            markers=True,
            color_discrete_map={"DecisionTree": "#D97706", "RandomForest": "#2563EB", "GradientBoosting": "#10B981"},
            title="Estabilidad del F1-Score a través de los 5 Folds Compartidos de Train",
            template="plotly_white",
        )
        fig_folds.update_traces(line=dict(width=3), marker=dict(size=9))
        fig_folds.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_folds, use_container_width=True)

    with cv_col2:
        t_curve = pd.DataFrame(tuning[winner_name]["oof_threshold_tuning"]["threshold_curve"])
        fig_tau = go.Figure()
        fig_tau.add_trace(go.Scatter(x=t_curve["threshold"], y=t_curve["f1"], name="F1-Score (OOF)", line=dict(color="#10B981", width=3)))
        fig_tau.add_trace(go.Scatter(x=t_curve["threshold"], y=t_curve["precision"], name="Precision (OOF)", line=dict(color="#2563EB", width=2, dash="dot")))
        fig_tau.add_trace(go.Scatter(x=t_curve["threshold"], y=t_curve["recall"], name="Recall (OOF)", line=dict(color="#D97706", width=2, dash="dash")))
        fig_tau.add_vline(
            x=opt_tau,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Umbral Óptimo Validación τ* = {opt_tau:.2f}",
        )
        fig_tau.update_layout(
            title=f"Calibración de Umbral Out-of-Fold en Validación ({winner_name} — Sin tocar Test)",
            xaxis_title="Umbral de Decisión (τ)",
            yaxis_title="Métrica en Validación Cruzada (OOF)",
            template="plotly_white",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_tau, use_container_width=True)

    st.markdown("##### 🔎 Explorador de Configuraciones de Hiperparámetros Evaluadas en `RandomizedSearchCV`")
    sel_model_params = st.selectbox("Seleccionar algoritmo para ver su Top 10 de configuraciones en CV:", ["GradientBoosting", "RandomForest", "DecisionTree"])
    top_cands_df = pd.DataFrame(tuning[sel_model_params]["top_candidates"])
    top_cands_df["params"] = top_cands_df["params"].astype(str)
    st.dataframe(top_cands_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: EVALUACIÓN FINAL EN TEST E INTERPRETACIÓN (ETAPAS 13 Y 14)
# ==============================================================================
with tabs[2]:
    st.subheader(f"Etapa 13: Evaluación Final Única en el Conjunto Test Reservado ({winner_name})")

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    t1.metric(
        f"F1-Score Test (τ*={opt_tau:.2f})",
        f"{test_eval['f1_tuned']:.4f}",
        f"+{test_eval['f1_tuned'] - test_eval['f1_default_050']:.4f} vs τ=0.50",
    )
    t2.metric("Recall Test (Sensibilidad)", f"{test_eval['recall_tuned']:.4f}", f"Base 0.50: {test_eval['recall_default_050']:.4f}")
    t3.metric("Precision Test", f"{test_eval['precision_tuned']:.4f}", f"Base 0.50: {test_eval['precision_default_050']:.4f}")
    t4.metric("ROC-AUC Test", f"{test_eval['roc_auc']:.4f}", "Capacidad discriminante")
    t5.metric("PR-AUC Test", f"{test_eval['pr_auc']:.4f}", f"Base azar: {split_info['test_positive_rate_pct']/100:.3f}")
    t6.metric("Accuracy Test", f"{test_eval['accuracy_tuned']:.4f}", f"{test_eval['n_test_samples']} casos evaluados")

    cm_col, roc_col, pr_col = st.columns(3)

    with cm_col:
        cm = np.array(test_eval["confusion_matrix_tuned"])
        labels_cm = [["TN (Permanece OK)", "FP (Falsa Alarma)"], ["FN (Fuga No Detectada)", "TP (Fuga Detectada)"]]
        text_cm = [[f"{labels_cm[i][j]}<br><b>{cm[i][j]}</b>" for j in range(2)] for i in range(2)]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred: 0 (Permanece)", "Pred: 1 (Renuncia)"],
                y=["Real: 0 (Permanece)", "Real: 1 (Renuncia)"],
                text=text_cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title=f"Matriz de Confusión en Test (τ* = {opt_tau:.2f})",
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            font=dict(color="#0F172A"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with roc_col:
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=test_eval["roc_curve"]["fpr"],
                y=test_eval["roc_curve"]["tpr"],
                mode="lines",
                name=f"ROC (AUC = {test_eval['roc_auc']:.4f})",
                line=dict(color="#10B981", width=3),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.18)",
            )
        )
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (0.50)", line=dict(color="#64748B", dash="dash"))
        )
        fig_roc.update_layout(
            title=f"Curva ROC en Test (AUC = {test_eval['roc_auc']:.4f})",
            xaxis_title="False Positive Rate (1 - Especificidad)",
            yaxis_title="True Positive Rate (Recall)",
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with pr_col:
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=test_eval["pr_curve"]["recall"],
                y=test_eval["pr_curve"]["precision"],
                mode="lines",
                name=f"PR (AUC = {test_eval['pr_auc']:.4f})",
                line=dict(color="#4F46E5", width=3),
                fill="tozeroy",
                fillcolor="rgba(79, 70, 229, 0.18)",
            )
        )
        fig_pr.update_layout(
            title=f"Curva Precision-Recall en Test (PR-AUC = {test_eval['pr_auc']:.4f})",
            xaxis_title="Recall (Sensibilidad)",
            yaxis_title="Precision (Valor Predictivo Positivo)",
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")
    st.subheader("Etapa 14: Importancia de Variables (MDI & Permutation) y Estudio Obligatorio por Antigüedad")

    imp_col1, imp_col2 = st.columns(2)
    with imp_col1:
        mdi_df = pd.DataFrame(interp["mdi_top_features"]).head(12).sort_values("importance_mdi")
        fig_mdi = px.bar(
            mdi_df,
            x="importance_mdi",
            y="feature",
            orientation="h",
            text=mdi_df["importance_mdi"].round(3),
            color="importance_mdi",
            color_continuous_scale="Tealgrn",
            title="Top 12 Variables por Reducción de Impureza (MDI)",
            template="plotly_white",
        )
        fig_mdi.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_mdi, use_container_width=True)

    with imp_col2:
        perm_df = pd.DataFrame(interp["permutation_top_features"]).head(12).sort_values("importance_mean")
        fig_perm = px.bar(
            perm_df,
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
            text=perm_df["importance_mean"].round(3),
            color="importance_mean",
            color_continuous_scale="Purpor",
            title="Top 12 Variables por Permutation Importance (Impacto en F1 Test)",
            template="plotly_white",
        )
        fig_perm.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("##### 📊 Análisis Específico Obligatorio: Diferencias por Antigüedad (`YearsAtCompany`) y Contraste Recall / Precision / F1")
    ten_col1, ten_col2 = st.columns([1.15, 1])
    with ten_col1:
        ten_test_df = pd.DataFrame(interp["tenure_band_test_metrics"])
        st.dataframe(
            ten_test_df.rename(
                columns={
                    "tenure_band": "Tramo de Antigüedad",
                    "n_employees_test": "Empleados (Test)",
                    "actual_attrition_count": "Salidas Reales",
                    "actual_attrition_rate_pct": "Tasa Rotación Real (%)",
                    "mean_predicted_risk_pct": "Riesgo Medio Predicho (%)",
                    "recall": "Recall",
                    "precision": "Precision",
                    "f1": "F1-Score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(HR_ACADEMIC_CONTENT["tenure_and_metric_tradeoff_analysis"]["tenure_differences_study"])

    with ten_col2:
        fig_ten = go.Figure()
        fig_ten.add_trace(
            go.Bar(
                name="Tasa Rotación Real (%)",
                x=ten_test_df["tenure_band"],
                y=ten_test_df["actual_attrition_rate_pct"],
                marker_color="#EF4444",
                text=ten_test_df["actual_attrition_rate_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(color="#0F172A"),
            )
        )
        fig_ten.add_trace(
            go.Bar(
                name="Riesgo Medio Predicho (%)",
                x=ten_test_df["tenure_band"],
                y=ten_test_df["mean_predicted_risk_pct"],
                marker_color="#2563EB",
                text=ten_test_df["mean_predicted_risk_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(color="#0F172A"),
            )
        )
        fig_ten.update_layout(
            title="Tasa de Rotación Real vs. Riesgo Predicho por Antigüedad",
            barmode="group",
            template="plotly_white",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
            yaxis=dict(range=[0, 32]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ten, use_container_width=True)

# ==============================================================================
# TAB 4: 5 PERFILES FICTICIOS Y SIMULADOR PREDICTIVO EN TIEMPO REAL (ETAPAS 15 Y 16)
# ==============================================================================
with tabs[3]:
    st.markdown(
        """
        <div class="ethics-alert">
            ⚠️ <b>Advertencia Ética y Legal Obligatoria (Sección 6 de la Guía):</b> Las estimaciones de riesgo generadas
            por este sistema son indicadores estadísticos preventivos para orientar programas de bienestar, ajuste salarial
            y retención voluntaria. <b>Bajo ninguna circunstancia implican ni justifican una decisión laboral automática,
            sanción o desvinculación del colaborador.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Etapa 15: Evaluación de Cinco Perfiles Nuevos, Razonables y Completamente Descritos")
    preds_summary_rows = []
    for p in five_preds:
        preds_summary_rows.append(
            {
                "ID": p["profile_id"],
                "Perfil de Colaborador": p["profile_name"],
                "Probabilidad de Salida (%)": f"{p['attrition_probability_pct']:.2f}%",
                f"Predicción (τ*={p['threshold_applied']})": p["predicted_label"],
                "Nivel de Alerta": p["risk_level"],
                "Factores Clave Detectados": " | ".join(p["key_drivers"][:3]),
                "Acción Preventiva RRHH": p["hr_action"],
            }
        )
    st.dataframe(pd.DataFrame(preds_summary_rows), use_container_width=True, hide_index=True)

    with st.expander("🔍 Ver Descripción Completa y Atributos de los 5 Perfiles Evaluados", expanded=False):
        for p in five_preds:
            st.markdown(
                f"**{p['profile_name']}** — **Probabilidad:** `{p['attrition_probability_pct']:.2f}%` ({p['risk_level']})\n\n"
                f"*{p['narrative_description']}*\n\n"
                f"- **Factores:** {', '.join(p['key_drivers'])}\n"
                f"- **Recomendación de Bienestar:** {p['hr_action']}"
            )
            st.markdown("---")

    st.subheader("Etapa 16: Formulario Predictivo Interactivo en Tiempo Real (Pipeline `joblib` Pre-entrenado)")
    st.markdown(
        "Puedes cargar como plantilla cualquiera de los 5 perfiles ficticios o configurar libremente los atributos "
        "de un colaborador para estimar su riesgo de rotación instantáneamente sin re-entrenar el modelo:"
    )

    preset_options = ["Personalizado (Valores Medianos de Plantilla)"] + [p["profile_name"] for p in five_preds]
    selected_preset = st.selectbox("Cargar perfil base en el formulario:", preset_options, index=1)

    if selected_preset == "Personalizado (Valores Medianos de Plantilla)":
        base_feats = {
            k: (int(v["median"]) if v["type"] == "numeric" else v["default"])
            for k, v in ui_specs.items()
        }
    else:
        matched_p = next(p for p in five_preds if p["profile_name"] == selected_preset)
        base_feats = matched_p["features"].copy()

    form_c1, form_c2, form_c3, form_c4 = st.columns(4)
    with form_c1:
        f_age = st.slider("Edad (Age)", 18, 60, int(base_feats["Age"]))
        f_income = st.number_input("Ingreso Mensual ($ MonthlyIncome)", 1000, 20000, int(base_feats["MonthlyIncome"]), step=150)
        f_overtime = st.selectbox("Horas Extra (OverTime)", ["Yes", "No"], index=0 if base_feats["OverTime"] == "Yes" else 1)
        f_travel = st.selectbox(
            "Viajes de Negocio (BusinessTravel)",
            ui_specs["BusinessTravel"]["options"],
            index=ui_specs["BusinessTravel"]["options"].index(base_feats["BusinessTravel"]),
        )
    with form_c2:
        f_dept = st.selectbox(
            "Departamento (Department)",
            ui_specs["Department"]["options"],
            index=ui_specs["Department"]["options"].index(base_feats["Department"]),
        )
        f_role = st.selectbox(
            "Rol Laboral (JobRole)",
            ui_specs["JobRole"]["options"],
            index=ui_specs["JobRole"]["options"].index(base_feats["JobRole"]),
        )
        f_level = st.slider("Nivel Jerárquico (JobLevel)", 1, 5, int(base_feats["JobLevel"]))
        f_dist = st.slider("Distancia al Hogar (km DistanceFromHome)", 1, 29, int(base_feats["DistanceFromHome"]))
    with form_c3:
        f_years_comp = st.slider("Antigüedad en la Empresa (YearsAtCompany)", 0, 40, int(base_feats["YearsAtCompany"]))
        f_years_role = st.slider("Años en el Rol Actual (YearsInCurrentRole)", 0, 18, int(base_feats["YearsInCurrentRole"]))
        f_years_promo = st.slider("Años desde Última Promoción", 0, 15, int(base_feats["YearsSinceLastPromotion"]))
        f_total_years = st.slider("Experiencia Laboral Total (TotalWorkingYears)", 0, 40, int(base_feats["TotalWorkingYears"]))
    with form_c4:
        f_env_sat = st.slider("Satisfacción Ambiental (1-4)", 1, 4, int(base_feats["EnvironmentSatisfaction"]))
        f_job_sat = st.slider("Satisfacción Laboral (1-4)", 1, 4, int(base_feats["JobSatisfaction"]))
        f_wlb = st.slider("Balance Vida-Trabajo (1-4)", 1, 4, int(base_feats["WorkLifeBalance"]))
        f_stock = st.slider("Nivel Opciones Acciones (0-3)", 0, 3, int(base_feats["StockOptionLevel"]))

    sim_features = base_feats.copy()
    sim_features.update(
        {
            "Age": f_age,
            "MonthlyIncome": f_income,
            "OverTime": f_overtime,
            "BusinessTravel": f_travel,
            "Department": f_dept,
            "JobRole": f_role,
            "JobLevel": f_level,
            "DistanceFromHome": f_dist,
            "YearsAtCompany": f_years_comp,
            "YearsInCurrentRole": f_years_role,
            "YearsSinceLastPromotion": f_years_promo,
            "TotalWorkingYears": f_total_years,
            "EnvironmentSatisfaction": f_env_sat,
            "JobSatisfaction": f_job_sat,
            "WorkLifeBalance": f_wlb,
            "StockOptionLevel": f_stock,
        }
    )

    sim_result = predict_employee_profiles(
        hr["winner_pipeline"],
        [{"profile_id": "LIVE", "profile_name": "Simulación Interactiva", "features": sim_features}],
        optimal_threshold=opt_tau,
    )[0]

    res_c1, res_c2 = st.columns([1, 1.35])
    with res_c1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sim_result["attrition_probability_pct"],
                number={"suffix": "%", "font": {"size": 42, "color": "#0F172A"}},
                title={"text": f"Probabilidad de Rotación (Umbral τ* = {opt_tau*100:.0f}%)", "font": {"size": 15, "color": "#0F172A"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#334155"},
                    "bar": {"color": sim_result["badge_color"]},
                    "bgcolor": "#FFFFFF",
                    "borderwidth": 1,
                    "bordercolor": "#CBD5E1",
                    "steps": [
                        {"range": [0, opt_tau * 100], "color": "#D1FAE5"},
                        {"range": [opt_tau * 100, 55], "color": "#FEF3C7"},
                        {"range": [55, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "#DC2626", "width": 4},
                        "thickness": 0.8,
                        "value": opt_tau * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=290,
            margin=dict(l=20, r=20, t=45, b=10),
            paper_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_c2:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color: {sim_result['badge_color']};">
                <h4>Diagnóstico del Modelo ({winner_name}): <span style="color:{sim_result['badge_color']};">{sim_result['risk_level']}</span></h4>
                <p><b>Clasificación Binaria (τ* = {opt_tau:.2f}):</b> <code>{sim_result['predicted_label']}</code></p>
                <p style="margin-top:0.5rem;"><b>Plan de Acción RRHH Sugerido:</b> {sim_result['hr_action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Señales Observadas en el Perfil:**")
        for drv in sim_result["key_drivers"]:
            st.markdown(f"- {drv}")

# ==============================================================================
# TAB 5: ÉTICA EN RRHH Y RESPUESTAS A LAS 5 PREGUNTAS DE SUSTENTACIÓN (SECCIÓN 9)
# ==============================================================================
with tabs[4]:
    st.subheader("⚖️ Marco Ético, Privacidad, Sesgos y Límites de Uso en Recursos Humanos")
    eth = HR_ACADEMIC_CONTENT["ethical_and_legal_analysis"]
    st.markdown(eth["no_automated_decisions_principle"])
    st.markdown(eth["privacy_and_confidentiality"])
    st.markdown(eth["algorithmic_bias_and_fairness"])

    st.markdown("---")
    st.subheader("🎓 Respuestas Rigurosas a las 5 Preguntas Obligatorias de Sustentación (Sección 9 del PDF)")
    hr_qas = format_hr_defense_answers(hr)
    for qa in hr_qas:
        with st.expander(f"📌 {qa['question']}", expanded=True):
            st.markdown(f"**Foco de Rúbrica:** `{qa.get('rubric_focus', 'Sustentación Grupo 08')}`")
            st.markdown(qa["answer"])
