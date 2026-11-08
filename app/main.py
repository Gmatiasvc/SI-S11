"""Punto de entrada principal de la aplicación Streamlit Multipage (SI-S11)."""
import sys
from pathlib import Path
import streamlit as st

# Asegurar que la raíz del repositorio esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="SI-S11 | Business Intelligence & ML Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    """Inyecta la hoja de estilos corporativa custom.css."""
    css_path = Path(__file__).resolve().parent / "styles" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css()

# Definir navegación multipágina moderna con st.navigation
pages_dir = Path(__file__).resolve().parent / "pages"
overview_page = st.Page(
    str(pages_dir / "1_📊_Overview.py"),
    title="Dashboard Ejecutivo Consolidado",
    icon="📊",
    default=True,
)
ecommerce_page = st.Page(
    str(pages_dir / "2_🛒_Customer_Segmentation.py"),
    title="Módulo 1: Segmentación E-Commerce (No Supervisado)",
    icon="🛒",
)
hr_page = st.Page(
    str(pages_dir / "3_👥_HR_Employee_Attrition.py"),
    title="Módulo 2: Retención y Rotación RRHH (Supervisado)",
    icon="👥",
)

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.75rem 0.6rem 1rem 0.6rem; border-bottom: 1.5px solid #CBD5E1; margin-bottom: 0.8rem; background: #E2E8F0; border-radius: 10px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #4338CA; text-transform: uppercase; letter-spacing: 0.08em;">
                Sistemas Inteligentes · S11
            </div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-top: 0.2rem;">
                🧠 Analítica Unificada
            </div>
            <div style="font-size: 0.8rem; color: #334155; font-weight: 500; margin-top: 0.2rem;">
                E-Commerce RFM + HR Attrition ML
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

nav = st.navigation(
    {
        "Visión Ejecutiva": [overview_page],
        "Casos de Estudio": [ecommerce_page, hr_page],
    }
)

with st.sidebar:
    st.markdown("---")
    st.caption(
        "**Arquitectura Reproducible**\n\n"
        "• Semilla global fija: `random_state = 42`\n"
        "• Sin *Data Leakage*: `Pipeline` + `ColumnTransformer`\n"
        "• Persistencia de modelos: `joblib` + `JSON`"
    )

nav.run()
