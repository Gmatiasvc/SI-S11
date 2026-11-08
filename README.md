# 🧠 SI-S11: Plataforma Unificada de Inteligencia de Negocios y Analítica Avanzada

Plataforma interactiva y modular desarrollada en **Python + Streamlit + Scikit-Learn + Plotly** para el curso **Sistemas Inteligentes y Machine Learning**, que unifica en una sola interfaz ejecutiva los dos trabajos aplicativos:

1. **Módulo 1 (Aprendizaje No Supervisado — Trabajo Aplicativo 01):** Segmentación Estratégica de Clientes de Comercio Electrónico (*UCI Online Retail II*: 1,067,371 transacciones $\rightarrow$ 5,878 clientes únicos agregados por `Customer ID` con 6 variables cuantitativas RFM enriquecidas, comparando rigurosamente **K-Means**, **DBSCAN**, **PCA + K-Means** y **PCA + DBSCAN**).
2. **Módulo 2 (Aprendizaje Supervisado — Trabajo Aplicativo Grupo 08):** Predicción Preventiva de Rotación de Empleados (*IBM HR Analytics Employee Attrition & Performance*: 1,470 colaboradores, 16 etapas obligatorias, prevención estricta de *data leakage* mediante `Pipeline` + `ColumnTransformer`, validación cruzada estratificada en los mismos 5 folds y optimización de **Decision Tree**, **Random Forest** y **Gradient Boosting** con F1 de clase positiva).

---

## 🏗️ Estructura del Proyecto

```text
SI-S11/
├── .gitignore
├── requirements.txt
├── README.md
├── plan/
│   └── design_doc_consolidado.md          # Documento de diseño arquitectónico y metodológico
├── pdfs/
│   ├── Trabajo_01_segmentacion_de_clientes_de_comercio_electronico.pdf
│   └── Grupo_08_Rotacion_Empleados.pdf
├── csv/ & data/                           # Datasets locales (ignorados en git por .gitignore)
│   ├── online_retail_II.csv
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── models/                                # Artefactos entrenados serializados (.joblib y .json)
│   ├── kmeans_rfm_pipeline.joblib
│   ├── ecommerce_clustering_results.joblib
│   ├── ecommerce_evaluation_results.json
│   ├── hr_best_pipeline.joblib
│   ├── hr_all_pipelines.joblib
│   └── hr_evaluation_results.json
├── src/
│   ├── config.py                          # Semillas (random_state=42), rutas y constantes
│   ├── data/
│   │   ├── download_datasets.py           # Verificación de integridad y descarga automática
│   │   └── loaders.py                     # Carga tipada y resúmenes exploratorios (EDA)
│   ├── ecommerce/                         # Motor No Supervisado (10 Etapas del Trabajo 01)
│   │   ├── feature_engineering.py         # RFM + AvgTicket + UniqueCategories + CancellationRate
│   │   ├── pca_transform.py               # Scree plot, varianza >= 90%, cargas y MSE reconstrucción
│   │   ├── clustering.py                  # K-Means (K=2..10), curva k-distances y DBSCAN
│   │   ├── evaluation.py                  # Comparación justa en espacio original y estabilidad ARI
│   │   └── business_insights.py           # Perfiles en unidades reales y 3 estrategias comerciales
│   ├── hr_attrition/                      # Motor Supervisado (16 Etapas del Grupo 08)
│   │   ├── preprocessor.py                # Limpieza sin fuga, split 80/20 estratificado y ColumnTransformer
│   │   ├── models.py                      # Pipelines para DecisionTree, RandomForest y GradientBoosting
│   │   ├── training.py                    # 5-Fold StratifiedKFold compartido, RandomizedSearchCV y umbral OOF
│   │   ├── evaluation.py                  # Evaluación única en Test, MDI/Permutation e impacto por antigüedad
│   │   └── inference.py                   # Inferencia sobre 5 perfiles ficticios y simulador en tiempo real
│   └── utils/
│       ├── metrics_logger.py              # Codificador JSON científico
│       └── academic_content.py            # Diccionarios de datos, ética en RRHH y respuestas de sustentación
├── app/
│   ├── main.py                            # Entrada principal Streamlit Multipage
│   ├── styles/
│   │   └── custom.css                     # Diseño visual corporativo (Dark Slate & Indigo)
│   └── pages/
│       ├── 1_📊_Overview.py               # Dashboard ejecutivo integrador
│       ├── 2_🛒_Customer_Segmentation.py  # Interfaz interactiva de Comercio Electrónico (Trabajo 01)
│       └── 3_👥_HR_Employee_Attrition.py  # Interfaz interactiva y Simulador de Rotación RRHH (Grupo 08)
└── tests/
    └── test_pipelines.py                  # Suite de pruebas automatizadas con pytest
```

---

## 🚀 Instrucciones de Instalación y Ejecución

### 1. Activar el entorno virtual e instalar dependencias
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación interactiva en Streamlit
```bash
.venv/bin/streamlit run app/main.py
```

### 3. (Opcional) Re-entrenar modelos desde cero o ejecutar pruebas de auditoría QA
```bash
# Ejecutar suite de tests automatizados (auditoría de no-leakage, folds compartidos y artefactos)
.venv/bin/pytest -v

# Regenerar artefactos de Comercio Electrónico y RRHH
.venv/bin/python -c "from src.ecommerce.business_insights import run_full_ecommerce_pipeline; run_full_ecommerce_pipeline(force_retrain=True)"
.venv/bin/python -c "from src.hr_attrition.inference import run_full_hr_pipeline; run_full_hr_pipeline(force_retrain=True)"
```

---

## 📊 Resumen de Resultados Principales

### Módulo 1: Segmentación de Clientes de Comercio Electrónico (Trabajo 01)
- **Población analizada:** **5,878 clientes únicos** con compras válidas (agregados desde 1,067,371 registros).
- **PCA:** Con **$d = 4$ componentes principales** se captura el **93.62%** de la varianza total con un error cuadrático medio de reconstrucción de `0.0638`.
- **Alternativa seleccionada:** **K-Means ($K=4$, `k-means++`, `n_init=20`)** sobre variables RFM enriquecidas transformadas (`log1p` + `StandardScaler`), alcanzando cobertura del 100%, estabilidad entre semillas **ARI = 0.9749** y concordancia con `PCA + K-Means` de **ARI = 0.9005**, complementado con **DBSCAN ($\varepsilon=0.85, \text{min\_samples}=5$)** para aislar **226 clientes atípicos (3.84%)** mayoristas y de alta cancelación.

### Módulo 2: Predicción de Rotación de Empleados — HR Analytics (Grupo 08)
- **Partición sin fuga:** 1,176 empleados en Train (80%) y 294 en Test reservado (20%), manteniendo 16.1% de clase positiva.
- **Comparación en los mismos 5 folds de validación cruzada (`StratifiedKFold`):**
  - `DecisionTree`: CV Inicial F1 = `0.2835` $\rightarrow$ Mejor CV F1 = `0.4342` (OOF $\tau^*=0.44 \rightarrow 0.4393$, ROC-AUC = `0.6630`).
  - `RandomForest`: CV Inicial F1 = `0.4378` $\rightarrow$ Mejor CV F1 = `0.5023` (OOF $\tau^*=0.49 \rightarrow 0.5136$, ROC-AUC = `0.7884`).
  - **`GradientBoosting` (Ganador):** CV Inicial F1 = `0.4707` $\rightarrow$ Mejor CV F1 = `0.5157` (OOF $\tau^*=0.30 \rightarrow \mathbf{0.5470}$, ROC-AUC = **`0.8220`**, PR-AUC = **`0.5916`**).
- **Evaluación única en Test reservado:** ROC-AUC = **`0.8082`**, PR-AUC = **`0.5069`**, F1 con umbral calibrado ($\tau^*=0.30$) = **`0.4198`** (superando al F1 con umbral por defecto $0.50$ de `0.3636`).
