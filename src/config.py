"""Configuración centralizada del proyecto SI-S11."""
from pathlib import Path

# Directorios raíz
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "csv"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Asegurar que existan directorios clave
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Nombres de archivos de datasets
RETAIL_FILENAME = "online_retail_II.csv"
HR_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def get_dataset_path(filename: str) -> Path:
    """Devuelve la ruta existente del dataset buscando primero en csv/ y luego en data/."""
    csv_candidate = CSV_DIR / filename
    if csv_candidate.exists():
        return csv_candidate
    data_candidate = DATA_DIR / filename
    return data_candidate


RETAIL_CSV_PATH = get_dataset_path(RETAIL_FILENAME)
HR_CSV_PATH = get_dataset_path(HR_FILENAME)

# Rutas de artefactos serializados (.joblib y .json)
HR_BEST_PIPELINE_PATH = MODELS_DIR / "hr_best_pipeline.joblib"
HR_ALL_PIPELINES_PATH = MODELS_DIR / "hr_all_pipelines.joblib"
HR_METRICS_JSON_PATH = MODELS_DIR / "hr_evaluation_results.json"

ECOMMERCE_PIPELINE_PATH = MODELS_DIR / "kmeans_rfm_pipeline.joblib"
ECOMMERCE_RESULTS_PATH = MODELS_DIR / "ecommerce_clustering_results.joblib"
ECOMMERCE_METRICS_JSON_PATH = MODELS_DIR / "ecommerce_evaluation_results.json"

# Semillas y constantes de reproducibilidad
RANDOM_STATE = 42
STABILITY_SEEDS = [10, 42, 99, 123, 2024]

# Parámetros Módulo 1 (Comercio Electrónico - No Supervisado)
KMEANS_K_RANGE = list(range(2, 11))
KMEANS_N_INIT = 20
PCA_VARIANCE_THRESHOLD = 0.90  # Objetivo 90-95% varianza acumulada

# Parámetros Módulo 2 (RRHH Rotación - Supervisado)
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 24  # >= 20 configuraciones aleatorias por modelo según rúbrica

# Columnas a descartar en HR Attrition por ser constantes o identificadores (sin valor predictivo / fuga)
HR_DROP_COLUMNS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
]
HR_TARGET_COL = "Attrition"
