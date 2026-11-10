"""Verificación y aprovisionamiento de los datasets del proyecto."""
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.config import (
    CSV_DIR,
    DATA_DIR,
    RETAIL_FILENAME,
    HR_FILENAME,
    get_dataset_path,
)

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "ecommerce": {
        "filename": RETAIL_FILENAME,
        "kaggle_slug": "mashlyn/online-retail-ii-uci",
        "url": "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci",
        "uci_url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "author": "Daqing Chen (UCI Machine Learning Repository / London South Bank University)",
        "license": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
        "access_date": "2026-11-07",
        "unit_of_analysis": "Cliente único (agregado a partir de líneas de transacción por Customer ID)",
        "min_rows": 500_000,
        "required_columns": [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ],
    },
    "hr_attrition": {
        "filename": HR_FILENAME,
        "kaggle_slug": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "url": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "author": "IBM Watson Analytics / Pavan Subhash T (Kaggle)",
        "license": "Database Contents License (DbCL) v1.0 / Open Academic Use",
        "access_date": "2026-11-07",
        "unit_of_analysis": "Empleado individual en nómina (1 fila por colaborador)",
        "target_meaning": "Attrition: 1 = dejó la empresa (Yes), 0 = permanece en la empresa (No)",
        "min_rows": 1400,
        "required_columns": [
            "Age",
            "Attrition",
            "BusinessTravel",
            "DailyRate",
            "Department",
            "DistanceFromHome",
            "Education",
            "EducationField",
            "EnvironmentSatisfaction",
            "Gender",
            "HourlyRate",
            "JobInvolvement",
            "JobLevel",
            "JobRole",
            "JobSatisfaction",
            "MaritalStatus",
            "MonthlyIncome",
            "MonthlyRate",
            "NumCompaniesWorked",
            "OverTime",
            "PercentSalaryHike",
            "PerformanceRating",
            "RelationshipSatisfaction",
            "StockOptionLevel",
            "TotalWorkingYears",
            "TrainingTimesLastYear",
            "WorkLifeBalance",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsSinceLastPromotion",
            "YearsWithCurrManager",
        ],
    },
}


def ensure_datasets_available(sync_to_data_dir: bool = False) -> Dict[str, Path]:
    """
    Verifica que ambos datasets existan en `csv/` o `data/`.
    Si no existen, intenta descargarlos vía Kaggle CLI.
    Opcionalmente sincroniza enlaces o copias en `data/`.
    """
    resolved_paths: Dict[str, Path] = {}
    for key, meta in DATASET_METADATA.items():
        filename = meta["filename"]
        path = get_dataset_path(filename)
        if not path.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            slug = meta["kaggle_slug"]
            try:
                subprocess.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        slug,
                        "-p",
                        str(DATA_DIR),
                        "--unzip",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"No se encontró {filename} en {CSV_DIR} ni en {DATA_DIR}, "
                    f"y la descarga automática de Kaggle ({slug}) falló: {exc}"
                ) from exc
            path = get_dataset_path(filename)

        if sync_to_data_dir and path.parent != DATA_DIR:
            target = DATA_DIR / filename
            if not target.exists():
                try:
                    target.symlink_to(path)
                except OSError:
                    shutil.copy2(path, target)

        resolved_paths[key] = path
    return resolved_paths


def validate_dataset_integrity(key: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Valida columnas requeridas y conteo mínimo de filas."""
    meta = DATASET_METADATA[key]
    missing_cols = [c for c in meta["required_columns"] if c not in df.columns]
    row_count = len(df)
    is_valid = (len(missing_cols) == 0) and (row_count >= meta["min_rows"])
    return {
        "dataset": key,
        "is_valid": is_valid,
        "row_count": row_count,
        "col_count": df.shape[1],
        "missing_required_columns": missing_cols,
        "metadata": meta,
    }


if __name__ == "__main__":
    paths = ensure_datasets_available(sync_to_data_dir=True)
    for k, p in paths.items():
        print(f"[OK] {k}: {p} ({p.stat().st_size / (1024*1024):.2f} MB)")
