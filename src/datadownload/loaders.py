"""Funciones de carga tipada y resúmenes exploratorios (EDA) para ambos datasets."""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.config import (
    RETAIL_FILENAME,
    HR_FILENAME,
    HR_DROP_COLUMNS,
    HR_TARGET_COL,
    get_dataset_path,
)
from src.datadownload.download_datasets import ensure_datasets_available, DATASET_METADATA


def load_raw_retail(nrows: int | None = None) -> pd.DataFrame:
    """
    Carga el dataset Online Retail II desde csv/ o data/ con manejo de codificación
    y conversión de tipos básica.
    """
    ensure_datasets_available(sync_to_data_dir=True)
    path = get_dataset_path(RETAIL_FILENAME)
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        nrows=nrows,
        dtype={
            "Invoice": str,
            "StockCode": str,
            "Description": str,
            "Country": str,
        },
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df


def load_raw_hr() -> pd.DataFrame:
    """
    Carga el dataset IBM HR Employee Attrition eliminando caracteres BOM (utf-8-sig)
    y espacios en los nombres de columnas.
    """
    ensure_datasets_available(sync_to_data_dir=True)
    path = get_dataset_path(HR_FILENAME)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def get_retail_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera el resumen exploratorio obligatorio de la Etapa 2 del Trabajo 01:
    shape, tipos, faltantes, duplicados, cancelaciones, estadísticas descriptivas y 4 hallazgos clave.
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isna().sum().to_dict()
    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    n_duplicates = int(df.duplicated().sum())
    is_cancelled = df["Invoice"].astype(str).str.upper().str.startswith("C")
    n_cancelled = int(is_cancelled.sum())
    n_negative_price = int((df["Price"] <= 0).sum())
    n_unique_customers = int(df["Customer ID"].dropna().nunique())
    n_countries = int(df["Country"].dropna().nunique())
    date_min = str(df["InvoiceDate"].min())
    date_max = str(df["InvoiceDate"].max())

    numeric_desc = (
        df[["Quantity", "Price"]]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .round(2)
        .to_dict()
    )

    top_countries = (
        df["Country"]
        .value_counts()
        .head(10)
        .rename_axis("Country")
        .reset_index(name="Transactions")
        .to_dict(orient="records")
    )

    findings = [
        (
            f"1. Faltantes concentrados en identificador de cliente: De {n_rows:,} registros transaccionales, "
            f"{missing_counts.get('Customer ID', 0):,} ({missing_pct.get('Customer ID', 0):.2f}%) carecen de 'Customer ID'. "
            "Al ser la unidad de análisis el cliente individual, estos registros de compras de invitados (guest checkout) "
            "no pueden atribuirse a un historial RFM individual y deben excluirse de la agregación por cliente."
        ),
        (
            f"2. Presencia de devoluciones/cancelaciones y ajustes contables: Se detectaron {n_cancelled:,} líneas de factura "
            f"con prefijo 'C' (cancelaciones con Quantity negativa) y {n_negative_price:,} registros con Price <= 0. "
            "En lugar de descartar ciegamente las cancelaciones, las contabilizamos para construir la variable derivada "
            "'Tasa de Cancelación/Descuento Ajustado' por cliente antes de calcular el gasto neto en compras válidas."
        ),
        (
            f"3. Asimetría extrema positiva (colas pesadas): Tanto Quantity como Price presentan valores extremos "
            "producto de compradores mayoristas (B2B) conviviendo con compradores minoristas (B2C). Esto sesga la media "
            "muy por encima de la mediana y hace indispensable aplicar transformación logarítmica (log1p) previa a StandardScaler."
        ),
        (
            f"4. Concentración geográfica y amplitud temporal: El dataset abarca desde {date_min[:10]} hasta {date_max[:10]} "
            f"con {n_unique_customers:,} clientes únicos distribuidos en {n_countries} países, donde Reino Unido concentra "
            "más del 90% de las transacciones."
        ),
    ]

    return {
        "shape": {"rows": n_rows, "cols": n_cols},
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "n_duplicates": n_duplicates,
        "n_cancelled": n_cancelled,
        "n_negative_price": n_negative_price,
        "n_unique_customers": n_unique_customers,
        "n_countries": n_countries,
        "date_range": {"min": date_min, "max": date_max},
        "numeric_describe": numeric_desc,
        "top_countries": top_countries,
        "findings": findings,
        "metadata": DATASET_METADATA["ecommerce"],
    }


def get_hr_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera el resumen exploratorio obligatorio de las Etapas 02-05 del Grupo 08:
    tipos, faltantes, duplicados, distribución de clases, variables sin varianza/identificadores y correlaciones.
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isna().sum().to_dict()
    n_duplicates = int(df.duplicated().sum())
    target_counts = df[HR_TARGET_COL].value_counts().to_dict()
    target_pct = (df[HR_TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict()

    constant_or_id_cols = {
        "EmployeeNumber": "Identificador único secuencial sin significado predictivo (riesgo de sobreajuste/memorización).",
        "EmployeeCount": f"Constante sin varianza (valor único = {df['EmployeeCount'].nunique()}).",
        "Over18": f"Constante sin varianza (valor único = {df['Over18'].nunique()}, todos 'Y').",
        "StandardHours": f"Constante sin varianza (valor único = {df['StandardHours'].nunique()}, todos 80).",
    }

    df_clean = df.drop(columns=[c for c in HR_DROP_COLUMNS if c in df.columns]).copy()
    y_bin = (df_clean[HR_TARGET_COL] == "Yes").astype(int)
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [
        c for c in df_clean.select_dtypes(exclude=[np.number]).columns if c != HR_TARGET_COL
    ]

    correlations = (
        df_clean[num_cols]
        .apply(lambda s: s.corr(y_bin))
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .round(4)
        .to_dict()
    )

    return {
        "shape": {"rows": n_rows, "cols": n_cols},
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_total": int(sum(missing_counts.values())),
        "missing_counts": missing_counts,
        "n_duplicates": n_duplicates,
        "target_counts": target_counts,
        "target_pct": target_pct,
        "dropped_columns_justification": constant_or_id_cols,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "top_numeric_correlations_with_attrition": correlations,
        "metadata": DATASET_METADATA["hr_attrition"],
    }
