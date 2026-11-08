"""Ingeniería de características RFM enriquecida y preprocesamiento compartido (Etapas 2 y 3)."""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: List[str] = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgTicket",
    "UniqueCategories",
    "CancellationRate",
]

FEATURE_DICTIONARY: Dict[str, Dict[str, str]] = {
    "Recency": {
        "nombre_esp": "Recencia (días)",
        "formula": "(T_ref - max(InvoiceDate_i)).days, donde T_ref = max(InvoiceDate) + 1 día",
        "justificacion": "Mide la frescura de la relación comercial; distingue clientes activos de clientes dormidos o en riesgo de abandono.",
    },
    "Frequency": {
        "nombre_esp": "Frecuencia de compra (facturas)",
        "formula": "count_distinct(Invoice) de compras válidas por cliente",
        "justificacion": "Cuantifica la recurrencia y lealtad transaccional del cliente a lo largo del horizonte temporal.",
    },
    "Monetary": {
        "nombre_esp": "Valor Monetario Total (£)",
        "formula": "sum(Quantity * Price) en transacciones válidas",
        "justificacion": "Representa el aporte directo de ingresos brutos de cada cliente (Customer Lifetime Value histórico observado).",
    },
    "AvgTicket": {
        "nombre_esp": "Ticket Promedio por Pedido (£)",
        "formula": "Monetary / Frequency",
        "justificacion": "Desacopla el volumen total del tamaño medio de orden, permitiendo separar compradores mayoristas (B2B) de minoristas frecuentes.",
    },
    "UniqueCategories": {
        "nombre_esp": "Variedad de Catálogo (SKUs distintos)",
        "formula": "count_distinct(StockCode) en compras válidas",
        "justificacion": "Evalúa la amplitud de exploración del catálogo (cross-buying) frente a compras concentradas en un único producto.",
    },
    "CancellationRate": {
        "nombre_esp": "Tasa de Cancelación / Devolución",
        "formula": "max(facturas_canceladas / total_facturas, lineas_canceladas / total_lineas) en [0, 1]",
        "justificacion": "Proxy verificable de fricción post-venta, devoluciones o descuentos por nota de crédito ('C' prefix en UCI Online Retail II).",
    },
}


def build_customer_rfm_dataset(
    df_raw: pd.DataFrame,
    winsor_quantile: float = 0.995,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict[str, Any]]:
    """
    Construye el dataset agregado por cliente con variables RFM + Ticket Promedio +
    Variedad de Catálogo + Tasa de Cancelación/Devolución, aplica tratamiento de
    asimetría/atípicos (winsorización suave al 99.5% + log1p) y ajusta el
    StandardScaler compartido para garantizar una comparación justa de las 4 alternativas.
    """
    n_total_rows = int(len(df_raw))

    # 1. Filtrar filas con Customer ID válido
    df_with_cid = df_raw.dropna(subset=["Customer ID", "InvoiceDate"]).copy()
    df_with_cid["Customer ID"] = (
        df_with_cid["Customer ID"].astype(float).astype(int).astype(str)
    )
    n_rows_with_cid = int(len(df_with_cid))
    n_dropped_no_cid = n_total_rows - n_rows_with_cid

    # 2. Contabilizar actividad total y cancelaciones por cliente antes de filtrar compras válidas
    invoice_str = df_with_cid["Invoice"].astype(str).str.strip().str.upper()
    df_with_cid["is_cancelled"] = invoice_str.str.startswith("C") | (
        df_with_cid["Quantity"] < 0
    )

    total_invoices_per_cust = df_with_cid.groupby("Customer ID")["Invoice"].nunique()
    cancelled_df = df_with_cid[df_with_cid["is_cancelled"]]
    cancelled_invoices_per_cust = cancelled_df.groupby("Customer ID")["Invoice"].nunique()

    total_lines_per_cust = df_with_cid.groupby("Customer ID").size()
    cancelled_lines_per_cust = cancelled_df.groupby("Customer ID").size()

    # 3. Filtrar compras válidas: no empiezan con 'C', Quantity > 0, Price > 0
    valid_mask = (
        (~df_with_cid["is_cancelled"])
        & (df_with_cid["Quantity"] > 0)
        & (df_with_cid["Price"] > 0)
    )
    df_valid = df_with_cid[valid_mask].copy()
    df_valid["LineTotal"] = df_valid["Quantity"] * df_valid["Price"]
    n_valid_rows = int(len(df_valid))

    # 4. Fecha de referencia T_ref = max(InvoiceDate) + 1 día
    max_date = df_valid["InvoiceDate"].max()
    t_ref = max_date + pd.Timedelta(days=1)

    # 5. Agregación por Customer ID
    grouped = df_valid.groupby("Customer ID").agg(
        last_purchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
        UniqueCategories=("StockCode", "nunique"),
        TotalQuantity=("Quantity", "sum"),
        Country=("Country", "first"),
    )

    grouped["Recency"] = (t_ref - grouped["last_purchase"]).dt.days.astype(float)
    grouped["Frequency"] = grouped["Frequency"].astype(float)
    grouped["Monetary"] = grouped["Monetary"].astype(float).round(2)
    grouped["AvgTicket"] = (grouped["Monetary"] / grouped["Frequency"]).round(2)
    grouped["UniqueCategories"] = grouped["UniqueCategories"].astype(float)

    cust_total_inv = total_invoices_per_cust.reindex(grouped.index).fillna(1).astype(float)
    cust_canc_inv = cancelled_invoices_per_cust.reindex(grouped.index).fillna(0.0).astype(float)
    cust_total_lines = total_lines_per_cust.reindex(grouped.index).fillna(1).astype(float)
    cust_canc_lines = cancelled_lines_per_cust.reindex(grouped.index).fillna(0.0).astype(float)

    invoice_cancel_rate = (cust_canc_inv / cust_total_inv).clip(0.0, 1.0)
    line_cancel_rate = (cust_canc_lines / cust_total_lines).clip(0.0, 1.0)
    grouped["CancellationRate"] = np.maximum(invoice_cancel_rate, line_cancel_rate).round(4)

    df_rfm = grouped[FEATURE_COLUMNS + ["TotalQuantity", "Country"]].copy()
    df_rfm.index.name = "Customer ID"

    # 6. Identificación explícita de clientes atípicos extremos (percentil 99.5%)
    winsor_caps: Dict[str, float] = {}
    skew_before: Dict[str, float] = {}
    skew_after: Dict[str, float] = {}

    monetary_p995 = float(df_rfm["Monetary"].quantile(winsor_quantile))
    frequency_p995 = float(df_rfm["Frequency"].quantile(winsor_quantile))
    avgticket_p995 = float(df_rfm["AvgTicket"].quantile(winsor_quantile))

    extreme_mask = (
        (df_rfm["Monetary"] >= monetary_p995)
        | (df_rfm["Frequency"] >= frequency_p995)
        | (df_rfm["AvgTicket"] >= avgticket_p995)
    )
    df_rfm["IsExtremeOutlier"] = extreme_mask

    extreme_customers_df = (
        df_rfm[extreme_mask]
        .sort_values(by="Monetary", ascending=False)
        .head(20)
        .reset_index()
    )

    # 7. Winsorización suave al percentil 99.5% + log1p
    X_winsor = df_rfm[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        skew_before[col] = round(float(df_rfm[col].skew()), 3)
        cap_val = float(df_rfm[col].quantile(winsor_quantile))
        winsor_caps[col] = round(cap_val, 4)
        X_winsor[col] = np.clip(X_winsor[col], a_min=0.0, a_max=cap_val)

    X_log = np.log1p(X_winsor)
    for col in FEATURE_COLUMNS:
        skew_after[col] = round(float(X_log[col].skew()), 3)

    # 8. StandardScaler compartido
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_log)
    X_scaled = pd.DataFrame(
        X_scaled_arr,
        index=df_rfm.index,
        columns=FEATURE_COLUMNS,
    )

    corr_original = df_rfm[FEATURE_COLUMNS].corr().round(4).to_dict()
    corr_scaled = X_scaled.corr().round(4).to_dict()

    summary_stats_original = (
        df_rfm[FEATURE_COLUMNS]
        .describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .round(3)
        .to_dict()
    )

    prep_metadata: Dict[str, Any] = {
        "n_raw_rows": n_total_rows,
        "n_rows_with_customer_id": n_rows_with_cid,
        "n_dropped_missing_customer_id": n_dropped_no_cid,
        "n_valid_purchase_rows": n_valid_rows,
        "n_unique_customers": int(len(df_rfm)),
        "t_ref": str(t_ref),
        "feature_columns": FEATURE_COLUMNS,
        "feature_dictionary": FEATURE_DICTIONARY,
        "winsor_quantile": winsor_quantile,
        "winsor_caps": winsor_caps,
        "skewness_before_log1p": skew_before,
        "skewness_after_log1p": skew_after,
        "scaler_means": {
            col: round(float(m), 4) for col, m in zip(FEATURE_COLUMNS, scaler.mean_)
        },
        "scaler_scales": {
            col: round(float(s), 4) for col, s in zip(FEATURE_COLUMNS, scaler.scale_)
        },
        "n_extreme_outliers_p995": int(extreme_mask.sum()),
        "top_extreme_customers": extreme_customers_df[
            ["Customer ID", "Recency", "Frequency", "Monetary", "AvgTicket", "UniqueCategories", "CancellationRate", "Country"]
        ].to_dict(orient="records"),
        "corr_original": corr_original,
        "corr_scaled": corr_scaled,
        "summary_stats_original": summary_stats_original,
        "preprocessing_decisions": [
            "Exclusión justificada de registros sin Customer ID (compras de invitado no trazables a un cliente único).",
            "Separación de facturas canceladas (prefijo 'C' o Quantity < 0) para construir CancellationRate antes de filtrar compras válidas (Quantity > 0, Price > 0).",
            "Cálculo de 6 variables cuantitativas a nivel de cliente: Recency, Frequency, Monetary, AvgTicket, UniqueCategories y CancellationRate.",
            f"Winsorización suave al percentil {winsor_quantile*100:.1f}% antes de log1p para acotar distorsión de compradores mayoristas extremos (preservando sus valores reales en df_rfm).",
            "Transformación logarítmica np.log1p(X) que corrige la asimetría positiva en variables monetarias y de frecuencia.",
            "Ajuste de un único StandardScaler compartido (X_scaled) para evaluar bajo las mismas condiciones K-Means, DBSCAN, PCA+K-Means y PCA+DBSCAN.",
        ],
    }

    return df_rfm, X_scaled, scaler, prep_metadata
