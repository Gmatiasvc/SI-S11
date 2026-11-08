"""Etapas 04, 05, 06 y 07 del Grupo 08: Limpieza inicial, separación X/y, split estratificado y ColumnTransformer."""
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import HR_DROP_COLUMNS, HR_TARGET_COL, RANDOM_STATE, TEST_SIZE


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Etapas 04 y 05:
    - Elimina columnas sin varianza o identificadores ('EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours').
    - Separa X e y = (df['Attrition'] == 'Yes').astype(int).
    - Identifica columnas numéricas y categóricas disponibles antes del evento de salida.
    """
    df_clean = df.copy()
    cols_to_drop = [c for c in HR_DROP_COLUMNS if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    if HR_TARGET_COL not in df_clean.columns:
        raise ValueError(f"La columna objetivo '{HR_TARGET_COL}' no está presente en el DataFrame.")

    y = (df_clean[HR_TARGET_COL].astype(str).str.strip() == "Yes").astype(int)
    X = df_clean.drop(columns=[HR_TARGET_COL]).copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    for col in categorical_cols:
        X[col] = X[col].astype(object)

    return X, y, numeric_cols, categorical_cols


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Etapa 06:
    Divide el dataset en Train (80%) y Test (20%) de forma reproducible y estratificada por clase.
    El conjunto Test se reserva intacto hasta la Etapa 13.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Etapa 07:
    Construye el preprocesador ColumnTransformer sin fuga de datos (se ajusta únicamente sobre Train):
    - Numéricas: SimpleImputer(strategy='median') + StandardScaler()
    - Categóricas: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
