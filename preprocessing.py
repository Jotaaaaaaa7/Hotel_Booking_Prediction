from __future__ import annotations
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def split_features(df: pd.DataFrame, target: str = "is_canceled") -> Tuple[List[str], List[str]]:
    """
    Separa las columnas del DataFrame en numéricas y categóricas.
    El DataFrame debe estar ya limpio.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference([target]).tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.difference([target]).tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop='first'  # Equivalente a drop_first=True en pd.get_dummies()
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza inicial del dataset igual que en el notebook"""
    # Eliminar duplicados
    df_clean = df.drop_duplicates().copy()

    # Eliminar columnas no útiles
    drop_cols = ['company', 'reservation_status', 'reservation_status_date']
    df_clean = df_clean.drop(columns=drop_cols, errors='ignore')

    # Imputar valores
    df_clean['children'] = df_clean['children'].fillna(df_clean['children'].median())
    df_clean['country'] = df_clean['country'].fillna(df_clean['country'].mode()[0])
    df_clean['agent'] = df_clean['agent'].fillna(0)

    return df_clean