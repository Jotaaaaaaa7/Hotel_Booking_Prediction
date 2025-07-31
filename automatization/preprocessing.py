from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


def split_features(df: pd.DataFrame, target: str = "is_canceled") -> Tuple[List[str], List[str]]:
    """
    Separa las columnas numéricas y categóricas del DataFrame.
    :param df: DataFrame con los datos.
    :param target: Nombre de la columna objetivo (predicción).
    :return: Tuple con listas de columnas numéricas y categóricas.
    """
    num = df.select_dtypes(include=["number"]).columns.difference([target]).tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.difference([target]).tolist()
    logger.info(f"{len(num)} columnas numéricas: {', '.join(num)}")
    logger.info(f"{len(cat)} columnas categóricas: {', '.join(cat)}")
    return num, cat


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Construye un preprocesador para transformar datos numéricos y categóricos.
    :param num_cols: Lista de columnas numéricas.
    :param cat_cols: Lista de columnas categóricas.
    :return: ColumnTransformer configurado.
    """

    # Rellenamos NaN con la media en las columnas numéricas
    # y aplicamos estandarización (media=0, varianza=1)
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    # Rellenamos NaN con el valor más frecuente en las columnas categóricas
    # y aplicamos codificación one-hot (sin la primera categoría para evitar multicolinealidad)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ]
    )

    logger.info("Preprocesador construido")

    # Combinamos ambos pipelines en un ColumnTransformer
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando duplicados y columnas innecesarias,
    :param df: DataFrame original con datos de reservas.
    :return: DataFrame limpio y preprocesado.
    """
    df = df.drop_duplicates().copy() # Fuera duplicados

    # Company tiene 94% de valores nulos, y las otras 2 no aportan información relevante para la predicción
    df = df.drop(columns=["company", "reservation_status", "reservation_status_date"], errors="ignore")

    # Rellenamos valores nulos con la mediana en children (numérica) y valor más frecuente en country (categórica)
    df["children"] = df["children"].fillna(df["children"].median())
    df["country"] = df["country"].fillna(df["country"].mode()[0])
    df["agent"] = df["agent"].fillna(0) # Rellenamos NaN en agent con 0 (sin agente asignado)

    logger.info("Dataset limpio → %s filas × %s columnas", df.shape[0], df.shape[1])
    return df
