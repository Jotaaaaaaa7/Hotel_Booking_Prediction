from __future__ import annotations
import os, argparse, json, logging, warnings           # ➊  warnings + os
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from data_loader import load_data
from preprocessing import split_features, build_preprocessor, clean_dataset
from models import get_base_pipelines
from tuner import build_searchers


# Eliminar Warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Found unknown categories.*encoded as all zeros",
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("trainer")

def evaluate(y_true, y_pred, y_prob) -> dict:
    """
    Evalúa las métricas de clasificación y devuelve un diccionario con los resultados.
    :param y_true: Valores verdaderos de la variable objetivo.
    :param y_pred: Predicciones del modelo (0 o 1).
    :param y_prob: Probabilidades predichas por el modelo (0.0 a 1.0).
    :return: dict con las métricas de evaluación.
    """
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred),
        recall    = recall_score(y_true, y_pred),
        f1        = f1_score(y_true, y_pred),
        roc_auc   = roc_auc_score(y_true, y_prob),
    )


def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de la línea de comandos.
    :return: Objeto Namespace con los argumentos.
    """
    p = argparse.ArgumentParser("Trainer notebooks → .joblib (sin warnings)")
    p.add_argument("--data", required=True, help="CSV original sin procesar")
    p.add_argument("--outdir", default="outputs", help="Carpeta de salida")
    return p.parse_args()


def main():
    """
    Función principal que ejecuta el flujo de entrenamiento:
    1. Carga y limpieza de datos.
    2. División en conjuntos de entrenamiento y prueba.
    3. Preprocesamiento de datos.
    4. Definición de pipelines y búsqueda de hiperparámetros.
    5. Entrenamiento de modelos y evaluación.
    6. Guardado de modelos y métricas.
    """

    # Cargamos los argumentos de la línea de comandos
    args = parse_args()

    # Directorio donde se guardarán los modelos y métricas
    outdir = Path(args.outdir)

    # Creamos el directorio de salida si no existe
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Salida en %s", outdir.resolve())

    # Cargamos el dataset, lo limpiamos y separamos en X e y
    df = clean_dataset(load_data(args.data))
    y  = df["is_canceled"]
    X  = df.drop(columns=["is_canceled"])

    # Dividimos en datos de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Cargamos las columnas numéricas y categóricas, construimos el preprocesador
    # Cargamos los pipelines base y los objetos de búsqueda de hiperparámetros
    num_cols, cat_cols = split_features(X_train)
    preproc   = build_preprocessor(num_cols, cat_cols)
    pipelines = get_base_pipelines(preproc)
    searchers = build_searchers(pipelines)

    # Inicializamos un diccionario para almacenar los resultados
    summary = {}

    # Iteramos sobre los modelos, entrenamos y evaluamos
    for name, searcher in searchers.items():
        logger.info("🔧 Entrenando %s …", name)

        # Ajustamos los datos de train y test al GridSearchCV o RandomizedSearchCV correspondiente del modelo
        searcher.fit(X_train, y_train)

        # # Obtenemos el mejor modelo del GridSearchCV o RandomizedSearchCV
        # best = searcher.best_estimator_ if hasattr(searcher, "best_estimator_") else searcher

        # Obtenemos el mejor modelo del GridSearchCV o RandomizedSearchCV
        best = searcher.best_estimator_ if hasattr(searcher, "best_estimator_") else searcher

        if name in ["decision_tree", "random_forest", "xgboost"]:  # LR suele venir bien calibrada
            # 1) reservamos un 15 % de train para calibrar
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
            )
            best.fit(X_fit, y_fit)  # volvemos a entrenar sólo con X_fit
            best = CalibratedClassifierCV(
                best, method="sigmoid", cv="prefit"  # o "isotonic" si tienes >10 k filas
            ).fit(X_cal, y_cal)

        # Después de calibrar el modelo 'best'
        y_prob = best.predict_proba(X_test)[:, 1]  # Obtener probabilidades de la clase positiva
        y_pred = (y_prob >= 0.5).astype(int)       # Convertir probabilidades a predicciones binarias

        # Evaluamos las métricas de clasificación
        metrics = evaluate(y_test, y_pred, y_prob)

        # Actualizamos el resumen con las métricas y los mejores parámetros
        summary[name] = {
            "metrics": metrics,
            "best_params": getattr(searcher, "best_params_", None),
        }

        logger.info("✅ %s – AUC: %.4f", name, metrics["roc_auc"])

        # Guardamos el mejor modelo de cada pipeline en archivos .joblib
        joblib.dump(best, outdir / f"{name}_model.joblib", compress=3)

    # Generamos una marca de tiempo
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardamos las métricas en un JSON y registramos la acción
    with open(outdir / f"metrics_{ts}.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Métricas guardadas en metrics_%s.json", ts)


if __name__ == "__main__":
    main()


#* python trainer.py --data <ruta al CSV sin procesar>
