from pathlib import Path
import argparse, logging
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from Automatization.data_loader import load_data
from Automatization.preprocessing import clean_dataset, split_features

# Definimos el logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mlflow_eval")

def get_args():
    """
    Parsea los argumentos de la línea de comandos para evaluar modelos en MLflow.
    :return: argparse.Namespace con los argumentos.
    """
    p = argparse.ArgumentParser("Evalúa modelos .joblib en MLflow")
    p.add_argument("--data",   required=True, help="CSV crudo (mismo usado al entrenar)")
    p.add_argument("--models", default="outputs", help="Carpeta con *_model.joblib")
    return p.parse_args()

def evaluate(y_true, y_pred, y_prob) -> dict:
    """
    Evalúa las métricas de clasificación y devuelve un diccionario con los resultados.
    :param y_true: Valores verdaderos de la variable objetivo.
    :param y_pred: Predicciones del modelo (0 o 1).
    :param y_prob: Probabilidades predichas por el modelo (0.0 a 1.0).
    :return:
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred),
        recall    = recall_score(y_true, y_pred),
        f1        = f1_score(y_true, y_pred),
        roc_auc   = roc_auc_score(y_true, y_prob),
        pr_auc    = auc(rec, prec),
    )

def main():
    """
    Función principal para evaluar modelos guardados en MLflow.
    :return: None
    """

    # Cargamos los argumentos de la línea de comandos
    args = get_args()
    data_path   = Path(args.data)
    models_dir  = Path(args.models)

    # Cargamos l,os datos y los limpiamos
    df_raw   = load_data(data_path)
    df_clean = clean_dataset(df_raw)

    # Separamos las características y la variable objetivo
    y = df_clean["is_canceled"]
    X = df_clean.drop(columns=["is_canceled"])

    # Separamos los datos de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Separamos las columnas numéricas y categóricas
    num_cols, cat_cols = split_features(X_train)


    # Configuramos La URL y el experimento de MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Churn-Models-Eval")

    # Recorremos los modelos que vayamos a evaluar
    for model_path in models_dir.glob("*_model.joblib"):
        model_name = model_path.stem.replace("_model", "")
        log.info("📊 Evaluando %s", model_name)

        # Cargamos el Pipeline completo
        model = joblib.load(model_path)

        # Predecimos las etiquetas y probabilidades
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Evaluamos las métricas
        metrics = evaluate(y_test, y_pred, y_prob)

        # Guardamos las métricas en MLflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("algorithm", model_name)
            mlflow.log_param("threshold", 0.5)
            mlflow.log_metrics(metrics)

            # Registramos el modelo en MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=model_name
            )

        log.info("   → ROC-AUC: %.4f | F1: %.4f",
                 metrics["roc_auc"], metrics["f1"])

    log.info("✅ Todo registrado. Ejecuta ahora:  mlflow ui --port 5000")

if __name__ == "__main__":
    main()

# python mlflow_eval.py --data <ruta al CSV> ---------> Evaluación de todos los modelos
# mlflow models serve -m "models:/random_forest/1" -p 1234 -----------> Servir 1 modelo

# mlflow ui --port 5000 -----------> Ver en el navegador
