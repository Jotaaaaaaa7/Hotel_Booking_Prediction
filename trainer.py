# trainer.py
from __future__ import annotations
import argparse, json
import warnings
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Filtrar advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning,
                       message="Found unknown categories.+during transform")

from data_loader import load_data
from preprocessing import split_features, build_preprocessor, clean_dataset
from models import get_base_pipelines
from tuner import build_searchers

# ---------- Utils ----------
def evaluate(y_true, y_pred, y_prob) -> dict:
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_prob),
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hotel Cancellation – Trainer con tuning")
    p.add_argument("--data", required=True, help="CSV de reservas de hotel")
    p.add_argument("--outdir", default="outputs", help="Carpeta de salida")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos brutos
    df_raw = load_data(args.data)

    # 2. Aplicar limpieza (una sola vez)
    df_clean = clean_dataset(df_raw)

    # 3. Split DESPUÉS de la limpieza
    y = df_clean["is_canceled"]
    X = df_clean.drop(columns=["is_canceled"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # 4. Preprocesador (con columnas del DataFrame limpio)
    num_cols, cat_cols = split_features(X)  # X ya está limpio
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 5. Pipelines base + Searchers
    pipelines = get_base_pipelines(preprocessor)
    searchers = build_searchers(pipelines)

    # 6. Entrenamiento + evaluación
    summary = {}
    for name, model in searchers.items():
        print(f"🔧 Afinando '{name}' …")
        model.fit(X_train, y_train)

        best_estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = best_estimator.predict(X_test)
        y_prob = (
            best_estimator.predict_proba(X_test)[:, 1]
            if hasattr(best_estimator, "predict_proba")
            else y_pred
        )

        metrics = evaluate(y_test, y_pred, y_prob)
        summary[name] = {
            "metrics": metrics,
            "best_params": getattr(model, "best_params_", None),
        }
        print(f"✅ {name} – AUC: {metrics['roc_auc']:.4f}")

        # 7. Guardar modelo
        joblib.dump(best_estimator, outdir / f"{name}_model.joblib")

    # 8. Guardar métricas
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(outdir / f"metrics_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n📊 Resultados:\n", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()