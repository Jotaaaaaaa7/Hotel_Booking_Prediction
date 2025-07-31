from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

# Página con las métricas de los modelos
def metrics_models_page():

    # Título
    st.title("🧮 Métricas de Modelos")

    # Cargamos el archivo de métricas
    METRICS_FILE = Path("outputs/metrics_20250731_170005.json")

    if not METRICS_FILE.exists():
        st.warning(f"No se encontró el archivo de métricas en {METRICS_FILE}. Mostramos datos de ejemplo.")

        # Datos de ejemplo para mostrar en caso de no encontrar el archivo con las métricas
        # Warning encima para que se sepa
        metrics_raw = {
            "Logistic Regression": {"metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.76, "f1": 0.79, "roc_auc": 0.88}},
            "Random Forest": {"metrics": {"accuracy": 0.88, "precision": 0.85, "recall": 0.79, "f1": 0.82, "roc_auc": 0.91}},
            "XGBoost": {"metrics": {"accuracy": 0.89, "precision": 0.86, "recall": 0.80, "f1": 0.83, "roc_auc": 0.92}},
        }
    else:
        # Cargamos las métricas desde el archivo JSON
        with METRICS_FILE.open() as f:
            metrics_raw = json.load(f)

    records = []
    for model_name, info in metrics_raw.items():
        row = {"model": model_name, **info["metrics"]}
        records.append(row)

    df_metrics = pd.DataFrame(records)

    metric_choice = st.selectbox(
        "Selecciona la métrica a comparar",
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.bar(
            df_metrics,
            x="model",
            y=metric_choice,
            text=metric_choice,
            height=400,
            labels={"model": "Modelo", metric_choice: metric_choice.capitalize()},
        )
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_layout(yaxis_tickformat=".3f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Detalles de cada modelo")
        for model_name, info in metrics_raw.items():
            with st.expander(f"🔍 {model_name}"):
                st.markdown("**Métricas**")
                st.table(pd.DataFrame(info["metrics"], index=["valor"]).T)

                st.markdown("**Mejores hiperparámetros**")
                if info.get("best_params"):
                    st.json(info["best_params"])
                else:
                    st.info("Este modelo no tiene hiperparámetros ajustados.")
