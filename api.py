# api.py (versión corregida)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from preprocessing import clean_dataset  # Importar la función de limpieza

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Cancelaciones de Hotel",
    description="API para predecir si una reserva será cancelada",
    version="1.0"
)

# Ruta a los modelos y mapeo de nombres
MODELS_DIR = Path("outputs")
MODEL_NAME_MAPPING = {
    "decision": "decision_tree",
    "logistic": "logistic_regression",
    "neural": "neural_network",
    "random": "random_forest",
    "xgboost": "xgboost"
}

# Cargar los modelos y guardar sus características esperadas
models = {}
model_features = {}

for model_path in MODELS_DIR.glob("*_model.joblib"):
    original_name = model_path.stem.split("_")[0]
    # Usar el nombre correcto según el mapeo
    if original_name in MODEL_NAME_MAPPING.values():
        # Encontrar el nombre corto para la API
        api_name = [k for k, v in MODEL_NAME_MAPPING.items() if v == original_name][0]
    else:
        api_name = original_name

    print(f"Cargando modelo: {api_name} desde {model_path}")
    model = joblib.load(model_path)
    models[api_name] = model

    # Guardar las características que espera el modelo
    if hasattr(model, 'get_feature_names_out'):
        model_features[api_name] = model.get_feature_names_out()
    elif hasattr(model, 'feature_names_in_'):
        model_features[api_name] = model.feature_names_in_
    elif hasattr(model, 'steps') and hasattr(model.steps[0][1], 'get_feature_names_out'):
        model_features[api_name] = model.steps[0][1].get_feature_names_out()

if not models:
    raise RuntimeError(f"No se encontraron modelos en {MODELS_DIR}")

class BookingData(BaseModel):
    # Mantener todas las columnas posibles, incluyendo company
    hotel: Optional[str] = None
    lead_time: Optional[int] = None
    arrival_date_year: Optional[int] = None
    arrival_date_month: Optional[str] = None
    arrival_date_week_number: Optional[int] = None
    arrival_date_day_of_month: Optional[int] = None
    stays_in_weekend_nights: Optional[int] = None
    stays_in_week_nights: Optional[int] = None
    adults: Optional[int] = None
    children: Optional[float] = None
    babies: Optional[int] = None
    meal: Optional[str] = None
    country: Optional[str] = None
    market_segment: Optional[str] = None
    distribution_channel: Optional[str] = None
    is_repeated_guest: Optional[int] = None
    previous_cancellations: Optional[int] = None
    previous_bookings_not_canceled: Optional[int] = None
    reserved_room_type: Optional[str] = None
    assigned_room_type: Optional[str] = None
    booking_changes: Optional[int] = None
    deposit_type: Optional[str] = None
    agent: Optional[float] = None
    company: Optional[float] = None  # Descomentar para incluirlo
    days_in_waiting_list: Optional[int] = None
    customer_type: Optional[str] = None
    adr: Optional[float] = None
    required_car_parking_spaces: Optional[int] = None
    total_of_special_requests: Optional[int] = None

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica el mismo preprocesamiento que se usó durante el entrenamiento"""
    # Aplicar la limpieza básica del dataset
    data_processed = clean_dataset(data.copy())

    # Asegurar que todos los tipos de datos sean correctos
    numeric_cols = ['lead_time', 'arrival_date_year', 'adults', 'children', 'babies',
                   'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
                   'booking_changes', 'days_in_waiting_list', 'adr',
                   'required_car_parking_spaces', 'total_of_special_requests']

    for col in numeric_cols:
        if col in data_processed.columns:
            data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce').fillna(0)

    return data_processed

@app.get("/")
def read_root():
    return {"message": "API de predicción de cancelaciones de hotel",
            "modelos_disponibles": list(models.keys())}

@app.post("/predict/{model_name}")
def predict(model_name: str, booking: BookingData):
    """Predice si una reserva será cancelada usando el modelo especificado."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")

    # Convertir los datos a DataFrame y aplicar el preprocesamiento completo
    data = pd.DataFrame([booking.model_dump()])
    data_processed = preprocess_data(data)

    model = models[model_name]
    try:
        # Manejar todos los modelos de manera consistente
        prediction = model.predict(data_processed)[0]

        # Obtener probabilidades cuando sea posible
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data_processed)[0][1]
        elif model_name == "neural":
            # Para el modelo neural, extraer la probabilidad directamente
            raw_prediction = model.predict(data_processed)
            if hasattr(raw_prediction, "numpy"):
                probability = float(raw_prediction.numpy().flatten()[0])
            else:
                probability = float(raw_prediction.flatten()[0])
        else:
            # Si no hay método de probabilidad
            probability = float(prediction)

        return {
            "model": model_name,
            "prediction": int(prediction),
            "will_cancel": bool(prediction == 1),
            "probability": float(probability)
        }
    except Exception as e:
        error_msg = f"Error al predecir con {model_name}: {str(e)}"
        print(error_msg)
        print(f"Forma de los datos: {data_processed.shape}")
        print(f"Columnas: {data_processed.columns.tolist()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/predict_all")
def predict_all(booking: BookingData):
    """Predice usando todos los modelos disponibles."""
    results = {}

    # Preprocesamiento de los datos una sola vez
    data = pd.DataFrame([booking.model_dump()])
    data_processed = preprocess_data(data)

    for name, model in models.items():
        try:
            # Predicción consistente para todos los modelos
            prediction = model.predict(data_processed)[0]

            # Manejo de probabilidades
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(data_processed)[0][1]
            elif name == "neural":
                # Para el modelo neural
                raw_prediction = model.predict(data_processed)
                if hasattr(raw_prediction, "numpy"):
                    probability = float(raw_prediction.numpy().flatten()[0])
                else:
                    probability = float(raw_prediction.flatten()[0])
            else:
                probability = float(prediction)

            results[name] = {
                "prediction": int(prediction),
                "will_cancel": bool(prediction == 1),
                "probability": float(probability)
            }
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"Error en modelo {name}: {str(e)}")

    return results

