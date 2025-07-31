from pathlib import Path
from typing import Dict, Optional, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hotel_api")

app = FastAPI(
    title="Hotel Cancellation Predictor",
    version="1.0.0",
    description="API basada en los modelos de los notebooks.",
)

MODEL_DIR = Path("outputs")
MODEL_MAP = {
    "logistic_regression": MODEL_DIR / "logistic_regression_model.joblib",
    "decision_tree":       MODEL_DIR / "decision_tree_model.joblib",
    "random_forest":       MODEL_DIR / "random_forest_model.joblib",
    "xgboost":             MODEL_DIR / "xgboost_model.joblib",
    "neural_network":      MODEL_DIR / "neural_network_model.joblib",
}

models: Dict[str, object] = {n: joblib.load(p) for n, p in MODEL_MAP.items() if p.exists()}

class BookingData(BaseModel):
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
    company: Optional[float] = None
    days_in_waiting_list: Optional[int] = None
    customer_type: Optional[str] = None
    adr: Optional[float] = None
    required_car_parking_spaces: Optional[int] = None
    total_of_special_requests: Optional[int] = None


def _align(model, df: pd.DataFrame) -> pd.DataFrame:
    try:
        prep = model.named_steps["prep"]
        expected = list(prep.feature_names_in_)
        return df.reindex(columns=expected, fill_value=np.nan)
    except Exception:
        return df


def _proba(model, df: pd.DataFrame) -> float:
    p = model.predict_proba(df)

    # Primero verificar la dimensionalidad del array
    if p.ndim == 1:
        # Si es unidimensional, simplemente devolvemos el primer valor
        return float(p[0])

    # Si es bidimensional, intentamos obtener la probabilidad de la clase positiva
    if hasattr(model, "classes_") and len(model.classes_) > 1:
        # Obtener el índice de la clase positiva (generalmente 1)
        pos_idx = np.where(model.classes_ == 1)[0][0]
        return float(p[0, pos_idx])
    else:
        # Si no podemos determinar la clase positiva, asumimos que es la columna 1
        return float(p[0, 1])


@app.post("/predict_all")
def predict_all(
    data: BookingData,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Umbral de decisión"),
) -> Dict[str, Any]:
    if not models:
        raise HTTPException(503, "No hay modelos cargados.")

    raw = pd.DataFrame([data.model_dump()])
    out = {}
    for name, model in models.items():
        try:
            df = _align(model, raw)
            p = _proba(model, df)
            out[name] = {"probability": p, "prediction": int(p >= threshold)}
        except Exception as e:
            out[name] = {"error": str(e)}
    return out

@app.get("/models")
def get_models() -> Dict[str, str]:
    """Lista los modelos disponibles."""
    return {name: str(path) for name, path in MODEL_MAP.items() if path.exists()}


@app.post("/predict/{model_name}")
def predict(
    model_name: str,
    data: BookingData,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Umbral de decisión"),
) -> Dict[str, Any]:
    if model_name not in models:
        raise HTTPException(404, f"Modelo '{model_name}' no encontrado.")

    model = models[model_name]
    raw = pd.DataFrame([data.model_dump()])
    df = _align(model, raw)

    try:
        probability = _proba(model, df)
        prediction = int(probability >= threshold)
        return {"probability": probability, "prediction": prediction}
    except Exception as e:
        raise HTTPException(500, f"Error al predecir: {str(e)}")