import requests
import streamlit as st
import pandas as pd

# Cargamos los modelos disponibles desde el API
@st.cache_data(show_spinner=False)
def get_available_models(url: str) -> list[str]:
    try:
        resp = requests.get(f"{url}/models")
        resp.raise_for_status()
        return list(resp.json().keys())
    except Exception as ex:
        st.warning(f"No se pudieron obtener los modelos ({ex}).")
        return []

# Función para llamar al endpoint de predicción
def predict(endpoint: str, payload: dict, threshold: float) -> dict:
    params = {"threshold": threshold}
    resp = requests.post(endpoint, json=payload, params=params)
    resp.raise_for_status()
    return resp.json()

# Página de predicciones
def predictions_page():

    # Título de la página
    st.title("💎 Predicción de Cancelaciones")

    # URL de la API
    API_URL = "http://localhost:8000"


    # Cargamos los modelos disponibles
    models = get_available_models(API_URL)
    if not models:
        st.stop()

    col1, col2 = st.columns([2, 3])

    # Seleccionamos el modelo o si queremos predecir con todos
    with col1:
        mode = st.radio("¿Con qué modelos quieres predecir?",
                        ["Un solo modelo", "Todos"],
                        horizontal=True)

        if mode == "Un solo modelo":
            chosen_model = st.selectbox("Modelo", models, index=0)

    # Formulario con 15 variables de entrada posibles para la predicción
    with st.form("prediction_form"):
        st.subheader("Datos de la reserva")
        c1, c2, c3 = st.columns(3)

        hotel         = c1.selectbox("Tipo de hotel", ["City Hotel", "Resort Hotel"])
        lead_time     = c1.number_input("Lead Time", 0, 500, 60)
        arrival_year  = c1.selectbox("Año llegada", [2015, 2016, 2017])
        arrival_month = c1.selectbox("Mes llegada", ["January","February","March","April","May","June","July","August","September","October","November","December"])
        weekend_nights = c1.number_input("Noches fin de semana", 0, 20, 2)
        week_nights    = c2.number_input("Noches entre semana", 0, 50, 5)
        adults         = c2.number_input("Adultos", 1, 10, 2)
        children       = c2.number_input("Niños", 0, 10, 0)
        babies         = c2.number_input("Bebés", 0, 5, 0)
        adr            = c2.number_input("ADR (€)", 0.0, 1000.0, 100.0, step=1.0)
        is_repeated_guest = c3.selectbox("¿Es huésped repetido?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        deposit_type = c3.selectbox("Tipo de depósito", ["No Deposit", "Refundable", "Non Refund"])
        previous_cancellations = c3.number_input("Cancelaciones previas", 0, 20, 0)
        total_special_requests = c3.number_input("Solicitudes especiales", 0, 10, 0)
        market_segment = c3.selectbox("Segmento de mercado", ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups", "Aviation", "Complementary"])

        submitted = st.form_submit_button("Predecir")

    # Si se ha enviado el formulario, hacemos la predicción
    if submitted:
        base_payload = {
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": arrival_year,
            "arrival_date_month": arrival_month,
            "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "adr": adr,
            # Nuevas variables añadidas al payload
            "is_repeated_guest": is_repeated_guest,
            "deposit_type": deposit_type,
            "previous_cancellations": previous_cancellations,
            "total_of_special_requests": total_special_requests,
            "market_segment": market_segment
        }

        try:
            if mode == "Un solo modelo":
                endpoint = f"{API_URL}/predict/{chosen_model}"
                result = predict(endpoint, base_payload, threshold=0.5)
                st.success(f"**{chosen_model}** → Probabilidad: {result['probability']:.2%} · "
                           f"Predicción: {'❌ Cancelada' if result['prediction'] else '✅ No cancelada'}")
            else:
                endpoint = f"{API_URL}/predict_all"
                result = predict(endpoint, base_payload, threshold=0.5)
                df = pd.DataFrame(result).T
                df["probabilidad"] = df["probability"].apply(lambda x: f"{x*100:.2f}%")
                df["prediction"] = df["prediction"].map({0: "✅ No", 1: "❌ Sí"})
                df.drop(columns=["probability"], inplace=True)
                df.rename(columns={"prediction": "cancelación"}, inplace=True)
                st.dataframe(df, use_container_width=True)

        except requests.HTTPError as err:
            st.error(f"Error {err.response.status_code}: {err.response.text}")
        except Exception as ex:
            st.error(f"Ocurrió un error inesperado: {ex}")