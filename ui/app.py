import streamlit as st
from pages_.dataset_info import dataset_info_page
from pages_.metrics_models import metrics_models_page
from pages_.predictions import predictions_page

# Configuración de la página
st.set_page_config(page_title="Hotel Analytics Suite",
                   page_icon="🏨",
                   layout="wide")

# Título y separador
st.sidebar.title("🏨 Hotel Analytics Suite")
st.sidebar.markdown("---")

# Inicializamos el estado de la sesión para la navegación
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dataset Info"

page_options = ["Dataset Info", "Metrics & Models", "Predictions"]
icons = {"Dataset Info": "📊", "Metrics & Models": "🧮", "Predictions": "💎"}

# Botones de navegación en la barra lateral
for page in page_options:
    if st.sidebar.button(f"{icons[page]} {page}", use_container_width=True):
        st.session_state.current_page = page
        st.rerun()

# Renderizamos la página seleccionada
if st.session_state.current_page == "Dataset Info":
    dataset_info_page()
elif st.session_state.current_page == "Metrics & Models":
    metrics_models_page()
elif st.session_state.current_page == "Predictions":
    predictions_page()