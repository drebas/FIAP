import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuração da página
st.set_page_config(
    page_title="Obesidade | FIAP TC4",
    page_icon="🏥",
    layout="wide",
)

# Carregar modelo e artefatos
@st.cache_resource
def load_artifacts():
    model          = joblib.load("model.pkl")
    feature_cols   = joblib.load("feature_columns.pkl")
    target_decoder = joblib.load("target_decoder.pkl")
    return model, feature_cols, target_decoder

model, FEATURE_COLS, TARGET_DECODER = load_artifacts()

# Navegação entre páginas
pagina = st.sidebar.selectbox("Navegação", ["🔍 Predição", "📊 Dashboard Analítico"])
