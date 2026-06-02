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

if pagina == "🔍 Predição":
    st.title("🏥 Preditor de Nível de Obesidade")
    st.markdown("Preencha os dados do paciente para obter a predição do modelo.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        gender         = st.selectbox("Gênero", ["Female", "Male"])
        age            = st.number_input("Idade (anos)", min_value=14, max_value=80, value=25)
        family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])
        favc           = st.selectbox("Come alimentos calóricos com frequência?", ["yes", "no"])
        fcvc           = st.slider("Frequência de consumo de vegetais (1=raramente, 3=sempre)", 1, 3, 2)
        ncp            = st.slider("Número de refeições principais por dia", 1, 4, 3)
        caec           = st.selectbox("Come entre as refeições?", ["Sometimes", "no", "Frequently", "Always"])

    with col2:
        smoke  = st.selectbox("Fuma?", ["no", "yes"])
        ch2o   = st.slider("Consumo de água diário (1=<1L, 2=1-2L, 3=>2L)", 1, 3, 2)
        scc    = st.selectbox("Monitora as calorias ingeridas?", ["no", "yes"])
        faf    = st.slider("Frequência de atividade física/semana (0=nenhuma, 3=intensa)", 0, 3, 1)
        tue    = st.slider("Tempo com eletrônicos/dia (0=<2h, 1=3-5h, 2=>5h)", 0, 2, 1)
        calc   = st.selectbox("Consome álcool?", ["Sometimes", "no", "Frequently", "Always"])
        mtrans = st.selectbox("Meio de transporte habitual",
                              ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

if pagina == "🔍 Predição":
    st.title("🏥 Preditor de Nível de Obesidade")
    st.markdown("Preencha os dados do paciente para obter a predição do modelo.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gênero", ["Female", "Male"],
                              format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
        age            = st.number_input("Idade (anos)", min_value=14, max_value=80, value=25)
        family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"],
                                      format_func=lambda x: "Sim" if x == "yes" else "Não")
        favc           = st.selectbox("Come alimentos calóricos com frequência?", ["yes", "no"],
                                      format_func=lambda x: "Sim" if x == "yes" else "Não")
        fcvc           = st.slider("Frequência de consumo de vegetais (1=raramente, 3=sempre)", 1, 3, 2)
        ncp            = st.slider("Número de refeições principais por dia", 1, 4, 3)
        caec           = st.selectbox("Come entre as refeições?",
                                      ["Sometimes", "no", "Frequently", "Always"],
                                      format_func=lambda x: {
                                          "Sometimes": "Às vezes",
                                          "no": "Não",
                                          "Frequently": "Frequentemente",
                                          "Always": "Sempre"
                                      }[x])

    with col2:
        smoke  = st.selectbox("Fuma?", ["no", "yes"],
                              format_func=lambda x: "Não" if x == "no" else "Sim")
        ch2o   = st.slider("Consumo de água diário (1=<1L, 2=1-2L, 3=>2L)", 1, 3, 2)
        scc    = st.selectbox("Monitora as calorias ingeridas?", ["no", "yes"],
                              format_func=lambda x: "Não" if x == "no" else "Sim")
        faf    = st.slider("Frequência de atividade física/semana (0=nenhuma, 3=intensa)", 0, 3, 1)
        tue    = st.slider("Tempo com eletrônicos/dia (0=<2h, 1=3-5h, 2=>5h)", 0, 2, 1)
        calc   = st.selectbox("Consome álcool?",
                              ["Sometimes", "no", "Frequently", "Always"],
                              format_func=lambda x: {
                                  "Sometimes": "Às vezes",
                                  "no": "Não",
                                  "Frequently": "Frequentemente",
                                  "Always": "Sempre"
                              }[x])
        mtrans = st.selectbox("Meio de transporte habitual",
                              ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
                              format_func=lambda x: {
                                  "Public_Transportation": "Transporte Público",
                                  "Automobile": "Automóvel",
                                  "Walking": "A pé",
                                  "Motorbike": "Moto",
                                  "Bike": "Bicicleta"
                              }[x])

    st.divider()

    if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):

        binary_map = {"yes": 1, "no": 0, "Male": 1, "Female": 0}

        exercicio_score = faf * (2 - tue)
        sedentarismo    = tue * (3 - faf)

        input_data = {
            "Gender":                      binary_map[gender],
            "Age":                         age,
            "family_history":              binary_map[family_history],
            "FAVC":                        binary_map[favc],
            "FCVC":                        fcvc,
            "NCP":                         ncp,
            "SMOKE":                       binary_map[smoke],
            "CH2O":                        ch2o,
            "SCC":                         binary_map[scc],
            "FAF":                         faf,
            "TUE":                         tue,
            "CAEC_Always":                 int(caec == "Always"),
            "CAEC_Frequently":             int(caec == "Frequently"),
            "CAEC_Sometimes":              int(caec == "Sometimes"),
            "CAEC_no":                     int(caec == "no"),
            "CALC_Always":                 int(calc == "Always"),
            "CALC_Frequently":             int(calc == "Frequently"),
            "CALC_Sometimes":              int(calc == "Sometimes"),
            "CALC_no":                     int(calc == "no"),
            "MTRANS_Automobile":           int(mtrans == "Automobile"),
            "MTRANS_Bike":                 int(mtrans == "Bike"),
            "MTRANS_Motorbike":            int(mtrans == "Motorbike"),
            "MTRANS_Public_Transportation":int(mtrans == "Public_Transportation"),
            "MTRANS_Walking":              int(mtrans == "Walking"),
            "exercicio_score":             exercicio_score,
            "sedentarismo":                sedentarismo,
        }

        df_input = pd.DataFrame([input_data])[FEATURE_COLS]

        pred_class = model.predict(df_input)[0]
        pred_label = TARGET_DECODER[pred_class]
        proba      = model.predict_proba(df_input)[0]
        confianca  = proba[pred_class]

        # Tradução do resultado
        traducao_resultado = {
            "Insufficient_Weight": "Peso Insuficiente",
            "Normal_Weight":       "Peso Normal",
            "Overweight_Level_I":  "Sobrepeso Nível I",
            "Overweight_Level_II": "Sobrepeso Nível II",
            "Obesity_Type_I":      "Obesidade Tipo I",
            "Obesity_Type_II":     "Obesidade Tipo II",
            "Obesity_Type_III":    "Obesidade Tipo III",
        }

        st.subheader("📊 Resultado")
        st.metric("Nível de Obesidade Previsto", traducao_resultado[pred_label])
        st.metric("Confiança do Modelo", f"{confianca:.1%}")

        with st.expander("Ver probabilidades por classe"):
            TARGET_ORDER = [
                "Insufficient_Weight", "Normal_Weight",
                "Overweight_Level_I", "Overweight_Level_II",
                "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
            ]
            prob_df = pd.DataFrame({
                "Classe": [traducao_resultado[t] for t in TARGET_ORDER],
                "Probabilidade": [f"{proba[i]:.1%}" for i in range(len(TARGET_ORDER))]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        st.info("⚠️ Este sistema é um auxílio à decisão médica. O diagnóstico definitivo deve ser realizado por um profissional de saúde.")
