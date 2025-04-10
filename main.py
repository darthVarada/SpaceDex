import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import pickle

# ========== CONFIGURAÇÃO ==========
st.set_page_config(page_title="SpaceDEX Game", layout="centered")
st.title("🧠 SpaceDEX — Quem é esse Objeto?!")
st.markdown("Tente adivinhar qual tipo de objeto é esse com base em suas características!")

# ========== CARREGAMENTO ==========
with open("random_forest_model.pkl", "rb") as f:
    rf = pickle.load(f)

df = pd.read_csv("star_classification.csv")
feature_cols = ['u', 'g', 'r', 'i', 'z', 'redshift']
target_names = {0: 'GALAXY', 1: 'STAR', 2: 'QSO'}
target_options = list(target_names.values())

# ========== INICIALIZA ESTADO ==========
if 'amostra' not in st.session_state:
    st.session_state.amostra = None
    st.session_state.pred_real = None
    st.session_state.show_feedback = False

# ========== NOVA AMOSTRA ==========
def nova_amostra():
    linha = df.sample(n=1).iloc[0]
    st.session_state.amostra = linha
    X_input = pd.DataFrame([linha[feature_cols]])
    pred = rf.predict(X_input)[0]
    st.session_state.pred_real = target_names[pred]
    st.session_state.show_feedback = False

# ========== BOTÃO NOVA AMOSTRA ==========
if st.button("🔁 Nova Amostragem"):
    nova_amostra()

# Primeira vez
if st.session_state.amostra is None:
    nova_amostra()

linha = st.session_state.amostra
pred_real = st.session_state.pred_real

# ========== MOSTRA IMAGEM ==========
ra, dec = linha['alpha'], linha['delta']
url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.2&width=300&height=300"
try:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img, caption="🔭 Objeto Espacial")
except:
    st.warning("Imagem não disponível no momento.")

# ========== EXIBE INFO ==========
with st.expander("📊 Informações do Objeto"):
    st.write(f"**RA**: {ra:.2f}")
    st.write(f"**DEC**: {dec:.2f}")
    for col in feature_cols:
        st.write(f"**{col.upper()}**: {linha[col]:.3f}")

# ========== APOSTA DO USUÁRIO ==========
st.markdown("## 💡 Qual é esse objeto?")
resposta = st.selectbox("Selecione uma classe:", target_options, key="resposta")

if st.button("Verificar"):
    st.session_state.show_feedback = True

# ========== FEEDBACK ==========
if st.session_state.show_feedback:
    if resposta == pred_real:
        st.success(f"✅ Correto! Era um **{pred_real}**!")
    else:
        st.error(f"❌ Ops! Era **{pred_real}**...")

