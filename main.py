import pandas as pd
import numpy as np
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier

# ========== CONFIGURAÇÕES ========== 
st.set_page_config(page_title="SpaceDEX Game", layout="wide")
st.title("🧐 SpaceDEX — Quem é esse Objeto?!")
st.markdown("Tente adivinhar qual tipo de objeto é esse com base em suas características!")

# ========== FUNÇÕES DE CACHED ========== 
@st.cache_data(show_spinner="🔭 Carregando dados...")
def carregar_dados():
    return pd.read_csv("star_classification.csv")

@st.cache_resource
def carregar_modelo():
    df = carregar_dados()
    X = df[feature_cols]
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_data(show_spinner="🔭 Baixando imagem...")
def carregar_imagem(ra, dec):
    url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.15&width=250&height=250"
    response = requests.get(url, timeout=5)
    return Image.open(BytesIO(response.content))

# ========== VARIÁVEIS GLOBAIS ========== 
feature_cols = ['u', 'g', 'r', 'i', 'z', 'redshift']
target_col = 'class'
df = carregar_dados()
target_options = df[target_col].unique()
rf = carregar_modelo()

# ========== SESSION STATE ========== 
if "linha" not in st.session_state:
    st.session_state.linha = df.sample(1).iloc[0]
    st.session_state.show_feedback = False

# ========== FUNÇÃO NOVA AMOSTRA ========== 
def nova_amostra():
    st.session_state.linha = df.sample(1).iloc[0]
    st.session_state.show_feedback = False

# ========== AMOSTRA ATUAL ========== 
linha = st.session_state.linha
ra, dec = linha['alpha'], linha['delta']
X_input = pd.DataFrame([linha[feature_cols]])
pred_real = rf.predict(X_input)[0]

# ========== LAYOUT: IMAGEM E INFO ========== 
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🔭 Visualização do Objeto")
    try:
        img = carregar_imagem(ra, dec)
        st.image(img, caption="Imagem capturada pelo SDSS")
    except:
        st.warning("Imagem não disponível.")

with col2:
    st.subheader("📊 Informações do Objeto")
    st.markdown(f"- **RA** (ascensão reta): `{ra:.2f}`")
    st.markdown(f"- **DEC** (declinação): `{dec:.2f}`")
    st.markdown(f"- **Redshift**: `{linha['redshift']:.4f}`")
    st.markdown("**Magnitudes:**")
    for col in feature_cols[:-1]:
        st.markdown(f"- `{col.upper()}`: {linha[col]:.2f}")

# ========== INTERAÇÃO ========== 
st.markdown("---")
st.subheader("💡 Qual é esse objeto?")
resposta = st.selectbox("Escolha a classe:", target_options, key="resposta")

col_verif, col_reset = st.columns([1, 1])
with col_verif:
    if st.button("✅ Verificar"):
        st.session_state.show_feedback = True

with col_reset:
    if st.button("🔁 Nova Amostragem"):
        nova_amostra()

# ========== FEEDBACK ========== 
if st.session_state.show_feedback:
    if resposta == pred_real:
        st.success(f"✅ Correto! Era um **{pred_real}**!")
        st.balloons()
    else:
        st.error(f"❌ Não era... A resposta certa era **{pred_real}**.")
