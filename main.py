import pandas as pd
import numpy as np
import streamlit as st
import requests
import time
from PIL import Image
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier

### ========== rodar localmente ==========

#poetry run streamlit run main.py

# ========== CONFIGURAÇÕES ==========
st.set_page_config(page_title="SpaceDEX Game", layout="wide")
st.title("🧐 SpaceDEX — Qual é esse Objeto?!")
st.markdown("Tente adivinhar qual tipo de objeto é esse com base em suas características!")

# ========== INICIALIZA VARIÁVEIS DE ESTADO ==========
if "linha" not in st.session_state:
    st.session_state.linha = None
if "pontos" not in st.session_state:
    st.session_state.pontos = 0
if "nivel" not in st.session_state:
    st.session_state.nivel = 1
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "user_choice" not in st.session_state:
    st.session_state.user_choice = None

# ========== HUD ==========
with st.container():
    st.markdown(f"""
    ### 🏆 Nível {st.session_state.nivel}
    - 🌌 Pontuação: `{st.session_state.pontos}` pontos  
    - 🚀 Progresso para o próximo nível: `{st.session_state.pontos % 1000}/1000`
    """)

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

# ========== FUNÇÃO NOVA AMOSTRA ==========
def nova_amostra():
    st.session_state.linha = df.sample(1).iloc[0]
    st.session_state.show_feedback = False

# ========== AMOSTRA ==========
if st.session_state.linha is None:
    nova_amostra()

linha = st.session_state.linha
ra, dec = linha['alpha'], linha['delta']
X_input = pd.DataFrame([linha[feature_cols]])
pred_real = rf.predict(X_input)[0]

# ========== LAYOUT ==========
if not st.session_state.show_feedback:
    with st.container():
        col1, col2 = st.columns([1.2, 1.8])

        with col1:
            st.subheader("🔭 Visualização do Objeto + Dados")
            col_img, col_info = st.columns([1, 1.2])

            with col_img:
                try:
                    img = carregar_imagem(ra, dec)
                    st.image(img, caption="Imagem capturada pelo SDSS")
                except:
                    st.warning("Imagem não disponível.")

            with col_info:
                st.markdown(f"- **RA** (ascensão reta): `{ra:.2f}`")
                st.markdown(f"- **DEC** (declinação): `{dec:.2f}`")
                st.markdown(f"- **Redshift**: `{linha['redshift']:.4f}`")
                st.markdown("**Magnitudes:**")
                for col in feature_cols[:-1]:
                    st.markdown(f"- `{col.upper()}`: {linha[col]:.2f}")

        with col2:
            st.subheader("📊 Guia Visual")
            col_g1, col_g2, col_g3 = st.columns(3)

            with col_g1:
                st.image("assets/exemplo_galaxy.jpg", caption="🌌 GALAXY", use_container_width=True)
            with col_g2:
                st.image("assets/exemplo_star.png", caption="⭐ STAR", use_container_width=True)
            with col_g3:
                st.image("assets/exemplo_qso.jpg", caption="✨ QSO", use_container_width=True)

# ========== INTERAÇÃO ========== 
st.markdown("---")
st.subheader("💡 Qual é esse objeto?")
st.selectbox("Escolha a classe:", target_options, key="resposta")

if st.button("✅ Verificar", key="verificar"):
    st.session_state.user_choice = st.session_state.resposta
    st.session_state.show_feedback = True

# ========== FEEDBACK ========== 
if st.session_state.show_feedback:
    resposta = st.session_state.user_choice
    if resposta == pred_real:
        st.session_state.pontos += 100
        novo_nivel = st.session_state.pontos // 1000 + 1
        if novo_nivel > st.session_state.nivel:
            st.session_state.nivel = novo_nivel
            st.balloons()
            st.success(f"🎉 Subiu para o Nível {st.session_state.nivel}!")
        else:
            st.success(f"✅ Correto! Era um **{pred_real}**! (+100 pontos)")
    else:
        st.error(f"❌ Não era... A resposta certa era **{pred_real}**.")

    st.markdown("⏳ Gerando nova amostragem...")
    time.sleep(1)
    nova_amostra()
    st.rerun()
