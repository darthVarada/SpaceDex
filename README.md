
# 🚀 SpaceDEX — Classificador de Objetos Astronômicos com IA

O **SpaceDEX** é uma aplicação interativa construída com **Streamlit** que combina **Machine Learning** com **gamificação** para classificar objetos astronômicos — como galáxias, estrelas e quasares — a partir de dados do Sloan Digital Sky Survey (SDSS).

---

## 🎯 Objetivo

Transformar aprendizado de astronomia e IA em uma experiência interativa e divertida. O usuário tenta adivinhar a classe de um objeto com base em seus dados e visual, enquanto um modelo de IA avalia a resposta.

---

## 🌐 Demo Online

Acesse a aplicação:  
🔗 [SpaceDEX - Streamlit Cloud](https://darthvarada-spacedex-main-rmxzjf.streamlit.app/)

---

## 🤖 Sobre a IA

O modelo usa **Random Forest** treinado com dados reais do SDSS. As features incluem magnitudes fotométricas (u, g, r, i, z) e redshift.

- 📊 Dataset: SDSS - Star Classification
- 📈 Modelo: RandomForestClassifier
- 💡 Features: `u`, `g`, `r`, `i`, `z`, `redshift`
- 🧠 Target: `class` (GALAXY, STAR, QSO)

---

## 🕹 Funcionalidades

- Classificação automatizada com IA
- Sistema de pontos e níveis 🏆
- HUD com progresso 🚀
- Visual comparativo dos tipos de objeto
- Feedback interativo com acertos e erros
- Design inspirado em Pokédex para gamificação

---

## 🛠 Como Rodar Localmente

```bash
# Clone o repositório
git clone https://github.com/seuusuario/spacedex.git
cd spacedex

# Instale as dependências
poetry install

# Execute a aplicação
poetry run streamlit run main.py
```

---

## 🧰 Tecnologias Usadas

- Python 3.12
- Streamlit
- scikit-learn
- pandas / numpy
- Pillow / requests
- Poetry

---

## 👨‍💻 Autor

Desenvolvido por **Victor Barradas**  
🔗 [LinkedIn](https://www.linkedin.com/in/victor-barradas/) | 🐙 [GitHub](https://github.com/darthVarada)

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT.
