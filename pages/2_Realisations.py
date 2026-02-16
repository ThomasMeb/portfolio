"""
Page Réalisations - Projets ML/Data Science
"""

import streamlit as st
from components import render_sidebar

st.set_page_config(
    page_title="Réalisations | Thomas Portfolio",
    page_icon="💻",
    layout="wide"
)

render_sidebar()

st.title("💻 Réalisations ML/Data Science")
st.caption("Projets démontrant mes compétences en Machine Learning et Data Science")

st.divider()

# Filtres
col1, col2 = st.columns([3, 1])
with col2:
    filtre = st.selectbox(
        "Filtrer par type",
        ["Tous", "Régression", "Clustering", "NLP", "Computer Vision", "Time Series", "Automation"]
    )

st.divider()

# Projet 1 - P3 Schneider ✅ DÉMO ACTIVE
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔋 Prédiction de Consommation Énergétique")
        st.caption("Schneider Electric | Régression | ✅ Démo disponible")

        st.markdown("""
        **Contexte :** Mission freelance pour Schneider Electric - prédiction de
        consommation énergétique et émissions CO2 du parc immobilier tertiaire.

        **Approche :**
        - Feature engineering sur 47 variables (34% valeurs manquantes traitées)
        - 18 modèles comparés (Random Forest champion)
        - Interprétabilité avec SHAP values

        **Stack :** Python, Scikit-learn, XGBoost, SHAP, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.page_link("pages/5_Schneider_Energy.py", label="🎮 Démo interactive", icon="🔋")
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/ThomasMeb/P3-schneider-energy-prediction")

    with col2:
        st.metric("Amélioration", "+45.5%", delta="vs baseline")
        st.metric("Bâtiments analysés", "1,650")
        st.progress(83, text="R² = 0.83")

st.divider()

# Projet 2 - P4 BackMarket ✅ DÉMO ACTIVE
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("👥 Segmentation Client RFM")
        st.caption("Back Market | Clustering | ✅ Démo disponible")

        st.markdown("""
        **Contexte :** Segmentation de la base clients de Back Market
        pour optimiser les campagnes marketing.

        **Approche :**
        - Analyse RFM (Récence, Fréquence, Montant)
        - Clustering avec KMeans (k=4 optimisé)
        - Visualisation 3D interactive des segments

        **Stack :** Python, Scikit-learn, Pandas, Plotly, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.page_link("pages/6_BackMarket_Segmentation.py", label="🎮 Démo interactive", icon="👥")
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/ThomasMeb/P4-backmarket-segmentation")

    with col2:
        st.metric("Clients", "95K", delta="segmentés")
        st.metric("Silhouette Score", "0.49")
        st.progress(49, text="Qualité du clustering")

st.divider()

# Projet 3 - P5 StackOverflow ✅ DÉMO ACTIVE
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🏷️ Suggestion de Tags NLP")
        st.caption("Stack Overflow | NLP - Classification Multi-label | ✅ Démo disponible")

        st.markdown("""
        **Contexte :** Système de suggestion automatique de tags
        pour les questions Stack Overflow.

        **Approche :**
        - Preprocessing NLP (tokenization, lemmatization)
        - Embeddings : TF-IDF, BERT, Universal Sentence Encoder
        - Classification multi-label avec seuil optimisé

        **Stack :** Python, Transformers, TensorFlow, FastAPI, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.page_link("pages/7_StackOverflow_NLP.py", label="🎮 Démo interactive", icon="🏷️")
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/ThomasMeb/P5-stackoverflow-nlp-tags")

    with col2:
        st.metric("Precision@5", "78%", delta="+8% vs baseline")
        st.metric("Recall@5", "62%")
        st.progress(78, text="Tags pertinents")

st.divider()

# Projet 4 - P6 SantéVet ✅ DÉMO ACTIVE
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🐕 Classification de Races de Chiens")
        st.caption("SantéVet | Computer Vision - Deep Learning | ✅ Démo disponible")

        st.markdown("""
        **Contexte :** Classification automatique de races de chiens
        pour l'application mobile SantéVet (LPA).

        **Approche :**
        - Transfer Learning avec ResNet50V2
        - Fine-tuning sur Stanford Dogs Dataset (120 races)
        - Classificateur SVM sur features 2048-dim

        **Stack :** Python, TensorFlow/Keras, scikit-learn, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.page_link("pages/8_SanteVet_Dogs.py", label="🎮 Démo interactive", icon="🐕")
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/ThomasMeb/P6-santevet-dog-classification")

    with col2:
        st.metric("Top-1 Accuracy", "87%")
        st.metric("Top-3 Accuracy", "96%")
        st.progress(87, text="Précision Top-1")

st.divider()

# Projet 5 - Grada ✅ LIVE TRADING
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Grada - Prédiction & Trading BTC")
        st.caption("Projet Personnel | Time Series - Live Trading | ✅ Vault dHEDGE actif")

        st.markdown("""
        **Contexte :** Prédiction de la direction du BTC à J+1
        et exécution automatisée via un vault dHEDGE sur Polygon.

        **Approche :**
        - 22 features (14 techniques + 8 macro-économiques)
        - XGBoost walk-forward sur 1500 jours glissants
        - Position sizing optimisé (EarningStrategy)
        - Execution on-chain via KyberSwap + dHEDGE SDK

        **Stack :** Python, XGBoost, TypeScript, dHEDGE SDK, Polygon
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.page_link("pages/9_Grada_Trading.py", label="Voir le dashboard", icon="📈")
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/ThomasMeb/Grada")

    with col2:
        st.metric("Accuracy", "61.6%", delta="+9.3% vs baseline")
        st.metric("Sharpe Ratio", "3.37")
        st.progress(62, text="Précision directionnelle")

st.divider()

# Projet 6 - JobScout
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 JobScout - Recherche d'Emploi Automatisée")
        st.caption("Projet Personnel | Automation IA | ✅ Open Source")

        st.markdown("""
        **Contexte :** Agent autonome qui scrape, score et notifie
        les offres d'emploi pertinentes en utilisant un LLM.

        **Approche :**
        - Scraping multi-source (5 plateformes, déduplication SHA256)
        - Scoring IA avec DeepSeek LLM (reasoning + keywords)
        - Notifications Telegram avec boutons d'action
        - Sync automatique vers Notion

        **Stack :** Python, AsyncIO, SQLite, DeepSeek API, Telegram Bot, Notion API, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.page_link("pages/10_Job_Agent.py", label="Voir le projet", icon="🔍")
        with col_btn2:
            st.link_button("Code GitHub", "https://github.com/ThomasMeb/JobScout")

    with col2:
        st.metric("Jobs scrapés", "1,989")
        st.metric("Score >= 60", "612", delta="pertinents")
        st.progress(61, text="Coût: $1.51 / $5.00 budget")

st.divider()

# Footer
st.success("✅ **6 projets disponibles** dont 5 démos actives + 1 outil open source!")
st.info("💡 **Tip :** Utilisez le menu latéral pour accéder aux démos interactives.")
