"""
Page Réalisations - Projets ML/Data Science
"""

import streamlit as st

st.set_page_config(
    page_title="Réalisations | Thomas Portfolio",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Réalisations ML/Data Science")
st.caption("Projets démontrant mes compétences en Machine Learning et Data Science")

st.divider()

# Filtres
col1, col2 = st.columns([3, 1])
with col2:
    filtre = st.selectbox(
        "Filtrer par type",
        ["Tous", "Régression", "Clustering", "NLP", "Computer Vision", "Time Series"]
    )

st.divider()

# Projet 1 - P3 Schneider
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔋 Prédiction de Consommation Énergétique")
        st.caption("Schneider Electric | Régression")

        st.markdown("""
        **Contexte :** Projet réalisé pour Schneider Electric visant à prédire
        la consommation énergétique de bâtiments.

        **Approche :**
        - Feature engineering avancé (données temporelles, météo)
        - Modèles : XGBoost, Random Forest, Linear Regression
        - Optimisation des hyperparamètres avec GridSearchCV

        **Stack :** Python, Scikit-learn, XGBoost, Pandas, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("🎮 Démo interactive", key="demo_p3", type="primary", disabled=True)
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/", disabled=True)

    with col2:
        st.metric("RMSE", "0.12", delta="-15% vs baseline")
        st.metric("R²", "0.94")
        st.progress(94, text="Précision du modèle")

st.divider()

# Projet 2 - P4 BackMarket
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("👥 Segmentation Client RFM")
        st.caption("Back Market | Clustering")

        st.markdown("""
        **Contexte :** Segmentation de la base clients de Back Market
        pour optimiser les campagnes marketing.

        **Approche :**
        - Analyse RFM (Récence, Fréquence, Montant)
        - Clustering avec KMeans et DBSCAN
        - Visualisation des segments avec t-SNE

        **Stack :** Python, Scikit-learn, Pandas, Plotly, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("🎮 Démo interactive", key="demo_p4", type="primary", disabled=True)
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/", disabled=True)

    with col2:
        st.metric("Segments", "5", delta="identifiés")
        st.metric("Silhouette Score", "0.72")
        st.progress(72, text="Qualité du clustering")

st.divider()

# Projet 3 - P5 StackOverflow
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🏷️ Suggestion de Tags NLP")
        st.caption("Stack Overflow | NLP - Classification Multi-label")

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
            st.button("🎮 Démo interactive", key="demo_p5", type="primary", disabled=True)
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/", disabled=True)

    with col2:
        st.metric("F1 Score", "0.68", delta="+12% vs baseline")
        st.metric("Recall@5", "0.85")
        st.progress(85, text="Tags pertinents")

st.divider()

# Projet 4 - P6 SantéVet
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🐕 Classification de Races de Chiens")
        st.caption("SantéVet | Computer Vision - Deep Learning")

        st.markdown("""
        **Contexte :** Classification automatique de races de chiens
        pour l'application mobile SantéVet.

        **Approche :**
        - Transfer Learning avec ResNet50V2
        - Fine-tuning sur Stanford Dogs Dataset (120 races)
        - Data augmentation et régularisation

        **Stack :** Python, TensorFlow/Keras, OpenCV, Streamlit
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("🎮 Démo interactive", key="demo_p6", type="primary", disabled=True)
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/", disabled=True)

    with col2:
        st.metric("Accuracy", "87%", delta="Top-1")
        st.metric("Top-5 Accuracy", "96%")
        st.progress(96, text="Précision Top-5")

st.divider()

# Projet 5 - Alla2
with st.container(border=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Prédiction de Séries Temporelles")
        st.caption("Projet Personnel | Time Series - Trading")

        st.markdown("""
        **Contexte :** Modèle de prédiction pour séries temporelles financières.

        **Approche :**
        - Feature engineering temporel
        - Modèles : XGBoost, LSTM, Prophet
        - Backtesting et évaluation de performance

        **Stack :** Python, XGBoost, TensorFlow, Pandas
        """)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("🎮 Démo interactive", key="demo_alla2", type="primary", disabled=True)
        with col_btn2:
            st.link_button("📂 Code GitHub", "https://github.com/", disabled=True)

    with col2:
        st.metric("Direction Accuracy", "58%", delta="+8% vs random")
        st.metric("Sharpe Ratio", "1.2")
        st.progress(58, text="Précision directionnelle")

st.divider()

# Footer
st.info("💡 **Note :** Les démos interactives seront activées progressivement. Chaque projet sera enrichi avec des visualisations et des démonstrations en temps réel.")
