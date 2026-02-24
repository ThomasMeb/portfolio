"""
Page About - Parcours et compétences
"""

import streamlit as st
from components import render_sidebar

st.set_page_config(
    page_title="About | Thomas Portfolio",
    page_icon="👤",
    layout="wide"
)

render_sidebar()

st.title("👤 À Propos de Moi")
st.caption("ML Engineer & Entrepreneur")

st.divider()

# Intro
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Qui suis-je ?")
    st.markdown("""
    Je suis **Thomas**, passionné par l'**Intelligence Artificielle** et la **Data Science**.

    Actuellement, je développe **[egir.app](https://egir.app)**, une plateforme SaaS
    pour restaurateurs intégrant de l'IA pour optimiser leur rentabilité.

    Mon parcours combine une solide formation en Data Science avec une expérience
    entrepreneuriale concrète, me permettant de transformer des problèmes business
    complexes en solutions techniques efficaces.
    """)

with col2:
    st.image("assets/photo_thomas.png", use_container_width=True)

st.divider()

# Compétences
st.header("🛠️ Compétences Techniques")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Machine Learning")
    st.markdown("""
    - **Scikit-learn** ████████░░ 80%
    - **XGBoost** ████████░░ 80%
    - **TensorFlow/Keras** ███████░░░ 70%
    - **PyTorch** ██████░░░░ 60%
    """)

    st.subheader("NLP")
    st.markdown("""
    - **Transformers (BERT)** ███████░░░ 70%
    - **spaCy** ███████░░░ 70%
    - **Hugging Face** ██████░░░░ 60%
    """)

with col2:
    st.subheader("Data Engineering")
    st.markdown("""
    - **Python** █████████░ 90%
    - **Pandas/NumPy** █████████░ 90%
    - **SQL** ████████░░ 80%
    - **Streamlit** ████████░░ 80%
    """)

    st.subheader("Visualisation")
    st.markdown("""
    - **Plotly** ████████░░ 80%
    - **Matplotlib/Seaborn** ████████░░ 80%
    - **Tableau** ██████░░░░ 60%
    """)

with col3:
    st.subheader("Développement")
    st.markdown("""
    - **Git/GitHub** █████████░ 90%
    - **FastAPI** ███████░░░ 70%
    - **Docker** ██████░░░░ 60%
    - **CI/CD** ██████░░░░ 60%
    """)

    st.subheader("Cloud & MLOps")
    st.markdown("""
    - **AWS** ██████░░░░ 60%
    - **Hugging Face** ███████░░░ 70%
    - **MLflow** █████░░░░░ 50%
    """)

st.divider()

# Parcours
st.header("📚 Formation & Parcours")

with st.expander("🎓 Formation", expanded=True):
    st.markdown("""
    **Master Data Science — spécialité Machine Learning** (2023)
    _CentraleSupélec_
    - Machine Learning & Deep Learning
    - NLP & Computer Vision
    - Data Engineering & MLOps

    **Licence de Mathématiques — Mention Bien** (2021)
    _Université du Littoral Côte d'Opale (ULCO)_

    **Diplôme d'Artisan Cuisinier** (2025)
    _À l'École des Chefs — Formation chez un maître restaurateur_
    """)

with st.expander("💼 Expérience", expanded=True):
    st.markdown("""
    **Fondateur — EGIR** (2025 - Présent)
    - SaaS analytique pour restaurateurs avec IA intégrée
    - Développement full-stack et product management

    **Immersion métier — Artisan Cuisinier** (Déc 2024 - Juil 2025)
    _Café de Paris, Calais_
    - Analyse terrain des marges, processus et contraintes opérationnelles

    **ML Engineer — Détection de Fraude** (Jan - Nov 2023)
    _Shift Technology (Licorne Insurtech)_
    - Feature engineering et modélisation de scoring de risque

    **Projets ML/Data Science** (2023 - 2024)
    - _Schneider Electric_ : Prédiction énergétique (Régression)
    - _Back Market_ : Segmentation clients (Clustering)
    - _Stack Overflow_ : Recommandation de tags (NLP Multi-label)
    - _SantéVet_ : Classification de races (Computer Vision)
    """)

st.divider()

# Certifications / Projets
st.header("🏆 Certifications & Réalisations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Projets ML réalisés :**
    - ✅ 5 projets end-to-end déployés
    - ✅ Couverture : Régression, Classification, Clustering, NLP, CV
    - ✅ Code open-source sur GitHub
    """)

with col2:
    st.markdown("""
    **Entrepreneuriat & Communauté :**
    - 🚀 egir.app en production
    - 📈 Clients actifs
    - 💡 Intégration IA réussie
    - 🐦 [14k+ abonnés sur X/Twitter](https://x.com/_elmeb_) — veille ML/produit
    """)

st.divider()

# CTA
st.info("📧 **Intéressé par mon profil ?** N'hésitez pas à me contacter via la page Contact !")
