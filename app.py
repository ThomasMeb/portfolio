"""
Portfolio - Thomas
Vitrine de réalisations ML/Data Science

Point d'entrée principal pour Hugging Face Spaces
"""

import streamlit as st
from components import render_sidebar

# Configuration de la page
st.set_page_config(
    page_title="Thomas | Portfolio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Portfolio de réalisations ML/Data Science par Thomas"
    }
)

# CSS personnalisé
st.markdown("""
<style>
    /* Style global */
    .main {
        padding: 2rem;
    }

    /* Cards pour les projets */
    .project-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }

    /* Badges de type de projet */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    .badge-ml { background-color: #00d4aa; color: #000; }
    .badge-nlp { background-color: #ff6b6b; color: #fff; }
    .badge-cv { background-color: #4ecdc4; color: #000; }
    .badge-saas { background-color: #ffd93d; color: #000; }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 0;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .hero p {
        font-size: 1.2rem;
        color: #666;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Hide Streamlit branding and default nav */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

render_sidebar()

# Page principale - Landing
def main():
    # Hero Section
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("🚀 Bienvenue sur mon Portfolio")
    st.markdown("**ML Engineer & Entrepreneur** | Passionné par l'IA et la Data Science")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Projet Actuel - Highlight
    st.header("🎯 Projet Actuel")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("egir.app")
        st.markdown('<span class="badge badge-saas">SaaS</span><span class="badge badge-ml">IA</span>', unsafe_allow_html=True)
        st.markdown("""
        **Plateforme de gestion pour restaurateurs avec IA intégrée**

        - 📊 Calcul automatisé des coûts matières
        - 🤖 Fiches techniques assistées par IA
        - 📈 Dashboard d'analyse de rentabilité

        *+10% de marge en moyenne | 80% de temps économisé*
        """)
        st.link_button("Découvrir egir.app →", "https://egir.app", type="primary")

    with col2:
        st.image("assets/egir_logo.png", use_container_width=True)

    st.divider()

    # Réalisations - Aperçu
    st.header("💻 Réalisations ML/Data Science")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🔋 Prédiction Énergétique")
            st.caption("Schneider Electric | Régression | ✅ Démo active")
            st.markdown("Prédiction de consommation énergétique avec Random Forest")
            st.page_link("pages/5_Schneider_Energy.py", label="Explorer →", icon="🔋")

        with st.container(border=True):
            st.subheader("🏷️ NLP Tag Suggestion")
            st.caption("Stack Overflow | NLP | ✅ Démo active")
            st.markdown("Classification multi-label avec BERT/USE")
            st.page_link("pages/7_StackOverflow_NLP.py", label="Explorer →", icon="🏷️")

    with col2:
        with st.container(border=True):
            st.subheader("👥 Segmentation Client")
            st.caption("Back Market | Clustering | ✅ Démo active")
            st.markdown("Segmentation RFM avec KMeans")
            st.page_link("pages/6_BackMarket_Segmentation.py", label="Explorer →", icon="👥")

        with st.container(border=True):
            st.subheader("🐕 Classification Races")
            st.caption("SantéVet | Computer Vision | ✅ Démo active")
            st.markdown("Classification d'images avec ResNet50V2")
            st.page_link("pages/8_SanteVet_Dogs.py", label="Explorer →", icon="🐕")

    # Ligne 3 - Projets personnels
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🔍 JobScout")
            st.caption("Projet Personnel | Automation IA | ✅ Open Source")
            st.markdown("Agent autonome de recherche d'emploi avec scoring LLM (1,989 jobs)")
            st.page_link("pages/10_Job_Agent.py", label="Explorer →", icon="🔍")

    with col2:
        with st.container(border=True):
            st.subheader("📈 Grada Trading")
            st.caption("Projet Personnel | Time Series | ✅ Live Trading")
            st.markdown("Prédiction BTC via ensemble 4 modèles ML + Deep Learning, vault dHEDGE automatisé")
            st.page_link("pages/9_Grada_Trading.py", label="Explorer →", icon="📈")

    st.divider()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.caption("© 2026 Thomas | Built with Streamlit")

if __name__ == "__main__":
    main()
