"""
Sidebar réutilisable pour toutes les pages du portfolio
"""

import streamlit as st


def render_sidebar():
    """Affiche la sidebar avec photo et liens"""
    with st.sidebar:
        st.image("assets/photo_thomas.png", width=100)
        st.title("Thomas")
        st.caption("ML Engineer & Entrepreneur")

        st.divider()

        st.markdown("### Navigation")
        st.page_link("app.py", label="Accueil", icon="🏠")
        st.page_link("pages/1_Projet_Actuel.py", label="Projet Actuel", icon="🚀")
        st.page_link("pages/2_Realisations.py", label="Réalisations", icon="💻")
        st.page_link("pages/3_About.py", label="About", icon="👤")
        st.page_link("pages/4_Contact.py", label="Contact", icon="📧")

        st.divider()

        st.markdown("### Démos Actives")
        st.page_link("pages/5_Schneider_Energy.py", label="Schneider Energy", icon="🔋")
        st.page_link("pages/6_BackMarket_Segmentation.py", label="BackMarket Segment.", icon="👥")
        st.page_link("pages/7_StackOverflow_NLP.py", label="StackOverflow NLP", icon="🏷️")
        st.page_link("pages/8_SanteVet_Dogs.py", label="SantéVet Dogs", icon="🐕")
        st.page_link("pages/9_Grada_Trading.py", label="Grada Trading", icon="📈")
        st.page_link("pages/10_Job_Agent.py", label="JobScout", icon="🔍")

        st.divider()

        st.markdown("### Liens")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ThomasMeb)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/thomasmebarki)")
