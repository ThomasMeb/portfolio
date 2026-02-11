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
        st.page_link("pages/1_🚀_Projet_Actuel.py", label="Projet Actuel", icon="🚀")
        st.page_link("pages/2_💻_Réalisations.py", label="Réalisations", icon="💻")
        st.page_link("pages/3_👤_About.py", label="About", icon="👤")
        st.page_link("pages/4_📧_Contact.py", label="Contact", icon="📧")

        st.divider()

        st.markdown("### Démos Actives")
        st.page_link("pages/5_🔋_Schneider_Energy.py", label="Schneider Energy", icon="🔋")
        st.page_link("pages/6_👥_BackMarket_Segmentation.py", label="BackMarket Segment.", icon="👥")
        st.page_link("pages/7_🏷️_StackOverflow_NLP.py", label="StackOverflow NLP", icon="🏷️")
        st.page_link("pages/8_🐕_SanteVet_Dogs.py", label="SantéVet Dogs", icon="🐕")
        st.page_link("pages/9_📈_Alla2_Trading.py", label="Alla2 Trading", icon="📈")
        st.page_link("pages/10_🤖_Job_Agent.py", label="Job Agent", icon="🤖")

        st.divider()

        st.markdown("### Liens")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ThomasMeb)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/thomasmebarki)")
