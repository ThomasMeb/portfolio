"""
Page Contact
"""

import streamlit as st
from components import render_sidebar

st.set_page_config(
    page_title="Contact | Thomas Portfolio",
    page_icon="📧",
    layout="wide"
)

render_sidebar()

st.title("📧 Contact")
st.caption("Restons en contact !")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("Me Contacter")

    st.markdown("""
    Je suis toujours ouvert aux discussions sur :
    - 💼 **Opportunités professionnelles** en ML/Data Science
    - 🤝 **Collaborations** sur des projets innovants
    - 💡 **Échanges** sur l'IA et l'entrepreneuriat
    - 🍽️ **egir.app** si vous êtes restaurateur !
    """)

    st.subheader("📬 Email")
    st.code("thomas.mebarki@protonmail.com", language=None)

    st.subheader("🔗 Réseaux")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.link_button("LinkedIn", "https://linkedin.com/in/thomasmebarki", use_container_width=True)

    with col_b:
        st.link_button("GitHub", "https://github.com/ThomasMeb", use_container_width=True)

    with col_c:
        st.link_button("X / Twitter", "https://x.com/_elmeb_", use_container_width=True)

    with col_d:
        st.link_button("egir.app", "https://egir.app", use_container_width=True)

with col2:
    st.header("Envoyer un Message")

    with st.form("contact_form"):
        name = st.text_input("Nom *")
        email = st.text_input("Email *")
        subject = st.selectbox(
            "Sujet",
            ["Opportunité professionnelle", "Collaboration", "Question technique", "egir.app", "Autre"]
        )
        message = st.text_area("Message *", height=150)

        submitted = st.form_submit_button("Envoyer", type="primary", use_container_width=True)

        if submitted:
            if name and email and message:
                st.success("✅ Message envoyé ! Je vous répondrai dans les plus brefs délais.")
                # Note: En production, implémenter l'envoi réel (email, webhook, etc.)
            else:
                st.error("❌ Veuillez remplir tous les champs obligatoires.")

st.divider()

# Localisation (optionnel)
st.header("📍 Localisation")
st.markdown("**France** - Disponible en remote")

# Footer
st.divider()
st.caption("💡 Réponse généralement sous 24-48h")
