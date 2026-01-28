"""
Page Projet Actuel - egir.app
"""

import streamlit as st

st.set_page_config(
    page_title="Projet Actuel | Thomas Portfolio",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Projet Actuel : egir.app")
st.caption("SaaS de gestion pour restaurateurs avec IA intégrée")

st.divider()

# Présentation
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Le Problème")
    st.markdown("""
    Les restaurateurs perdent un temps considérable à gérer leurs **fiches techniques**
    et à calculer leurs **coûts matières**. La plupart utilisent encore Excel,
    avec tous les risques d'erreurs que cela comporte.

    **Conséquences :**
    - Marges mal maîtrisées
    - Prix de vente sous-estimés
    - Temps administratif excessif
    """)

    st.header("La Solution")
    st.markdown("""
    **egir.app** est une plateforme SaaS qui automatise la gestion des fiches techniques
    et optimise la rentabilité des restaurateurs grâce à l'IA.

    **Fonctionnalités clés :**
    - 📊 **Calcul automatisé** des coûts matières
    - 🤖 **IA intégrée** pour la création de fiches techniques
    - 📈 **Dashboard** d'analyse de rentabilité
    - 📱 **Interface intuitive** accessible partout
    """)

with col2:
    st.image("https://via.placeholder.com/400x300.png?text=egir.app+Dashboard", use_container_width=True)
    st.caption("Dashboard egir.app")

st.divider()

# Résultats
st.header("📊 Résultats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Marge moyenne", value="+10%", delta="vs Excel")

with col2:
    st.metric(label="Temps économisé", value="80%", delta="sur la gestion")

with col3:
    st.metric(label="ROI estimé", value="19-33x", delta="par an")

st.divider()

# Mon rôle
st.header("👨‍💻 Mon Rôle")

st.markdown("""
**Fondateur & Développeur Full-Stack**

En tant que créateur d'egir.app, je gère :
- 🏗️ **Architecture technique** : Conception et développement de la plateforme
- 🤖 **Intégration IA** : Mise en place des fonctionnalités d'intelligence artificielle
- 📊 **Data Engineering** : Pipelines de données et analytics
- 🚀 **Product Management** : Vision produit et roadmap

*Ce projet représente la convergence de mes compétences en ML/Data Science
et en développement logiciel.*
""")

st.divider()

# Call to Action
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.link_button("🌐 Découvrir egir.app", "https://egir.app", type="primary", use_container_width=True)
    st.caption("Essai gratuit 14 jours | Sans engagement")
