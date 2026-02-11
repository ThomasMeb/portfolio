"""
Page Projet - Job Agent
Agent autonome de recherche d'emploi avec scoring IA
"""

import streamlit as st
from components import render_sidebar

st.set_page_config(
    page_title="Job Agent | Thomas Portfolio",
    page_icon="🤖",
    layout="wide"
)

render_sidebar()

st.title("🤖 Job Agent")
st.caption("Agent autonome de recherche d'emploi avec scoring IA")

st.divider()

# ============================================================================
# SECTION 1: CONTEXT & SOLUTION
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Le Problème")
    st.markdown("""
    La recherche d'emploi est un processus **chronophage et répétitif** :
    scruter manuellement des dizaines de plateformes, évaluer la pertinence
    de chaque offre, et préparer des candidatures personnalisées.

    **Job Agent** automatise l'intégralité du pipeline : du scraping
    de 5+ sources au scoring intelligent par LLM, en passant par les
    notifications Telegram et le suivi dans Notion.
    """)

with col2:
    st.metric("Jobs scrapés", "1,989")
    st.metric("Scoring IA", "1,989", delta="100% traités")
    st.metric("Offres pertinentes", "612", delta="score >= 60")
    st.metric("Coût total", "$1.51", delta="DeepSeek API")

st.divider()

# ============================================================================
# SECTION 2: ARCHITECTURE
# ============================================================================

st.header("Architecture du Pipeline")

st.code("""
 Toutes les 6h (7h-23h) :

 1. SCRAPE    5 sources (WTTJ, Adzuna, France Travail, RemoteOK, JobSpy)
              ↓
 2. DEDUP     SHA256 hash sur titre + entreprise + URL
              ↓
 3. SCORE     DeepSeek LLM : score 0-100 + reasoning + keywords
              ↓
 4. NOTIFY    Telegram : boutons Intéressé / Ignorer / Préparer CV
              ↓
 5. SYNC      Notion : jobs >= 50 + 8 entreprises cibles
              ↓
 6. PREP      Brief de candidature auto pour score >= 90
""", language="text")

st.divider()

# ============================================================================
# SECTION 3: FONCTIONNALITES
# ============================================================================

st.header("Fonctionnalités")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Scraping multi-source**
    - Welcome to the Jungle
    - Adzuna API
    - France Travail API
    - RemoteOK
    - JobSpy (Indeed + LinkedIn)
    - Déduplication inter-sources
    """)

with col2:
    st.markdown("""
    **Scoring IA (DeepSeek)**
    - Score 0-100 contre profil
    - Reasoning détaillé
    - Keywords matchés / manquants
    - Priorité (high/medium/low)
    - Budget LLM contrôlé ($5/mois)
    """)

with col3:
    st.markdown("""
    **Notifications & Suivi**
    - Telegram Bot avec boutons
    - Sync bidirectionnelle Notion
    - Dashboard Streamlit
    - Briefs de candidature auto
    - 8 entreprises cibles surveillées
    """)

st.divider()

# ============================================================================
# SECTION 4: RESULTATS
# ============================================================================

st.header("Résultats")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Sources actives", "5")
with col2:
    st.metric("Score >= 70", "78", delta="high priority")
with col3:
    st.metric("Entreprises cibles", "8")
with col4:
    st.metric("Coût / 1000 jobs", "$0.76")

st.markdown("""
**Top entreprises matchées** : Oney (ML Engineer, Lille), Coface, AXA, Hubvisory,
Paylead, Bitstack, Davidson Consulting, Matmut
""")

st.divider()

# ============================================================================
# SECTION 5: TECH STACK
# ============================================================================

st.header("Stack Technique")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Backend**
    - Python 3.12, AsyncIO
    - SQLite (WAL mode)
    - APScheduler
    - httpx (async HTTP)
    """)

with col2:
    st.markdown("""
    **APIs**
    - DeepSeek LLM (scoring)
    - Telegram Bot API
    - Notion API v2022-06-28
    - France Travail OAuth2
    - Adzuna REST API
    """)

with col3:
    st.markdown("""
    **Outils**
    - BeautifulSoup4
    - python-jobspy
    - Streamlit (dashboard)
    - Plotly (visualisation)
    - systemd (daemon)
    """)

st.divider()

# ============================================================================
# SECTION 6: LIENS
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.link_button(
        "Voir le code sur GitHub",
        "https://github.com/ThomasMeb/job-agent",
        type="primary",
        use_container_width=True,
    )

st.caption("Open source (MIT) - Adaptable par n'importe qui via config.yaml")
