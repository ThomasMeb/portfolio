"""
Page Projet - JobScout
Agent autonome de recherche d'emploi avec scoring IA
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from components import render_sidebar

st.set_page_config(
    page_title="JobScout | Thomas Portfolio",
    page_icon="🤖",
    layout="wide"
)

render_sidebar()

st.title("🔍 JobScout")
st.caption("Agent autonome de recherche d'emploi avec scoring IA")

st.divider()

# ============================================================================
# DATA LOADING (demo DB)
# ============================================================================

DEMO_DB = Path(__file__).parent.parent / "assets" / "jobscout_demo.db"


@st.cache_data(ttl=600)
def load_demo_data():
    if not DEMO_DB.exists():
        return None, None, None, None
    conn = sqlite3.connect(str(DEMO_DB), check_same_thread=False)
    jobs = pd.read_sql_query(
        """SELECT *, ROW_NUMBER() OVER (PARTITION BY title, company ORDER BY id) as dup_rank
        FROM jobs WHERE match_score IS NOT NULL ORDER BY match_score DESC""",
        conn,
    )
    jobs = jobs[jobs["dup_rank"] == 1].drop(columns=["dup_rank"])
    companies = pd.read_sql_query(
        "SELECT * FROM companies ORDER BY relevance_score DESC", conn
    )
    scrape_runs = pd.read_sql_query(
        "SELECT source, SUM(jobs_found) as found, SUM(jobs_new) as new, COUNT(*) as runs "
        "FROM scrape_runs WHERE status='success' GROUP BY source ORDER BY found DESC",
        conn,
    )
    llm_usage = pd.read_sql_query(
        "SELECT SUM(cost_usd) as total_cost, SUM(input_tokens + output_tokens) as total_tokens, "
        "COUNT(DISTINCT strftime('%Y-%m-%d', created_at)) as days FROM llm_usage",
        conn,
    )
    conn.close()
    return jobs, companies, scrape_runs, llm_usage


jobs_df, companies_df, scrape_df, llm_df = load_demo_data()

# ============================================================================
# TABS
# ============================================================================

tab_dashboard, tab_projet, tab_stack = st.tabs(["Dashboard", "Projet", "Stack Technique"])

# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================

with tab_dashboard:
    if jobs_df is None:
        st.warning("Base de données démo non disponible.")
    else:
        st.info("Données anonymisées — les noms d'entreprises ont été remplacés.")

        # --- KPIs ---
        total_jobs = len(jobs_df)
        score_60 = len(jobs_df[jobs_df["match_score"] >= 60])
        score_70 = len(jobs_df[jobs_df["match_score"] >= 70])
        total_companies = len(companies_df) if companies_df is not None else 0
        total_cost = llm_df["total_cost"].iloc[0] if llm_df is not None and not llm_df.empty else 0
        total_tokens = llm_df["total_tokens"].iloc[0] if llm_df is not None and not llm_df.empty else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total jobs", f"{total_jobs:,}")
        k2.metric("Score >= 60", f"{score_60:,}")
        k3.metric("Score >= 70", f"{score_70:,}")
        k4.metric("Entreprises cibles", total_companies)
        k5.metric("Coût LLM", f"${total_cost:.2f}")

        st.markdown("")

        # --- Score distribution ---
        st.subheader("Distribution des scores")

        score_col1, score_col2 = st.columns([2, 1])

        with score_col1:
            bins = [0, 30, 50, 70, 100]
            labels_cat = ["Faible (0-30)", "Moyen (30-50)", "Bon (50-70)", "Excellent (70+)"]
            colors_cat = ["#E74C3C", "#F39C12", "#4A90D9", "#2ECC71"]
            score_cats = pd.cut(jobs_df["match_score"], bins=bins, labels=labels_cat, include_lowest=True)
            cat_counts = score_cats.value_counts().reindex(labels_cat).fillna(0).astype(int)

            fig_cat = px.bar(
                x=cat_counts.index,
                y=cat_counts.values,
                color=cat_counts.index,
                color_discrete_map=dict(zip(labels_cat, colors_cat)),
                text=cat_counts.values,
                labels={"x": "", "y": "Nombre d'offres"},
            )
            fig_cat.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                bargap=0.2,
            )
            fig_cat.update_traces(textposition="outside")
            st.plotly_chart(fig_cat, use_container_width=True)

        with score_col2:
            for label, color, count in zip(labels_cat, colors_cat, cat_counts.values):
                pct = count / total_jobs * 100 if total_jobs > 0 else 0
                st.markdown(f"**{label}**")
                st.progress(pct / 100, text=f"{count} offres ({pct:.0f}%)")

        # --- Top jobs table ---
        st.subheader("Top offres (score >= 50)")

        top_jobs = jobs_df[jobs_df["match_score"] >= 50].copy()
        display_cols = ["title", "company", "match_score", "source", "location", "remote_type", "match_priority"]
        display_names = {
            "title": "Titre",
            "company": "Entreprise",
            "match_score": "Score",
            "source": "Source",
            "location": "Localisation",
            "remote_type": "Remote",
            "match_priority": "Priorité",
        }
        available_cols = [c for c in display_cols if c in top_jobs.columns]
        st.dataframe(
            top_jobs[available_cols].rename(columns=display_names),
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.0f /100"),
            },
        )

        # --- Job detail ---
        if not top_jobs.empty:
            with st.expander("Voir le détail du scoring IA"):
                detail_options = top_jobs[["title", "company", "match_score"]].copy()
                detail_options["label"] = detail_options.apply(
                    lambda r: f"[{int(r['match_score'])}] {r['title']} — {r['company']}", axis=1
                )
                selected = st.selectbox("Choisir une offre", detail_options["label"].tolist()[:50])
                if selected:
                    idx = detail_options[detail_options["label"] == selected].index[0]
                    job = top_jobs.loc[idx]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Reasoning**")
                        st.write(job.get("match_reasoning", "N/A"))
                    with c2:
                        st.markdown("**Keywords matchés**")
                        st.write(job.get("match_keywords", "N/A"))
                        st.markdown("**Keywords manquants**")
                        st.write(job.get("missing_keywords", "N/A"))

        # --- Sources + LLM costs ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Performance des sources")
            if scrape_df is not None and not scrape_df.empty:
                fig_sources = px.bar(
                    scrape_df,
                    x="source",
                    y=["found", "new"],
                    barmode="group",
                    labels={"value": "Jobs", "source": "Source", "variable": ""},
                    color_discrete_map={"found": "#4A90D9", "new": "#50C878"},
                )
                fig_sources.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=300,
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_sources, use_container_width=True)

        with col_right:
            st.subheader("Coûts LLM")
            cost_k1, cost_k2 = st.columns(2)
            cost_per_job = total_cost / total_jobs if total_jobs > 0 else 0
            cost_k1.metric("Coût total", f"${total_cost:.2f}")
            cost_k2.metric("Coût / job", f"${cost_per_job:.4f}")
            cost_k3, cost_k4 = st.columns(2)
            cost_k3.metric("Tokens utilisés", f"{total_tokens / 1e6:.1f}M")
            days = llm_df["days"].iloc[0] if llm_df is not None and not llm_df.empty else 0
            cost_k4.metric("Jours actifs", int(days))

# ============================================================================
# TAB 2: PROJET
# ============================================================================

with tab_projet:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Le Problème")
        st.markdown("""
        La recherche d'emploi est un processus **chronophage et répétitif** :
        scruter manuellement des dizaines de plateformes, évaluer la pertinence
        de chaque offre, et préparer des candidatures personnalisées.

        **JobScout** automatise l'intégralité du pipeline : du scraping
        de 5+ sources au scoring intelligent par LLM, en passant par les
        notifications Telegram et le suivi dans Notion.
        """)

    with col2:
        st.metric("Jobs scrapés", f"{total_jobs:,}" if jobs_df is not None else "3,400+")
        st.metric("Scoring IA", f"{total_jobs:,}" if jobs_df is not None else "3,400+", delta="100% traités")
        st.metric("Offres pertinentes", f"{score_60:,}" if jobs_df is not None else "1,100+", delta="score >= 60")
        st.metric("Coût total", f"${total_cost:.2f}" if llm_df is not None else "~$2.50", delta="DeepSeek API")

    st.divider()

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
        - Dashboard interactif
        - Briefs de candidature auto
        - 8 entreprises cibles surveillées
        """)

# ============================================================================
# TAB 3: STACK TECHNIQUE
# ============================================================================

with tab_stack:
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
        - systemd (daemon 24/7)
        """)

    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.link_button(
            "Voir le code sur GitHub",
            "https://github.com/ThomasMeb/JobScout",
            type="primary",
            use_container_width=True,
        )

    st.caption("Open source (MIT) — Adaptable par n'importe qui via config.yaml")
