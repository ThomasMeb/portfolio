"""
Page Projet Grada - Bitcoin Trading Prediction & Automated Execution
Dashboard avec données réelles de backtest walk-forward
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from components import render_sidebar

st.set_page_config(
    page_title="Grada Trading | Portfolio",
    page_icon="📈",
    layout="wide"
)

render_sidebar()

st.title("📈 Grada - BTC Prediction & Trading")
st.caption("Prédiction directionnelle BTC avec XGBoost + exécution automatisée via vault dHEDGE")

st.divider()

# =============================================================================
# DATA LOADING
# =============================================================================

BACKTEST_PATH = Path(__file__).parent.parent / "assets" / "grada_backtest.json"


@st.cache_data(ttl=3600)
def load_backtest_data():
    if not BACKTEST_PATH.exists():
        return None
    with open(BACKTEST_PATH) as f:
        return json.load(f)


data = load_backtest_data()

# =============================================================================
# TABS
# =============================================================================

tab_dashboard, tab_projet, tab_stack = st.tabs(["Dashboard", "Projet", "Stack Technique"])

# =============================================================================
# TAB 1: DASHBOARD
# =============================================================================

with tab_dashboard:
    if data is None:
        st.warning("Données de backtest non disponibles. Exécuter `export_portfolio_data.py` dans le projet Grada.")
    else:
        full = data["full_period"]
        oos = data["oos_period"]
        curve = data["equity_curve"]

        # --- Métriques comparées Full vs OOS ---
        st.subheader("Performance : Période Complète vs Out-of-Sample")

        col_full, col_sep, col_oos = st.columns([5, 1, 5])

        with col_full:
            st.markdown(f"**Période complète** ({full['start']} → {full['end']})")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{full['accuracy']*100:.1f}%")
            m2.metric("Sharpe", f"{full['sharpe']}")
            m3.metric("Earning", f"{full['earning']}x")
            m4, m5, m6 = st.columns(3)
            m4.metric("Sortino", f"{full['sortino']}")
            m5.metric("Max Drawdown", f"{full['max_drawdown']*100:.1f}%")
            m6.metric("Trades", f"{full['n_trades']:,}")

        with col_sep:
            st.markdown("")

        with col_oos:
            if oos:
                st.markdown(f"**Out-of-Sample** ({oos['start']} → {oos['end']})")
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{oos['accuracy']*100:.1f}%")
                m2.metric("Sharpe", f"{oos['sharpe']}")
                m3.metric("Earning", f"{oos['earning']}x")
                m4, m5, m6 = st.columns(3)
                m4.metric("Sortino", f"{oos['sortino']}")
                m5.metric("Max Drawdown", f"{oos['max_drawdown']*100:.1f}%")
                m6.metric("Trades", f"{oos['n_trades']:,}")
            else:
                st.info("Pas de données OOS disponibles")

        st.markdown("")

        # --- Equity Curve ---
        st.subheader("Equity Curve : Stratégie Grada vs Buy & Hold BTC")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=curve["dates"],
            y=curve["strategy"],
            name="Stratégie Grada",
            line=dict(color="#22c55e", width=2),
        ))

        fig.add_trace(go.Scatter(
            x=curve["dates"],
            y=curve["baseline"],
            name="Buy & Hold BTC",
            line=dict(color="#3b82f6", width=2, dash="dash"),
        ))

        # Zone OOS
        if oos:
            fig.add_vrect(
                x0=oos["start"], x1=oos["end"],
                fillcolor="rgba(168, 85, 247, 0.08)",
                line_width=0,
                annotation_text="OOS",
                annotation_position="top left",
                annotation_font_color="#a855f7",
            )

        fig.update_layout(
            yaxis_type="log",
            yaxis_title="Valeur du portfolio (log, base=1.0)",
            xaxis_title="",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Allocation Area Chart ---
        st.subheader("Allocation au fil du temps")

        fig_alloc = go.Figure()

        fig_alloc.add_trace(go.Scatter(
            x=curve["dates"],
            y=[a * 100 for a in curve["allocations"]],
            fill="tozeroy",
            name="% WBTC",
            line=dict(color="#f59e0b", width=1),
            fillcolor="rgba(245, 158, 11, 0.3)",
        ))

        fig_alloc.add_hline(y=50, line_dash="dot", line_color="#94a3b8",
                            annotation_text="Neutre (50%)")

        fig_alloc.update_layout(
            yaxis_title="Allocation WBTC (%)",
            yaxis_range=[0, 105],
            xaxis_title="",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )

        st.plotly_chart(fig_alloc, use_container_width=True)

# =============================================================================
# TAB 2: PROJET
# =============================================================================

with tab_projet:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Méthodologie")
        st.markdown("""
        **Walk-Forward Validation** sur fenêtre glissante de 1500 jours :
        chaque jour, le modèle est ré-entraîné sur les 1500 jours précédents
        et prédit la direction du BTC pour le lendemain.

        | Composant | Détail |
        |-----------|--------|
        | **Modèle** | XGBoost Classifier (max\\_depth=5, 200 estimators) |
        | **Features** | 22 = 14 techniques + 8 macro-économiques |
        | **Cible** | Direction du prix BTC à J+1 (hausse/baisse) |
        | **Sizing** | EarningStrategy (threshold 5%, deadzone 5%) |
        | **Frais** | 0.1% par portion rééquilibrée |
        """)

        st.divider()

        st.header("Découvertes clés")
        st.markdown("""
        - **Macro features (+269% earning)** : DXY, S&P 500, VIX et Gold
          apportent un signal fort. L'ajout de 8 features macro
          a multiplié par 3.7x le earning metric.

        - **On-chain features (-18.6%)** : MVRV, exchange flows, hash rate, SOPR...
          toutes dégradent les performances. Le bruit l'emporte sur le signal.

        - **Calibration des probabilités** : XGBoost surpasse LightGBM et CatBoost
          non pas par l'accuracy (+0.2%) mais par la qualité de ses probabilités,
          ce qui optimise le position sizing.

        - **Fee drag** : le levier qui compte est le fee rate, pas la fréquence.
          Break-even à ~0.75% de frais par trade.
        """)

    with col2:
        st.header("Vault dHEDGE")

        st.markdown("""
        **GRADA** — vault automatisé sur Polygon

        | Info | Valeur |
        |------|--------|
        | Réseau | Polygon (137) |
        | DEX | KyberSwap |
        | Tokens | WBTC / USDC |
        | SDK | dHEDGE v2 (ethers v5) |
        | Premier trade | 13 fév 2026 |
        """)

        st.link_button(
            "Voir le vault sur dHEDGE",
            "https://app.dhedge.org/vault/0x27462cd4f35d4b3d118eaa85acb61a2cb9ba4e08",
            use_container_width=True,
        )

        st.divider()

        st.markdown("### Pipeline quotidien")
        st.code("""
00:30 UTC — Python
  1. Fetch données BTC + macro
  2. Feature engineering (22)
  3. Train XGBoost (1500j)
  4. Prédire proba → signal.json

01:00 UTC — TypeScript
  5. Lire signal.json
  6. Lire position vault
  7. Calculer swap nécessaire
  8. Exécuter via dHEDGE SDK
  9. Notification Telegram
        """, language="text")

# =============================================================================
# TAB 3: STACK TECHNIQUE
# =============================================================================

with tab_stack:
    st.header("Architecture")

    st.code("""
Python (cron 00:30 UTC)              TypeScript (cron 01:00 UTC)
┌─────────────────────┐              ┌─────────────────────────┐
│ generate_signal.py   │              │ hedge-bot/src/index.ts  │
│  1. Fetch données    │  signal.json │  1. Lire signal.json    │
│  2. Features 22      │ ──────────► │  2. Lire position vault │
│  3. Train XGBoost    │              │  3. Calculer swap       │
│  4. Predict proba    │              │  4. Exécuter via SDK    │
│  5. Écrire signal    │              │  5. Logger résultat     │
└─────────────────────┘              └─────────────────────────┘
    """, language="text")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **ML & Data (Python)**
        - XGBoost 2.0
        - scikit-learn (StandardScaler)
        - Pandas, NumPy
        - Yahoo Finance (données)
        - Walk-forward validation
        """)

    with col2:
        st.markdown("""
        **Trading Bot (TypeScript)**
        - dHEDGE SDK v2.1.5
        - ethers.js v5
        - KyberSwap (DEX)
        - Polygon RPC (PublicNode)
        - Signal JSON interface
        """)

    with col3:
        st.markdown("""
        **Infrastructure**
        - systemd timer (cron)
        - Telegram Bot (monitoring)
        - WSL2 / Linux
        - Node.js 25
        - Python 3.12 venv
        """)

    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.link_button(
            "Voir le code sur GitHub",
            "https://github.com/ThomasMeb/Grada",
            type="primary",
            use_container_width=True,
        )

    st.caption("Vault dHEDGE actif sur Polygon — Premier trade le 13 février 2026")
