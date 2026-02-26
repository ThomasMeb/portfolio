"""
Page Projet Grada - Bitcoin Trading Prediction & Automated Execution
Dashboard avec données réelles de backtest walk-forward + vault dHEDGE live
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st
from components import render_sidebar

st.set_page_config(
    page_title="Grada Trading | Portfolio",
    page_icon="📈",
    layout="wide"
)

render_sidebar()

st.title("📈 Grada - BTC Prediction & Trading")
st.caption("Prédiction directionnelle BTC avec ensemble ML + Deep Learning, exécution automatisée via vault dHEDGE")

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

VAULT_ADDRESS = "0x27462cd4f35d4b3d118eaa85acb61a2cb9ba4e08"
DHEDGE_API = "https://api-v2.dhedge.org/graphql"


WEI = 1e18


@st.cache_data(ttl=600)
def load_vault_data():
    """Fetch live vault data from dHEDGE GraphQL API."""
    query = """
    {
      fund(address: "%s") {
        name
        tokenPrice
        totalValue
        performanceMetrics {
          week
          month
          year
        }
      }
      tokenPriceCandles(address: "%s", period: "1m", interval: "1h") {
        timestamp
        open
        close
      }
      tradeEvents(
        filter: { fundAddress: "%s" }
        limit: 20
        orderBy: "timestamp"
        orderDirection: "desc"
      ) {
        txHash
        timestamp
        displayType
        income { assetName displayAmount }
        outcome { assetName displayAmount }
      }
    }
    """ % (VAULT_ADDRESS, VAULT_ADDRESS, VAULT_ADDRESS)
    try:
        resp = requests.post(DHEDGE_API, json={"query": query}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data")
    except Exception:
        return None


vault = load_vault_data()

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

        # --- Allocation moyenne mobile ---
        st.subheader("Allocation moyenne (30 jours glissants)")

        raw_alloc = [a * 100 for a in curve["allocations"]]
        window = 30
        smoothed = []
        for i in range(len(raw_alloc)):
            start = max(0, i - window + 1)
            smoothed.append(sum(raw_alloc[start:i + 1]) / (i - start + 1))

        fig_alloc = go.Figure()

        fig_alloc.add_trace(go.Scatter(
            x=curve["dates"],
            y=smoothed,
            fill="tozeroy",
            name="Allocation WBTC (MA 30j)",
            line=dict(color="#f59e0b", width=2),
            fillcolor="rgba(245, 158, 11, 0.2)",
        ))

        fig_alloc.add_hline(y=50, line_dash="dot", line_color="#94a3b8",
                            annotation_text="Neutre (50%)")

        fig_alloc.update_layout(
            yaxis_title="Allocation WBTC (%)",
            yaxis_range=[0, 100],
            xaxis_title="",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )

        st.plotly_chart(fig_alloc, use_container_width=True)

        # --- Vault Live ---
        if vault and vault.get("fund"):
            st.divider()
            st.subheader("Vault dHEDGE — Live")

            fund = vault["fund"]
            token_price = int(fund["tokenPrice"]) / WEI
            total_value = int(fund["totalValue"]) / WEI
            perf = fund.get("performanceMetrics", {})

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Token Price", f"${token_price:.4f}")
            v2.metric("AUM", f"${total_value:.2f}")
            week_pct = (int(perf["week"]) / WEI - 1) * 100 if perf.get("week") else 0
            v3.metric("7j", f"{week_pct:+.2f}%")
            month_pct = (int(perf["month"]) / WEI - 1) * 100 if perf.get("month") else 0
            v4.metric("30j", f"{month_pct:+.2f}%")

            # Token price chart from candles
            candles = vault.get("tokenPriceCandles")
            if candles and len(candles) > 2:
                fig_vault = go.Figure()
                c_dates = [
                    datetime.fromtimestamp(int(c["timestamp"]) / 1000, tz=timezone.utc)
                    for c in candles
                ]
                c_close = [int(c["close"]) / WEI for c in candles]
                y_min = min(c_close)
                y_max = max(c_close)
                y_pad = max((y_max - y_min) * 0.3, 0.0002)
                fig_vault.add_trace(go.Scatter(
                    x=c_dates, y=c_close,
                    name="Token Price (GRADA)",
                    line=dict(color="#22c55e", width=2),
                ))
                fig_vault.update_layout(
                    yaxis_title="Token Price ($)",
                    yaxis_range=[y_min - y_pad, y_max + y_pad],
                    xaxis_title="",
                    height=250,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_vault, use_container_width=True)

            # --- Trade History ---
            trades = vault.get("tradeEvents", [])
            if trades:
                st.markdown("")
                st.markdown("**Trades récents**")

                rows = []
                for t in trades:
                    ts = datetime.fromtimestamp(int(t["timestamp"]), tz=timezone.utc)
                    income = t.get("income") or []
                    outcome = t.get("outcome") or []

                    # Determine direction: selling USDC → buying WBTC
                    out_names = [o["assetName"] for o in outcome]
                    in_names = [i["assetName"] for i in income]

                    if "USDC" in out_names and "WBTC" in in_names:
                        direction = "BUY WBTC"
                        wbtc_amt = next((i["displayAmount"] for i in income if i["assetName"] == "WBTC"), 0)
                        usdc_amt = next((o["displayAmount"] for o in outcome if o["assetName"] == "USDC"), 0)
                        detail = f"{usdc_amt:.2f} USDC -> {wbtc_amt:.8f} WBTC"
                    elif "WBTC" in out_names and "USDC" in in_names:
                        direction = "SELL WBTC"
                        wbtc_amt = next((o["displayAmount"] for o in outcome if o["assetName"] == "WBTC"), 0)
                        usdc_amt = next((i["displayAmount"] for i in income if i["assetName"] == "USDC"), 0)
                        detail = f"{wbtc_amt:.8f} WBTC -> {usdc_amt:.2f} USDC"
                    else:
                        direction = t.get("displayType", "Trade")
                        detail = ", ".join(
                            [f"{o['displayAmount']} {o['assetName']}" for o in outcome]
                            + [f"-> {i['displayAmount']} {i['assetName']}" for i in income]
                        )

                    tx_short = t["txHash"][:10] + "..."
                    tx_link = f"https://polygonscan.com/tx/{t['txHash']}"
                    rows.append({
                        "Date": ts.strftime("%Y-%m-%d %H:%M UTC"),
                        "Direction": direction,
                        "Détail": detail,
                        "TX": tx_short,
                        "_link": tx_link,
                    })

                # Display as markdown table with links
                header = "| Date | Direction | Détail | TX |"
                separator = "|------|-----------|--------|-----|"
                table_rows = [header, separator]
                for r in rows:
                    table_rows.append(
                        f"| {r['Date']} | {r['Direction']} | {r['Détail']} | [{r['TX']}]({r['_link']}) |"
                    )
                st.markdown("\n".join(table_rows))

            st.caption("Données live via dHEDGE GraphQL API — rafraîchies toutes les 10 min")

# =============================================================================
# TAB 2: PROJET
# =============================================================================

with tab_projet:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Méthodologie")
        st.markdown("""
        **Walk-Forward Validation** sur fenêtre glissante de 1500 jours :
        chaque jour, les 4 modèles sont ré-entraînés sur les 1500 jours précédents
        et prédisent la direction du BTC pour le lendemain.

        | Composant | Détail |
        |-----------|--------|
        | **Modèle** | Ensemble : XGBoost + LightGBM + CatBoost + GRU |
        | **Features** | 30 = 14 techniques + 8 macro + 3 régime + 5 feedback |
        | **Deep Learning** | GRU 2 couches + Attention Temporelle + MC Dropout |
        | **Cible** | Direction du prix BTC à J+1 (hausse/baisse) |
        | **Sizing** | EarningStrategy (threshold 5%, deadzone 5%) |
        | **Frais** | 0.1% par portion rééquilibrée |
        """)

        st.divider()

        st.header("Découvertes clés")
        st.markdown("""
        - **Ensemble 4 modèles (+14.4% edge)** : la combinaison pondérée
          XGBoost (35%), LightGBM (25%), CatBoost (25%) et GRU (15%)
          surpasse chaque modèle individuel.

        - **GRU avec attention temporelle** : capture les dépendances
          séquentielles sur 30 jours que les arbres ne voient pas.
          MC Dropout (20 passes) estime l'incertitude et réduit
          dynamiquement le poids du GRU quand il doute.

        - **Regime features** : volatility\_regime, trend\_strength et
          volume\_acceleration permettent au modèle de détecter
          implicitement les changements de régime de marché.

        - **Feedback features** : le modèle reçoit ses propres
          performances récentes (accuracy 5/10/20j, erreurs consécutives)
          pour moduler sa confiance.
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
GitHub Actions — 00:30 UTC

  Python:
  1. Fetch données BTC + macro
  2. Feature engineering (30)
  3. Train 3 modèles trees (1500j)
  4. Train GRU (30j lookback)
  5. Ensemble pondéré → signal.json

  TypeScript:
  6. Lire signal.json
  7. Lire position vault
  8. Calculer swap nécessaire
  9. Exécuter via dHEDGE SDK
  10. Notification Telegram
        """, language="text")

# =============================================================================
# TAB 3: STACK TECHNIQUE
# =============================================================================

with tab_stack:
    st.header("Architecture")

    st.code("""
Python (cron 00:30 UTC)                TypeScript (cron 01:00 UTC)
┌───────────────────────┐              ┌─────────────────────────┐
│ generate_signal.py     │              │ hedge-bot/src/index.ts  │
│  1. Fetch données      │  signal.json │  1. Lire signal.json    │
│  2. Features 30        │ ──────────► │  2. Lire position vault │
│  3. Train 3 trees      │              │  3. Calculer swap       │
│  4. Train GRU (30j)    │              │  4. Exécuter via SDK    │
│  5. Ensemble pondéré   │              │  5. Logger résultat     │
│  6. Écrire signal      │              │                         │
└───────────────────────┘              └─────────────────────────┘
    """, language="text")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **ML & Data (Python)**
        - XGBoost, LightGBM, CatBoost
        - PyTorch (GRU + Attention)
        - scikit-learn (StandardScaler)
        - Pandas, NumPy
        - Yahoo Finance (données)
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
        - GitHub Actions (cron quotidien)
        - Telegram Bot (monitoring)
        - Polygon RPC (PublicNode)
        - Node.js 22 + Python 3.12
        - dHEDGE vault on-chain
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
