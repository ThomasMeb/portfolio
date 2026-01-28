"""
Page Projet Alla2 - Bitcoin Trading Prediction
Demo interactive de prédiction de direction du prix BTC avec XGBoost
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="Alla2 Trading | Portfolio",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# DONNÉES SIMULÉES POUR LA DÉMO
# =============================================================================

def generate_btc_data(days: int = 365) -> pd.DataFrame:
    """Génère des données BTC simulées avec indicateurs techniques."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Prix simulé avec tendance et volatilité
    base_price = 30000
    returns = np.random.normal(0.001, 0.03, days)
    prices = base_price * np.cumprod(1 + returns)

    # Indicateurs techniques simulés
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices * (1 + np.random.uniform(-0.02, 0.02, days)),
        'high': prices * (1 + np.random.uniform(0, 0.05, days)),
        'low': prices * (1 - np.random.uniform(0, 0.05, days)),
        'volume': np.random.uniform(1e9, 5e9, days),
    })

    # Calcul des indicateurs
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['rsi'] = 50 + np.random.uniform(-30, 30, days)  # Simplifié

    # Bollinger Bands
    df['sma_20'] = df['close'].rolling(20).mean()
    df['std_20'] = df['close'].rolling(20).std()
    df['bollinger_upper'] = df['sma_20'] + 2 * df['std_20']
    df['bollinger_lower'] = df['sma_20'] - 2 * df['std_20']

    # Target (direction)
    df['returns'] = df['close'].pct_change()
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    # Prédictions simulées du modèle (61% accuracy)
    correct_mask = np.random.random(days) < 0.61
    df['prediction'] = df['target'].copy()
    df.loc[~correct_mask, 'prediction'] = 1 - df.loc[~correct_mask, 'target']

    df['confidence'] = np.random.uniform(0.52, 0.85, days)

    return df.dropna()

def calculate_backtest_performance(df: pd.DataFrame) -> dict:
    """Calcule les métriques de performance du backtesting."""
    # Stratégie : investir selon la prédiction
    df = df.copy()
    df['strategy_returns'] = df['returns'] * df['prediction'].shift(1)
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_market'] = (1 + df['returns']).cumprod()

    # Métriques
    total_return = df['cumulative_strategy'].iloc[-1] - 1
    market_return = df['cumulative_market'].iloc[-1] - 1

    # Sharpe Ratio (annualisé)
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)

    # Max Drawdown
    rolling_max = df['cumulative_strategy'].cummax()
    drawdown = (df['cumulative_strategy'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win Rate
    win_rate = (df['strategy_returns'] > 0).mean()

    return {
        'total_return': total_return,
        'market_return': market_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'df': df
    }

# =============================================================================
# VISUALISATIONS
# =============================================================================

def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """Crée un graphique de prix avec indicateurs."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=['Prix BTC avec Bollinger Bands', 'MACD', 'RSI']
    )

    # Prix et Bollinger
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'],
        name='Prix', line=dict(color='#3b82f6', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['bollinger_upper'],
        name='Bollinger Upper', line=dict(color='#94a3b8', dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['bollinger_lower'],
        name='Bollinger Lower', line=dict(color='#94a3b8', dash='dash'),
        fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)'
    ), row=1, col=1)

    # MACD
    colors = ['#22c55e' if v > 0 else '#ef4444' for v in df['macd']]
    fig.add_trace(go.Bar(
        x=df['date'], y=df['macd'],
        name='MACD', marker_color=colors
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rsi'],
        name='RSI', line=dict(color='#a855f7')
    ), row=3, col=1)

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=600, showlegend=True)
    fig.update_yaxes(title_text="Prix ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    return fig

def create_backtest_chart(df: pd.DataFrame) -> go.Figure:
    """Crée un graphique de performance du backtesting."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cumulative_strategy'] * 100,
        name='Stratégie Alla2', line=dict(color='#22c55e', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cumulative_market'] * 100,
        name='Buy & Hold', line=dict(color='#3b82f6', width=2, dash='dash')
    ))

    fig.update_layout(
        title="Performance : Stratégie vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Valeur du Portfolio (%)",
        height=400,
        hovermode='x unified'
    )

    return fig

def create_feature_importance_chart() -> go.Figure:
    """Crée un graphique d'importance des features."""
    features = {
        'RSI': 0.18,
        'MACD': 0.15,
        'Volume relatif': 0.12,
        'Bollinger Position': 0.11,
        'EMA Cross': 0.10,
        'Momentum': 0.09,
        'ATR': 0.08,
        'Stochastic K': 0.07,
        'OBV': 0.05,
        'Autres': 0.05
    }

    df = pd.DataFrame({
        'Feature': list(features.keys()),
        'Importance': list(features.values())
    }).sort_values('Importance', ascending=True)

    fig = px.bar(
        df, x='Importance', y='Feature',
        orientation='h',
        title="Importance des Features (XGBoost)",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)

    return fig

def create_prediction_distribution() -> go.Figure:
    """Crée un graphique de distribution des prédictions."""
    # Distribution simulée des confiances
    correct_conf = np.random.beta(5, 3, 610)  # 61% correct
    incorrect_conf = np.random.beta(3, 5, 390)  # 39% incorrect

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=correct_conf, name='Prédictions Correctes',
        marker_color='#22c55e', opacity=0.7, nbinsx=20
    ))

    fig.add_trace(go.Histogram(
        x=incorrect_conf, name='Prédictions Incorrectes',
        marker_color='#ef4444', opacity=0.7, nbinsx=20
    ))

    fig.update_layout(
        title="Distribution des Confiances du Modèle",
        xaxis_title="Score de Confiance",
        yaxis_title="Nombre de Prédictions",
        barmode='overlay',
        height=350
    )

    return fig

# =============================================================================
# PAGE PRINCIPALE
# =============================================================================

def main():
    st.title("📈 Prédiction Trading BTC - Alla2")

    tabs = st.tabs(["📊 Démo Interactive", "📋 Contexte & Méthodologie", "🔗 Ressources"])

    with tabs[0]:
        demo_section()

    with tabs[1]:
        context_section()

    with tabs[2]:
        resources_section()

def demo_section():
    """Section démo interactive."""
    st.info("🎮 **Mode Démo** : Les données et prédictions sont simulées pour illustrer le fonctionnement du système. Le modèle réel utilise des données live BTC.")

    st.markdown("---")

    # Générer les données
    df = generate_btc_data(365)
    backtest = calculate_backtest_performance(df)

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "61%",
            delta="+11% vs random",
            help="Précision du modèle sur données de test"
        )

    with col2:
        st.metric(
            "Earn Metric",
            "1.10",
            delta="Profitable",
            help="Métrique personnalisée de rentabilité"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{backtest['sharpe_ratio']:.2f}",
            help="Ratio rendement/risque annualisé"
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{backtest['max_drawdown']*100:.1f}%",
            help="Perte maximale depuis le pic"
        )

    st.markdown("---")

    # Graphique de prix
    st.subheader("📉 Analyse Technique")
    fig_price = create_price_chart(df.tail(180))
    st.plotly_chart(fig_price, use_container_width=True)

    # Backtesting
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Performance Backtesting")
        fig_backtest = create_backtest_chart(backtest['df'])
        st.plotly_chart(fig_backtest, use_container_width=True)

    with col2:
        st.subheader("📊 Résultats")

        st.metric(
            "Rendement Stratégie",
            f"{backtest['total_return']*100:.1f}%"
        )
        st.metric(
            "Rendement Buy & Hold",
            f"{backtest['market_return']*100:.1f}%"
        )
        st.metric(
            "Win Rate",
            f"{backtest['win_rate']*100:.1f}%"
        )

        # Dernière prédiction
        st.markdown("---")
        st.markdown("### 🔮 Dernière Prédiction")

        last_pred = df.iloc[-1]['prediction']
        last_conf = df.iloc[-1]['confidence']

        if last_pred == 1:
            st.success(f"📈 **HAUSSE** attendue (Confiance: {last_conf*100:.0f}%)")
        else:
            st.error(f"📉 **BAISSE** attendue (Confiance: {last_conf*100:.0f}%)")

    st.markdown("---")

    # Feature Importance et Distribution
    col1, col2 = st.columns(2)

    with col1:
        fig_importance = create_feature_importance_chart()
        st.plotly_chart(fig_importance, use_container_width=True)

    with col2:
        fig_dist = create_prediction_distribution()
        st.plotly_chart(fig_dist, use_container_width=True)

def context_section():
    """Section contexte et méthodologie."""
    st.subheader("📋 Contexte du Projet")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Projet Personnel - Alla2

        **Objectif :** Prédire la direction du prix du Bitcoin (hausse/baisse)
        pour le jour suivant en utilisant des indicateurs techniques.

        ---

        ### Problématique

        | Défi | Approche |
        |------|----------|
        | Volatilité élevée | Indicateurs techniques multiples |
        | Bruit dans les données | Feature engineering robuste |
        | Classification binaire | XGBoost optimisé |
        | Évaluation réaliste | Métrique "earn_metric" personnalisée |

        ---

        ### Pipeline de Prédiction

        ```
        Données OHLCV (BTC-USD)
            ↓
        Feature Engineering (20+ indicateurs)
        ├── Moyennes mobiles (EMA 12, 26)
        ├── MACD et Signal
        ├── RSI (Relative Strength Index)
        ├── Bollinger Bands
        ├── Stochastic Oscillator
        ├── ATR (Average True Range)
        ├── OBV (On-Balance Volume)
        └── Momentum
            ↓
        StandardScaler (normalisation)
            ↓
        XGBoost Classifier
        (max_depth=5, n_estimators=200)
            ↓
        Prédiction : Hausse (1) / Baisse (0)
        + Score de confiance
        ```

        ---

        ### Métrique Personnalisée : Earn Metric

        ```python
        earn_metric = (gains_corrects - pertes_incorrectes) / capital_initial
        ```

        Cette métrique pénalise les faux positifs (achat avant baisse) plus
        que les faux négatifs, reflétant le risque asymétrique du trading.

        **Stack :** Python, XGBoost, scikit-learn, Pandas, TA-Lib
        """)

    with col2:
        st.markdown("### 📊 Résultats")

        st.metric("Accuracy", "61%", delta="+11% vs baseline")
        st.metric("Earn Metric", "1.10", delta="Profitable")
        st.metric("Données", "4,700+ jours", delta="2011-2024")
        st.metric("Features", "20+", delta="Indicateurs techniques")

        st.markdown("---")

        st.markdown("### 🧪 Modèles Testés")
        st.markdown("""
        | Modèle | Accuracy |
        |--------|----------|
        | **XGBoost** | **61%** ✅ |
        | Random Forest | 58% |
        | Logistic Reg. | 54% |
        | LSTM | 52% ❌ |
        | MLP | 55% |
        """)

        st.markdown("---")

        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        Ce projet est à but éducatif.
        Les performances passées ne
        garantissent pas les résultats futurs.
        Ne pas utiliser pour du trading réel.
        """)

def resources_section():
    """Section ressources et liens."""
    st.subheader("🔗 Ressources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📂 Code Source")
        st.link_button(
            "🐙 GitHub Repository",
            "https://github.com/ThomasMeb/tradebtcai",
            use_container_width=True
        )

        st.markdown("### 📊 Données")
        st.markdown("""
        **Bitcoin Historical Data**

        - Source : CryptoCompare API
        - Période : 2011 - présent
        - Fréquence : Daily OHLCV

        [CryptoCompare →](https://www.cryptocompare.com/)
        """)

        st.markdown("### 📈 Indicateurs Techniques")
        st.markdown("""
        - EMA (12, 26 périodes)
        - MACD + Signal
        - RSI (14 périodes)
        - Bollinger Bands (20, 2σ)
        - Stochastic Oscillator
        - ATR (14 périodes)
        - OBV (On-Balance Volume)
        - Momentum
        """)

    with col2:
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - [Rapport de projet (PDF)](https://github.com/)
        - [Notebooks d'analyse](https://github.com/)
        - [Feature Engineering](https://github.com/)
        """)

        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        ```
        Python 3.8+
        XGBoost 1.7+
        scikit-learn 1.0+
        Pandas
        NumPy
        Matplotlib
        SHAP (interpretability)
        ```
        """)

        st.markdown("### 🎓 Apprentissages")
        st.markdown("""
        - Importance du feature engineering
        - LSTM pas toujours supérieur
        - Métriques business > accuracy
        - Backtesting réaliste essentiel
        """)

    st.markdown("---")

    st.info("""
    📝 **Note Portfolio** : Ce projet explore la prédiction de direction de prix
    sur des données financières. Les résultats sont à but éducatif et ne constituent
    pas des conseils d'investissement.
    """)

if __name__ == "__main__":
    main()
