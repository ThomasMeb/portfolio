"""
Page Projet P3 - Schneider Electric Energy Prediction
Démo interactive de prédiction énergétique
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

st.set_page_config(
    page_title="Schneider Energy | Portfolio",
    page_icon="🔋",
    layout="wide"
)

# =============================================================================
# CHARGEMENT DES MODÈLES
# =============================================================================

@st.cache_resource
def load_models():
    """Charge les modèles ML pré-entraînés."""
    models = {}
    model_dir = Path("models/p3_schneider")

    try:
        models['energy_model'] = joblib.load(model_dir / "energy_model.joblib")
        models['energy_scaler'] = joblib.load(model_dir / "energy_scaler.joblib")
        models['energy_features'] = joblib.load(model_dir / "energy_features.joblib")
        models['co2_model'] = joblib.load(model_dir / "co2_model.joblib")
        models['co2_scaler'] = joblib.load(model_dir / "co2_scaler.joblib")
        return models
    except Exception as e:
        return None

def prepare_features(property_gfa, floors, age, energy_star, building_type, feature_names):
    """Prépare les features pour la prédiction."""
    features = {
        'Age': age,
        'NumberofBuildings': 1,
        'NumberofFloors': floors,
        'PropertyGFATotal': property_gfa,
        'PropertyGFAParking_Pct': 5.0,
        'PropertyGFABuilding_Pct': 95.0,
        'LargestPropertyUseTypeGFA': property_gfa * 0.8,
        'ENERGYSTARScore': energy_star
    }

    type_mapping = {
        "Bureau (Petit/Moyen)": "PropType_Small- and Mid-Sized Office",
        "Bureau (Grand)": "PropType_Large Office",
        "Hôtel": "PropType_Hotel",
        "Commerce": "PropType_Retail Store",
        "Entrepôt": "PropType_Warehouse",
        "École": "PropType_K-12 School",
        "Université": "PropType_University",
        "Hôpital": "PropType_Other",
        "Autre": "PropType_Other"
    }

    all_features = {name: 0 for name in feature_names}
    for key, value in features.items():
        if key in all_features:
            all_features[key] = value

    prop_type_col = type_mapping.get(building_type, "PropType_Other")
    if prop_type_col in all_features:
        all_features[prop_type_col] = 1

    df = pd.DataFrame([all_features])[feature_names]
    return df

def create_feature_importance_plot(model, feature_names):
    """Crée un graphique d'importance des features."""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    name_mapping = {
        'PropertyGFATotal': 'Surface totale',
        'LargestPropertyUseTypeGFA': 'Surface usage principal',
        'ENERGYSTARScore': 'Score ENERGY STAR',
        'Age': 'Âge du bâtiment',
        'NumberofFloors': "Nombre d'étages",
        'PropertyGFABuilding_Pct': '% Surface bâtiment',
        'PropertyGFAParking_Pct': '% Surface parking',
        'NumberofBuildings': 'Nombre de bâtiments'
    }

    importance_df['Feature_Display'] = importance_df['Feature'].map(
        lambda x: name_mapping.get(x, x.replace('PropType_', '').replace('District_', ''))
    )

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature_Display',
        orientation='h',
        title="Importance des Features (Random Forest)",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        height=350
    )
    return fig

# =============================================================================
# PAGE PRINCIPALE
# =============================================================================

def main():
    # Header avec contexte
    st.title("🔋 Prédiction Énergétique - Schneider Electric")

    # Tabs pour organisation
    tab1, tab2, tab3 = st.tabs(["📊 Démo Interactive", "📋 Contexte & Méthodologie", "🔗 Ressources"])

    with tab1:
        demo_section()

    with tab2:
        context_section()

    with tab3:
        resources_section()

def demo_section():
    """Section démo interactive."""
    models = load_models()

    if not models:
        st.warning("⚠️ Modèles non disponibles. Affichage en mode démonstration.")
        demo_mode = True
    else:
        demo_mode = False
        st.success("✅ Modèles ML chargés - Prédictions en temps réel")

    st.markdown("---")

    # Inputs
    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.subheader("🏢 Caractéristiques du Bâtiment")

        property_gfa = st.number_input(
            "Surface totale (m²)",
            min_value=100,
            max_value=200000,
            value=5000,
            step=100,
            help="Surface totale du bâtiment"
        )

        floors = st.slider("Nombre d'étages", 1, 50, 5)
        age = st.slider("Âge du bâtiment (années)", 0, 100, 30)
        energy_star = st.slider("Score ENERGY STAR", 1, 100, 50)

        building_type = st.selectbox(
            "Type de bâtiment",
            ["Bureau (Petit/Moyen)", "Bureau (Grand)", "Hôtel", "Commerce",
             "Entrepôt", "École", "Université", "Hôpital", "Autre"]
        )

    with col_results:
        st.subheader("📈 Résultats de Prédiction")

        # Conversion m² en sq ft pour le modèle
        property_gfa_sqft = property_gfa * 10.764

        if not demo_mode:
            # Prédiction réelle
            X = prepare_features(property_gfa_sqft, floors, age, energy_star,
                                building_type, models['energy_features'])
            X_scaled = models['energy_scaler'].transform(X)
            predicted_energy = models['energy_model'].predict(X_scaled)[0]
            predicted_co2 = models['co2_model'].predict(
                models['co2_scaler'].transform(X)
            )[0]
        else:
            # Mode démo - estimation heuristique
            base = property_gfa_sqft * 50
            predicted_energy = base * (1 + age/100) * (2 - energy_star/100)
            predicted_co2 = predicted_energy * 0.0001

        # Affichage des métriques
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                label="⚡ Consommation Énergétique",
                value=f"{predicted_energy/1e6:.2f} M kBtu/an",
                delta="Modèle ML" if not demo_mode else "Estimation"
            )

        with metric_col2:
            st.metric(
                label="🌿 Émissions CO2",
                value=f"{predicted_co2:.1f} tonnes/an",
                delta=f"~{predicted_co2 * 45:.0f} arbres/an"
            )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_energy / 1e6,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Consommation (M kBtu)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 20], 'color': "#4ade80"},
                    {'range': [20, 50], 'color': "#fbbf24"},
                    {'range': [50, 100], 'color': "#f87171"}
                ]
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature Importance (si modèles disponibles)
    if not demo_mode:
        st.markdown("---")
        st.subheader("🔍 Importance des Features")

        col1, col2 = st.columns(2)

        with col1:
            fig_importance = create_feature_importance_plot(
                models['energy_model'],
                models['energy_features']
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        with col2:
            st.markdown("### 💡 Insights Clés")
            st.markdown("""
            **Facteurs les plus impactants :**

            1. **Surface totale (42%)** - Principal prédicteur de consommation
            2. **Surface usage principal (19%)** - Type d'activité déterminant
            3. **Score ENERGY STAR (12%)** - L'efficacité réduit la consommation
            4. **Âge du bâtiment (8%)** - Les anciens bâtiments consomment plus
            5. **Nombre d'étages (5%)** - Complexité thermique

            *Modèle : Random Forest avec 45.5% d'amélioration vs baseline*
            """)

def context_section():
    """Section contexte et méthodologie."""
    st.subheader("📋 Contexte du Projet")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Mission Freelance - Schneider Electric

        **Client :** Schneider Electric — Direction Immobilier & RSE

        **Durée :** 4 semaines (Nov-Déc 2023)

        **Objectif :** Développer un outil ML prédisant la consommation énergétique
        et les émissions CO2 du parc immobilier tertiaire, dans le cadre des
        engagements ESG et neutralité carbone.

        ---

        ### Problématique

        | Problème | Impact |
        |----------|--------|
        | Analyse manuelle | 2-3 semaines par rapport |
        | Méthode statistique basique | Prédictions peu fiables |
        | Pas d'identification des facteurs | Impossible de prioriser |
        | Rapports statiques | Pas d'interactivité |

        ---

        ### Approche Technique

        1. **Exploration des données** - 3,376 bâtiments, 47 variables
        2. **Feature Engineering** - Traitement des 34% de valeurs manquantes
        3. **Modélisation** - 18 modèles testés, validation croisée 10-fold
        4. **Déploiement** - Application Streamlit avec SHAP

        **Stack :** Python, scikit-learn, XGBoost, Streamlit, SHAP
        """)

    with col2:
        st.markdown("### 📊 Résultats")

        st.metric("Amélioration vs Baseline", "+45.5%", delta="Objectif: 30%")
        st.metric("Temps d'analyse", "< 1 sec", delta="-99.9%")
        st.metric("Bâtiments analysés", "1,650")
        st.metric("Modèles comparés", "18")

        st.markdown("---")

        st.markdown("### 🏆 Modèle Final")
        st.markdown("""
        **Random Forest**
        - RMSE : 12.9M kBtu
        - R² : 0.83
        - Features : 40
        """)

def resources_section():
    """Section ressources et liens."""
    st.subheader("🔗 Ressources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📂 Code Source")
        st.link_button(
            "🐙 GitHub Repository",
            "https://github.com/ThomasMeb/P3-schneider-energy-prediction",
            use_container_width=True
        )

        st.markdown("### 📊 Dataset")
        st.markdown("""
        **Seattle Building Energy Benchmarking**

        Version portfolio utilisant des données publiques similaires
        aux données client (confidentielles).

        [Voir sur Kaggle →](https://www.kaggle.com/datasets/city-of-seattle/sea-building-energy-benchmarking)
        """)

    with col2:
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - [README du projet](https://github.com/)
        - [Notebooks d'analyse](https://github.com/)
        - [Rapport technique](https://github.com/)
        """)

        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        ```
        Python 3.8+
        scikit-learn 1.0+
        XGBoost
        Streamlit
        SHAP
        Plotly
        ```
        """)

    st.markdown("---")

    st.info("""
    📝 **Note Portfolio** : Ce repository est une version portfolio d'une mission
    freelance réalisée pour Schneider Electric. Les données client ont été
    remplacées par un dataset public similaire pour des raisons de confidentialité.
    """)

if __name__ == "__main__":
    main()
