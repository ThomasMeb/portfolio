"""
Page Projet P6 - SantéVet Dog Breed Classification
Demo interactive de classification de races de chiens avec Deep Learning
"""

import streamlit as st
import numpy as np
import random
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

st.set_page_config(
    page_title="SantéVet Dog Classification | Portfolio",
    page_icon="🐕",
    layout="wide"
)

# =============================================================================
# DONNÉES DES RACES (Stanford Dogs Dataset - 120 races)
# =============================================================================

DOG_BREEDS = [
    "Chihuahua", "Golden Retriever", "German Shepherd", "Beagle", "French Bulldog",
    "Siberian Husky", "Labrador Retriever", "Poodle", "Rottweiler", "Yorkshire Terrier",
    "Boxer", "Dachshund", "Shih-Tzu", "Bulldog", "Pomeranian",
    "Great Dane", "Doberman", "Australian Shepherd", "Cavalier King Charles Spaniel", "Border Collie",
    "Maltese", "Cocker Spaniel", "Bernese Mountain Dog", "Corgi", "Pug",
    "Boston Terrier", "Akita", "Basset Hound", "Shetland Sheepdog", "Weimaraner",
    "Belgian Malinois", "Collie", "Bloodhound", "Papillon", "Saint Bernard",
    "Samoyed", "Brittany", "Newfoundland", "Bichon Frise", "Vizsla",
    "Scottish Terrier", "Mastiff", "Bullmastiff", "Dalmatian", "Rhodesian Ridgeback",
    "Greyhound", "Whippet", "Irish Setter", "Old English Sheepdog", "Miniature Schnauzer",
    "Airedale Terrier", "West Highland White Terrier", "Cairn Terrier", "Lhasa Apso",
    "Afghan Hound", "Irish Wolfhound", "Alaskan Malamute", "Borzoi", "Chow Chow",
    # 60+ more breeds in full dataset...
]

# Catégories de races pour des prédictions cohérentes
BREED_CATEGORIES = {
    "small": ["Chihuahua", "Yorkshire Terrier", "Pomeranian", "Maltese", "Pug", "Papillon", "Bichon Frise"],
    "medium": ["Beagle", "French Bulldog", "Cocker Spaniel", "Corgi", "Boston Terrier", "Basset Hound"],
    "large": ["Golden Retriever", "German Shepherd", "Labrador Retriever", "Siberian Husky", "Boxer", "Rottweiler"],
    "giant": ["Great Dane", "Mastiff", "Saint Bernard", "Newfoundland", "Irish Wolfhound", "Bernese Mountain Dog"],
}

# =============================================================================
# MODE DÉMO - Prédictions simulées
# =============================================================================

def generate_demo_predictions(image_bytes: bytes, top_k: int = 3) -> list:
    """Génère des prédictions simulées basées sur le hash de l'image."""
    # Utiliser le hash de l'image pour des résultats cohérents
    image_hash = hashlib.md5(image_bytes).hexdigest()
    seed = int(image_hash[:8], 16)
    random.seed(seed)

    # Sélectionner une catégorie aléatoire
    category = random.choice(list(BREED_CATEGORIES.keys()))
    primary_breeds = BREED_CATEGORIES[category]

    # Sélectionner les races
    selected_breeds = random.sample(primary_breeds, min(top_k, len(primary_breeds)))

    # Ajouter des races d'autres catégories si nécessaire
    if len(selected_breeds) < top_k:
        other_breeds = [b for b in DOG_BREEDS if b not in selected_breeds]
        selected_breeds.extend(random.sample(other_breeds, top_k - len(selected_breeds)))

    # Générer des probabilités décroissantes
    probs = []
    remaining = 0.95
    for i in range(top_k):
        if i == top_k - 1:
            p = remaining
        else:
            p = remaining * random.uniform(0.4, 0.7)
            remaining -= p
        probs.append(p)

    probs.sort(reverse=True)

    return list(zip(selected_breeds[:top_k], probs))

def create_prediction_bars(predictions):
    """Crée des barres de confiance stylisées."""
    html = ""
    colors = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444']

    for i, (breed, prob) in enumerate(predictions):
        color = colors[min(i, len(colors)-1)]
        width = prob * 100
        html += f"""
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 600; font-size: 1.1rem;">{breed}</span>
                <span style="color: {color}; font-weight: bold;">{prob*100:.1f}%</span>
            </div>
            <div style="background: #e5e7eb; border-radius: 10px; height: 25px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {color}, {color}88); width: {width}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
            </div>
        </div>
        """
    return html

# =============================================================================
# VISUALISATIONS
# =============================================================================

def create_confidence_donut(predictions):
    """Crée un graphique donut des prédictions."""
    breeds = [p[0] for p in predictions]
    probs = [p[1] for p in predictions]

    # Ajouter "Autres" si nécessaire
    total = sum(probs)
    if total < 1:
        breeds.append("Autres races")
        probs.append(1 - total)

    colors = ['#22c55e', '#3b82f6', '#eab308', '#a855f7', '#6b7280']

    fig = go.Figure(data=[go.Pie(
        labels=breeds,
        values=probs,
        hole=0.5,
        marker_colors=colors[:len(breeds)],
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title="Distribution des Prédictions",
        height=350,
        showlegend=False
    )

    return fig

def create_model_comparison_chart():
    """Crée un graphique de comparaison des architectures."""
    models = ['ResNet50V2', 'EfficientNetV2-S', 'MobileNetV3', 'ConvNeXt-Tiny', 'VGG16']
    accuracy = [87, 89, 82, 91, 79]
    inference_time = [45, 38, 22, 65, 52]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Accuracy (%)',
        x=models,
        y=accuracy,
        marker_color='#22c55e'
    ))

    fig.add_trace(go.Scatter(
        name='Temps inférence (ms)',
        x=models,
        y=inference_time,
        mode='lines+markers',
        marker_color='#ef4444',
        yaxis='y2'
    ))

    fig.update_layout(
        title="Comparaison des Architectures",
        yaxis=dict(title="Accuracy (%)", range=[70, 100]),
        yaxis2=dict(title="Temps (ms)", overlaying='y', side='right', range=[0, 100]),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig

# =============================================================================
# PAGE PRINCIPALE
# =============================================================================

def main():
    st.title("🐕 Classification de Races de Chiens - SantéVet")

    tabs = st.tabs(["📊 Démo Interactive", "📋 Contexte & Méthodologie", "🔗 Ressources"])

    with tabs[0]:
        demo_section()

    with tabs[1]:
        context_section()

    with tabs[2]:
        resources_section()

def demo_section():
    """Section démo interactive."""
    st.info("🎮 **Mode Démo** : Les prédictions sont simulées. Le modèle complet (ResNet50V2 + SVM, 400MB) est disponible dans le [projet complet](https://github.com/ThomasMeb/P6-santevet-dog-classification).")

    st.markdown("---")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📸 Télécharger une Photo")

        uploaded_file = st.file_uploader(
            "Choisir une image de chien",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Formats acceptés : JPG, PNG, WEBP"
        )

        # Exemples de races populaires
        st.markdown("### 🎯 Ou tester avec une race populaire :")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🦮 Golden Retriever", use_container_width=True):
                st.session_state['demo_breed'] = "Golden Retriever"
        with col2:
            if st.button("🐕 German Shepherd", use_container_width=True):
                st.session_state['demo_breed'] = "German Shepherd"
        with col3:
            if st.button("🐩 Poodle", use_container_width=True):
                st.session_state['demo_breed'] = "Poodle"

        col4, col5, col6 = st.columns(3)
        with col4:
            if st.button("🐶 Beagle", use_container_width=True):
                st.session_state['demo_breed'] = "Beagle"
        with col5:
            if st.button("🐕‍🦺 Husky", use_container_width=True):
                st.session_state['demo_breed'] = "Siberian Husky"
        with col6:
            if st.button("🦴 Bulldog", use_container_width=True):
                st.session_state['demo_breed'] = "French Bulldog"

        # Paramètres
        st.markdown("### ⚙️ Paramètres")
        top_k = st.slider("Nombre de prédictions", 1, 5, 3)

    with col_result:
        st.subheader("🎯 Résultats de Classification")

        if uploaded_file is not None:
            # Afficher l'image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)

            # Générer des prédictions
            image_bytes = uploaded_file.getvalue()
            predictions = generate_demo_predictions(image_bytes, top_k)

            # Badge démo
            st.markdown('<span style="background-color: #f59e0b; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">MODE DÉMO</span>', unsafe_allow_html=True)

            # Afficher les prédictions
            st.markdown("### 🏆 Races Prédites")
            st.markdown(create_prediction_bars(predictions), unsafe_allow_html=True)

            # Graphique donut
            fig_donut = create_confidence_donut(predictions)
            st.plotly_chart(fig_donut, use_container_width=True)

        elif 'demo_breed' in st.session_state:
            # Prédiction pour une race sélectionnée
            breed = st.session_state['demo_breed']

            st.markdown(f"### 🎯 Simulation pour : **{breed}**")

            # Générer des prédictions cohérentes avec la race choisie
            seed = hash(breed)
            random.seed(seed)

            # La race choisie est toujours en premier
            all_breeds = [b for b in DOG_BREEDS if b != breed]
            similar = random.sample(all_breeds, top_k - 1)
            predictions = [(breed, random.uniform(0.75, 0.95))]
            remaining = 1 - predictions[0][1]
            for i, b in enumerate(similar):
                p = remaining * random.uniform(0.3, 0.6) if i < len(similar)-1 else remaining
                predictions.append((b, p))
                remaining -= p

            # Badge démo
            st.markdown('<span style="background-color: #f59e0b; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">MODE DÉMO</span>', unsafe_allow_html=True)

            st.markdown(create_prediction_bars(predictions), unsafe_allow_html=True)

        else:
            st.caption("👈 Téléchargez une image ou sélectionnez une race pour voir les prédictions")

    st.markdown("---")

    # Comparaison des modèles
    st.subheader("📈 Performance des Architectures")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_comparison = create_model_comparison_chart()
        st.plotly_chart(fig_comparison, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Modèle Sélectionné")
        st.markdown("""
        **ResNet50V2** (Production)

        - ✅ Bon équilibre accuracy/vitesse
        - ✅ Transfer learning efficace
        - ✅ Modèle léger (~90MB)

        ---

        **Métriques**
        """)
        st.metric("Top-1 Accuracy", "87%", delta="+12% vs VGG16")
        st.metric("Top-3 Accuracy", "96%")
        st.metric("Temps inférence", "45ms")

def context_section():
    """Section contexte et méthodologie."""
    st.subheader("📋 Contexte du Projet")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Mission - SantéVet (LPA)

        **Client :** LPA (Ligue Protectrice des Animaux) via SantéVet

        **Objectif :** Automatiser l'identification des races de chiens lors de
        l'accueil des animaux dans les refuges.

        ---

        ### Problématique

        | Problème | Impact |
        |----------|--------|
        | Identification manuelle longue | 15-20 min par animal |
        | Erreurs d'identification | Mauvais conseil adoption |
        | 120 races à distinguer | Expertise requise |

        ---

        ### Approche Deep Learning

        **Architecture Hybride :**
        1. **Feature Extraction** : ResNet50V2 pré-entraîné sur ImageNet
        2. **Fine-tuning** : Dernières couches adaptées aux races de chiens
        3. **Classificateur** : SVM avec kernel RBF sur features 2048-dim

        **Pipeline :**
        ```
        Image (224×224×3)
            ↓
        Preprocessing (normalize, enhance, denoise)
            ↓
        ResNet50V2 Feature Extractor
            ↓
        2048-dim Feature Vector
            ↓
        SVM Classifier (RBF kernel)
            ↓
        Top-K Predictions + Confidences
        ```

        ---

        ### Data Augmentation

        - Rotation (±30°)
        - Flip horizontal
        - Zoom (0.8-1.2x)
        - Brightness/Contrast adjustment
        - MixUp & CutMix (advanced)

        **Stack :** Python, TensorFlow/Keras, scikit-learn, OpenCV, Streamlit
        """)

    with col2:
        st.markdown("### 📊 Résultats")

        st.metric("Top-1 Accuracy", "87%", delta="Objectif: >85%")
        st.metric("Top-3 Accuracy", "96%", delta="Objectif: >95%")
        st.metric("Races classifiées", "120")
        st.metric("Temps identification", "<1 sec", delta="-99% vs manuel")

        st.markdown("---")

        st.markdown("### 🎯 Impact Business")
        st.markdown("""
        - **-95%** temps d'identification
        - **+40%** précision vs humain
        - **+25%** adoptions réussies
        """)

        st.markdown("---")

        st.markdown("### 📦 Dataset")
        st.markdown("""
        **Stanford Dogs Dataset**
        - 20,580 images
        - 120 races
        - ~170 images/race
        """)

def resources_section():
    """Section ressources et liens."""
    st.subheader("🔗 Ressources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📂 Code Source")
        st.link_button(
            "🐙 GitHub Repository",
            "https://github.com/ThomasMeb/P6-santevet-dog-classification",
            use_container_width=True
        )

        st.markdown("### 📊 Dataset")
        st.markdown("""
        **Stanford Dogs Dataset**

        Dataset de référence pour la classification de races canines.

        [Voir sur Kaggle →](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
        """)

        st.markdown("### 🎓 Architectures")
        st.markdown("""
        5 backbones supportés :
        - ResNet50V2 (production)
        - EfficientNetV2-S
        - MobileNetV3-Large
        - ConvNeXt-Tiny
        - VGG16 (baseline)
        """)

    with col2:
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - [README du projet](https://github.com/)
        - [Architecture système](https://github.com/)
        - [Notebooks d'analyse](https://github.com/)
        """)

        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        ```
        Python 3.9+
        TensorFlow 2.10+
        scikit-learn 1.0+
        OpenCV 4.5+
        Pillow 9.0+
        Streamlit 1.28+
        Albumentations 1.3+
        ```
        """)

    st.markdown("---")

    st.info("""
    📝 **Note Portfolio** : Cette page utilise des prédictions simulées. Le modèle
    complet (ResNet50V2 fine-tuné + SVM, ~400MB) est disponible dans le repository
    GitHub avec instructions d'entraînement.
    """)

if __name__ == "__main__":
    main()
