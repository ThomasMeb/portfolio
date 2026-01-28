"""
Page Projet P5 - StackOverflow NLP Tag Suggestion
Demo interactive de classification multi-label pour tags
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter

st.set_page_config(
    page_title="StackOverflow NLP | Portfolio",
    page_icon="🏷️",
    layout="wide"
)

# =============================================================================
# SYSTÈME DE SUGGESTION PAR MOTS-CLÉS (Mode Démo)
# =============================================================================

# Mapping des mots-clés vers les tags
KEYWORD_TAG_MAP = {
    # Python ecosystem
    'python': ['python'],
    'pandas': ['python', 'pandas', 'dataframe'],
    'numpy': ['python', 'numpy'],
    'django': ['python', 'django', 'web'],
    'flask': ['python', 'flask', 'web'],
    'fastapi': ['python', 'fastapi', 'api'],
    'tensorflow': ['python', 'tensorflow', 'machine-learning'],
    'keras': ['python', 'keras', 'deep-learning'],
    'pytorch': ['python', 'pytorch', 'deep-learning'],
    'scikit': ['python', 'scikit-learn', 'machine-learning'],
    'sklearn': ['python', 'scikit-learn', 'machine-learning'],
    'matplotlib': ['python', 'matplotlib', 'visualization'],
    'seaborn': ['python', 'seaborn', 'visualization'],
    'plotly': ['python', 'plotly', 'visualization'],
    'jupyter': ['python', 'jupyter-notebook'],
    'pip': ['python', 'pip'],
    'venv': ['python', 'virtualenv'],
    'pytest': ['python', 'pytest', 'testing'],

    # JavaScript ecosystem
    'javascript': ['javascript'],
    'typescript': ['typescript', 'javascript'],
    'react': ['javascript', 'reactjs', 'frontend'],
    'reactjs': ['javascript', 'reactjs', 'frontend'],
    'vue': ['javascript', 'vue.js', 'frontend'],
    'angular': ['javascript', 'angular', 'frontend'],
    'nodejs': ['javascript', 'node.js', 'backend'],
    'node': ['javascript', 'node.js', 'backend'],
    'express': ['javascript', 'node.js', 'express'],
    'npm': ['javascript', 'npm'],
    'webpack': ['javascript', 'webpack'],
    'async': ['javascript', 'async-await'],
    'await': ['javascript', 'async-await'],
    'promise': ['javascript', 'promises'],

    # Data & ML
    'machine learning': ['machine-learning'],
    'deep learning': ['deep-learning'],
    'neural network': ['neural-network', 'deep-learning'],
    'nlp': ['nlp', 'machine-learning'],
    'bert': ['nlp', 'transformers', 'deep-learning'],
    'transformer': ['nlp', 'transformers', 'deep-learning'],
    'classification': ['classification', 'machine-learning'],
    'regression': ['regression', 'machine-learning'],
    'clustering': ['clustering', 'machine-learning'],

    # Databases
    'sql': ['sql', 'database'],
    'mysql': ['mysql', 'sql', 'database'],
    'postgresql': ['postgresql', 'sql', 'database'],
    'mongodb': ['mongodb', 'database', 'nosql'],
    'redis': ['redis', 'database', 'caching'],

    # DevOps
    'docker': ['docker', 'containers'],
    'kubernetes': ['kubernetes', 'docker', 'devops'],
    'k8s': ['kubernetes', 'docker', 'devops'],
    'aws': ['aws', 'cloud'],
    'azure': ['azure', 'cloud'],
    'gcp': ['google-cloud', 'cloud'],
    'ci/cd': ['ci-cd', 'devops'],
    'github actions': ['github-actions', 'ci-cd'],

    # Web
    'html': ['html', 'web'],
    'css': ['css', 'web'],
    'flexbox': ['css', 'flexbox'],
    'api': ['api', 'rest'],
    'rest': ['api', 'rest'],
    'graphql': ['graphql', 'api'],
    'json': ['json'],

    # Git
    'git': ['git', 'version-control'],
    'merge': ['git', 'merge-conflict'],
    'branch': ['git', 'branching'],

    # Java
    'java': ['java'],
    'spring': ['java', 'spring-boot'],
    'maven': ['java', 'maven'],
}

def predict_tags_keyword(text: str, top_k: int = 5) -> list:
    """Prédit les tags basé sur les mots-clés présents dans le texte."""
    text_lower = text.lower()
    tag_scores = Counter()

    for keyword, tags in KEYWORD_TAG_MAP.items():
        if keyword in text_lower:
            for tag in tags:
                tag_scores[tag] += 1

    # Si aucun tag trouvé, retourner des tags génériques
    if not tag_scores:
        return [('programming', 0.5), ('question', 0.4)]

    # Normaliser les scores
    max_score = max(tag_scores.values())
    results = [
        (tag, min(0.95, score / max_score * 0.8 + 0.15))
        for tag, score in tag_scores.most_common(top_k)
    ]

    return results

# =============================================================================
# EXEMPLES DE QUESTIONS
# =============================================================================

SAMPLE_QUESTIONS = [
    {
        "title": "How to parse JSON in Python?",
        "body": "I have a JSON string and I want to convert it to a Python dictionary. I tried using the json module but I'm getting errors.",
        "expected_tags": ["python", "json", "parsing"]
    },
    {
        "title": "React useState not updating immediately",
        "body": "I'm using useState in my React component but when I call setState, the value doesn't update immediately.",
        "expected_tags": ["javascript", "reactjs", "react-hooks"]
    },
    {
        "title": "TensorFlow model not converging during training",
        "body": "My neural network model isn't converging during training. The loss stays constant after a few epochs.",
        "expected_tags": ["python", "tensorflow", "machine-learning"]
    },
    {
        "title": "Docker container cannot connect to localhost",
        "body": "I have a Docker container running my Node.js app, but it cannot connect to a service running on localhost:5432.",
        "expected_tags": ["docker", "node.js", "postgresql"]
    }
]

# =============================================================================
# VISUALISATIONS
# =============================================================================

def create_confidence_chart(predictions):
    """Crée un graphique de confiance pour les prédictions."""
    tags = [p[0] for p in predictions]
    scores = [p[1] * 100 for p in predictions]

    # Couleurs basées sur la confiance
    colors = ['#22c55e' if s >= 70 else '#eab308' if s >= 40 else '#ef4444' for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=tags,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.0f}%" for s in scores],
        textposition='auto'
    ))

    fig.update_layout(
        title="Confiance des Prédictions",
        xaxis_title="Confiance (%)",
        yaxis_title="Tag",
        yaxis={'categoryorder': 'total ascending'},
        height=300,
        margin=dict(l=100)
    )

    return fig

def create_tag_distribution_chart():
    """Crée un graphique de distribution des tags (simulé)."""
    # Distribution simulée basée sur Stack Overflow réel
    top_tags = {
        'python': 2100000,
        'javascript': 2300000,
        'java': 1800000,
        'c#': 1500000,
        'php': 1400000,
        'android': 1300000,
        'html': 1100000,
        'jquery': 1000000,
        'c++': 780000,
        'css': 750000
    }

    fig = px.bar(
        x=list(top_tags.keys()),
        y=list(top_tags.values()),
        title="Top 10 Tags Stack Overflow (2024)",
        labels={'x': 'Tag', 'y': 'Questions'},
        color=list(top_tags.values()),
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=350, showlegend=False)
    return fig

# =============================================================================
# PAGE PRINCIPALE
# =============================================================================

def main():
    st.title("🏷️ Suggestion de Tags NLP - Stack Overflow")

    tabs = st.tabs(["📊 Démo Interactive", "📋 Contexte & Méthodologie", "🔗 Ressources"])

    with tabs[0]:
        demo_section()

    with tabs[1]:
        context_section()

    with tabs[2]:
        resources_section()

def demo_section():
    """Section démo interactive."""
    st.info("🎮 **Mode Démo** : Cette version utilise un système de mots-clés. Le modèle ML complet (TF-IDF + BERT) est disponible dans le [projet complet](https://classifier-questions-stackoverflow.streamlit.app/).")

    st.markdown("---")

    # Inputs
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("📝 Votre Question")

        # Exemples rapides
        example_idx = st.selectbox(
            "Choisir un exemple :",
            ["(Écrire ma propre question)"] + [q["title"] for q in SAMPLE_QUESTIONS]
        )

        if example_idx == "(Écrire ma propre question)":
            title = st.text_input(
                "Titre de la question",
                placeholder="Ex: How to parse JSON in Python?"
            )
            body = st.text_area(
                "Corps de la question",
                placeholder="Décrivez votre problème en détail...",
                height=150
            )
        else:
            idx = [q["title"] for q in SAMPLE_QUESTIONS].index(example_idx)
            title = st.text_input("Titre de la question", value=SAMPLE_QUESTIONS[idx]["title"])
            body = st.text_area(
                "Corps de la question",
                value=SAMPLE_QUESTIONS[idx]["body"],
                height=150
            )
            st.caption(f"Tags attendus : {', '.join(SAMPLE_QUESTIONS[idx]['expected_tags'])}")

        # Paramètres
        top_k = st.slider("Nombre de tags", 3, 10, 5)
        threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.3, 0.05)

    with col_result:
        st.subheader("🏷️ Tags Suggérés")

        if title or body:
            text = f"{title} {body}"
            predictions = predict_tags_keyword(text, top_k)

            # Filtrer par seuil
            predictions = [(tag, score) for tag, score in predictions if score >= threshold]

            if predictions:
                # Afficher les tags
                tags_html = " ".join([
                    f'<span style="background-color: {"#22c55e" if s >= 0.7 else "#eab308" if s >= 0.4 else "#ef4444"}; '
                    f'color: white; padding: 5px 12px; border-radius: 15px; margin: 3px; display: inline-block;">'
                    f'{tag} ({s*100:.0f}%)</span>'
                    for tag, s in predictions
                ])
                st.markdown(f'<div style="margin: 1rem 0;">{tags_html}</div>', unsafe_allow_html=True)

                # Graphique de confiance
                fig = create_confidence_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucun tag trouvé avec ce seuil de confiance.")
        else:
            st.caption("👈 Entrez une question pour voir les suggestions de tags")

    st.markdown("---")

    # Visualisation des tags Stack Overflow
    st.subheader("📈 Distribution des Tags Stack Overflow")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_dist = create_tag_distribution_chart()
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.markdown("### 📊 Statistiques")
        st.metric("Questions analysées", "50M+")
        st.metric("Tags uniques", "65,000+")
        st.metric("Questions/jour", "~8,000")

        st.markdown("---")

        st.markdown("### 🎯 Performance Modèle")
        st.metric("Precision@5", "78%", delta="+8% vs baseline")
        st.metric("Recall@5", "62%")
        st.metric("F1-Score", "0.69")

def context_section():
    """Section contexte et méthodologie."""
    st.subheader("📋 Contexte du Projet")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Mission - IntelliTag (Stack Overflow)

        **Client :** Stack Overflow - Équipe Qualité des Contenus

        **Objectif :** Développer un système de suggestion automatique de tags
        pour améliorer la catégorisation des questions.

        ---

        ### Problématique

        | Problème | Impact |
        |----------|--------|
        | 45% des questions mal tagguées | Difficultés de recherche |
        | Tags manquants | Questions sans réponse |
        | Inconsistance | Base de connaissances fragmentée |

        ---

        ### Approche Multi-Modèle

        Le système compare **4 techniques d'extraction de features** :

        1. **TF-IDF (Bag-of-Words)** - Baseline rapide et interprétable
        2. **Word2Vec** - Embeddings de mots (300 dimensions)
        3. **BERT** - Embeddings contextuels (768 dimensions)
        4. **USE** - Universal Sentence Encoder (512 dimensions)

        **Classifieur :** OneVsRest avec Logistic Regression

        ---

        ### Pipeline NLP

        ```
        Question (titre + corps)
            ↓
        Preprocessing (HTML cleaning, tokenization, lemmatization)
            ↓
        Feature Extraction (TF-IDF / BERT / USE)
            ↓
        Multi-Label Classification
            ↓
        Tags suggérés + Scores de confiance
        ```

        **Stack :** Python, scikit-learn, Transformers, FastAPI, Streamlit
        """)

    with col2:
        st.markdown("### 📊 Résultats")

        st.metric("Precision@5", "78%", delta="Objectif: >70%")
        st.metric("Recall@5", "62%", delta="Objectif: >50%")
        st.metric("F1-Score", "0.69", delta="Objectif: >0.60")
        st.metric("Latence API", "145ms", delta="Objectif: <200ms")

        st.markdown("---")

        st.markdown("### 🎯 Impact Business")
        st.markdown("""
        - **-31%** corrections manuelles modérateurs
        - **+52%** adoption par utilisateurs
        - **+18%** questions avec tags pertinents
        """)

        st.markdown("---")

        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        - **API** : FastAPI (REST)
        - **Demo** : Streamlit
        - **Tests** : 84 tests, 85% coverage
        - **CI/CD** : GitHub Actions
        """)

def resources_section():
    """Section ressources et liens."""
    st.subheader("🔗 Ressources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📂 Code Source")
        st.link_button(
            "🐙 GitHub Repository",
            "https://github.com/ThomasMeb/P5-stackoverflow-nlp-tags",
            use_container_width=True
        )

        st.markdown("### 🎮 Démo Complète")
        st.link_button(
            "🚀 Streamlit App (Production)",
            "https://classifier-questions-stackoverflow.streamlit.app/",
            use_container_width=True,
            type="primary"
        )

        st.markdown("### 📊 Dataset")
        st.markdown("""
        **Stack Overflow Python Questions**

        ~900MB de questions tagguées pour l'entraînement.

        [Voir sur Kaggle →](https://www.kaggle.com/datasets/stackoverflow/pythonquestions)
        """)

    with col2:
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - [Architecture du système](https://github.com/)
        - [Product Requirements](https://github.com/)
        - [API Documentation](https://github.com/)
        """)

        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        ```
        Python 3.9+
        scikit-learn 1.3+
        TensorFlow 2.13+
        Transformers 4.30+
        FastAPI 0.100+
        Streamlit 1.28+
        NLTK, Gensim
        ```
        """)

    st.markdown("---")

    st.info("""
    📝 **Note Portfolio** : Cette page utilise un système simplifié de suggestion par mots-clés.
    Le modèle ML complet (TF-IDF + classificateur multi-label) est disponible dans la
    [démo production](https://classifier-questions-stackoverflow.streamlit.app/).
    """)

if __name__ == "__main__":
    main()
