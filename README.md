---
title: Thomas Portfolio
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🚀 Portfolio Thomas - ML Engineer & Entrepreneur

Bienvenue sur mon portfolio interactif présentant mes réalisations en **Machine Learning** et **Data Science**.

## 🎯 Projet Actuel

**[egir.app](https://egir.app)** - Plateforme SaaS de gestion pour restaurateurs avec IA intégrée.
- 📊 Calcul automatisé des coûts matières
- 🤖 Fiches techniques assistées par IA
- 📈 Dashboard d'analyse de rentabilité

## 💻 Réalisations ML/Data Science

| Projet | Type | Métriques | Démo |
|--------|------|-----------|------|
| 🔋 **Schneider Energy** | Régression | R²=0.83, +45% vs baseline | ✅ Active |
| 👥 **BackMarket** | Clustering | 95K clients, 4 segments | ✅ Active |
| 🏷️ **StackOverflow** | NLP | Precision@5=78%, F1=0.69 | ✅ Active |
| 🐕 **SantéVet** | Computer Vision | Top-1=87%, 120 races | ✅ Active |
| 📈 **Alla2 Trading** | Time Series | Accuracy=61%, earn_metric=1.10 | ✅ Active |

**5 démos interactives disponibles** avec prédictions en temps réel ou mode simulation.

## 🛠️ Stack Technique

- **ML/DL**: Scikit-learn, XGBoost, TensorFlow, PyTorch
- **NLP**: Transformers, BERT, TF-IDF, USE
- **CV**: ResNet50V2, EfficientNet, Transfer Learning
- **Data**: Pandas, NumPy, SQL
- **Viz**: Plotly, Matplotlib, Streamlit
- **Deploy**: Hugging Face Spaces, Docker, FastAPI

## 📁 Structure

```
portfolio/
├── app.py                    # Point d'entrée principal
├── pages/
│   ├── 1_🚀_Projet_Actuel.py # egir.app
│   ├── 2_💻_Réalisations.py  # Vue d'ensemble projets
│   ├── 3_👤_About.py         # Parcours & compétences
│   ├── 4_📧_Contact.py       # Formulaire contact
│   ├── 5_🔋_Schneider_Energy.py
│   ├── 6_👥_BackMarket_Segmentation.py
│   ├── 7_🏷️_StackOverflow_NLP.py
│   ├── 8_🐕_SanteVet_Dogs.py
│   └── 9_📈_Alla2_Trading.py
├── models/                   # Modèles ML légers
│   ├── p3_schneider/        # Random Forest (~1.6MB)
│   └── p4_backmarket/       # KMeans (~400KB)
└── requirements.txt
```

## 🚀 Lancer localement

```bash
git clone https://github.com/ThomasMeb/portfolio.git
cd portfolio
pip install -r requirements.txt
streamlit run app.py
```

## 📧 Contact

- LinkedIn: [Thomas Mebarki](https://linkedin.com/in/thomasmebarki)
- GitHub: [@ThomasMeb](https://github.com/ThomasMeb)
- Email: thomas.mebarki@protonmail.com

---

*Built with Streamlit | Deployed on Hugging Face Spaces*
