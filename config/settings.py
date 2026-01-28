"""
Configuration du portfolio
"""

# Informations personnelles (à personnaliser)
PROFILE = {
    "name": "Thomas",
    "title": "ML Engineer & Entrepreneur",
    "email": "contact@example.com",
    "linkedin": "https://linkedin.com/in/",
    "github": "https://github.com/",
    "website": "https://egir.app",
}

# Projets ML
PROJECTS = {
    "p3_schneider": {
        "title": "Prédiction Énergétique",
        "subtitle": "Schneider Electric",
        "type": "Régression",
        "description": "Prédiction de consommation énergétique avec XGBoost",
        "path": "../P3-schneider-energy-prediction",
        "github": "",
        "enabled": False,  # Activer après intégration
    },
    "p4_backmarket": {
        "title": "Segmentation Client",
        "subtitle": "Back Market",
        "type": "Clustering",
        "description": "Segmentation RFM avec KMeans",
        "path": "../P4-backmarket-segmentation",
        "github": "",
        "enabled": False,
    },
    "p5_stackoverflow": {
        "title": "NLP Tag Suggestion",
        "subtitle": "Stack Overflow",
        "type": "NLP",
        "description": "Classification multi-label avec BERT/USE",
        "path": "../P5-stackoverflow-nlp-tags",
        "github": "",
        "enabled": False,
    },
    "p6_santevet": {
        "title": "Classification Races",
        "subtitle": "SantéVet",
        "type": "Computer Vision",
        "description": "Classification d'images avec ResNet50V2",
        "path": "../P6-santevet-dog-classification",
        "github": "",
        "enabled": False,
    },
    "alla2": {
        "title": "Prédiction Trading",
        "subtitle": "Projet Personnel",
        "type": "Time Series",
        "description": "Prédiction de séries temporelles financières",
        "path": "../Alla2",
        "github": "",
        "enabled": False,
    },
}

# Projet actuel
CURRENT_PROJECT = {
    "name": "egir.app",
    "url": "https://egir.app",
    "description": "Plateforme SaaS de gestion pour restaurateurs avec IA intégrée",
    "highlights": [
        "+10% de marge en moyenne",
        "80% de temps économisé",
        "ROI estimé 19-33x par an",
    ],
}
