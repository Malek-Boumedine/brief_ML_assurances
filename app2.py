import streamlit as st
from PIL import Image
import random
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Estimation des Primes d'Assurance",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Formulaire dans la sidebar
st.sidebar.header("\U0001F4DD Saisissez les informations client")
age = st.sidebar.number_input("Âge", min_value=0, max_value=120, value=30, step=1)
sex = st.sidebar.radio("Sexe", ("Homme", "Femme"))
poids = st.sidebar.number_input("Poids (en kg)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
taille = st.sidebar.number_input("Taille (en cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
bmi = round(poids / ((taille / 100) ** 2), 2) if taille > 0 else 0  # Calcul automatique du BMI
children = st.sidebar.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0, step=1)
smoker = st.sidebar.radio("Fumeur", ("Non", "Oui"))
region = st.sidebar.selectbox("Région", ("Sud-Ouest", "Sud-Est", "Nord-Ouest", "Nord-Est"))

# Conteneur : Présentation
with st.container():
    st.title("\U0001F4B8 Estimation des Primes d'Assurance")
    st.subheader("Un outil conçu pour Assur'aimant afin de simplifier et accélérer l'estimation des primes d'assurance.")
    st.markdown(
        """
        Bienvenue sur l'application dédiée à l'estimation des primes d'assurance. Cet outil a pour objectif de :
        
        - \U00002705 Simplifier le calcul des primes grâce à un modèle prédictif basé sur des données démographiques.
        - \U00002705 Fournir une analyse rapide et précise pour les courtiers d'assurance.
        - \U00002705 Aider Assur'aimant à mieux comprendre les profils clients grâce à des analyses statistiques détaillées.

        **Instructions** :
        1. Saisissez les informations du client dans les champs ci-contre.
        2. Obtenez une estimation immédiate de la prime d'assurance.
        """
    )

# Conteneur : Informations client
with st.container():
    st.markdown("### \U0001F4C3 Informations saisies :")
    data = {
        "Âge": [age],
        "Sexe": [sex],
        "Poids (kg)": [poids],
        "Taille (cm)": [taille],
        "BMI": [bmi],
        "Nombre d'enfants": [children],
        "Fumeur": [smoker],
        "Région": [region]
    }
    df = pd.DataFrame(data)
    st.table(df)

# Simulation de la prédiction (dummy model pour exemple)
predicted_premium = random.uniform(2000, 15000)  # Exemple de prédiction aléatoire

# Conteneur : Résultat de l'estimation
with st.container():
    st.markdown("### \U0001F4B3 Résultat de l'estimation")
    st.success(f"\U0001F4AA La prime d'assurance estimée est de **{predicted_premium:.2f} USD**.")

# Conteneur : Notes et disclaimers
with st.container():
    st.markdown(
        """
        **:warning:Note :**
        - Cette estimation est générée par un modèle de machine learning basé sur les données disponibles.
        - Les résultats sont indicatifs et ne se substituent pas à une évaluation professionnelle.
        
        **\U0001F512 Confidentialité :**
        - Les données saisies ne sont pas stockées.
        - L'application est conforme aux réglementations sur la protection des données.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        Conçu par votre équipe IA, avec \U0001F49C pour Assur'aimant.
        
        [Contactez-nous](mailto:support@assuraimant.com) pour toute question ou amélioration.
        """
    )
