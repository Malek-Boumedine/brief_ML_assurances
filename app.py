import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration de la page
st.set_page_config(
    page_title="AssuranceML",
    page_icon="logo.png",
    layout="wide"
)

# Charger les images si elles existent
try:
    st.sidebar.image(Image.open("datalock_logo.png"), width=280, use_container_width=True)
except FileNotFoundError:
    st.sidebar.error("Logo introuvable.")

# Header principal
st.title("\U0001F4B8 Estimation des Primes d'Assurance")
st.subheader("Un outil conçu pour Assur'aimant afin de simplifier et accélérer l'estimation des primes d'assurance.")

# Introduction et explications
st.markdown(
    """
    Bienvenue sur l'application dédiée à l'estimation des primes d'assurance. Cet outil a pour objectif de :
    
    - ✅ Simplifier le calcul des primes grâce à un modèle prédictif basé sur des données démographiques.
    - ✅ Fournir une analyse rapide et précise pour les courtiers d'assurance.
    - ✅ Aider Assur'aimant à mieux comprendre les profils clients grâce à des analyses statistiques détaillées.
    """
)

# Charger le modèle
@st.cache_resource
def charger_modele() -> object:
    """
    Charge le modèle préentraîné à partir d'un fichier.

    Ce bloc tente de charger le pipeline complet enregistré dans un fichier `.pkl`. 
    Si le fichier est introuvable, un message d'erreur est affiché à l'utilisateur, et l'exécution de l'application est arrêtée.
    """
    try:
        return joblib.load("complete_pipeline.pkl")
    except FileNotFoundError:
        st.error("Le fichier du modèle est introuvable. Assurez-vous qu'il est dans le même répertoire que l'application Streamlit.")
        st.stop()


# application 
def appli():
    """
    Affiche l'interface utilisateur pour le calcul du prix d'une assurance.
    """
    model = charger_modele()

    # Formulaire d'entrée
    with st.sidebar:
        st.sidebar.header("\U0001F4DD Saisissez les informations client")

        # Initialisation des valeurs dans st.session_state
        if "age_input" not in st.session_state:
            st.session_state.age_input = 0
        if "poids_input" not in st.session_state:
            st.session_state.poids_input = 0.0
        if "taille_input" not in st.session_state:
            st.session_state.taille_input = 0.0
        if "enfants_input" not in st.session_state:
            st.session_state.enfants_input = 0
        if "sex_radio" not in st.session_state:
            st.session_state.sex_radio = "male"
        if "smoker_radio" not in st.session_state:
            st.session_state.smoker_radio = "no"
        if "region_radio" not in st.session_state:
            st.session_state.region_radio = "southwest"

        # Form inputs
        # Widgets pour les entrées utilisateur
        st.sidebar.header("\U0001F4DD Saisissez les informations client")
        age = st.sidebar.number_input("Âge", min_value=0, max_value=120, value=30, step=1)
        sex = st.sidebar.radio("Sexe", ("Homme", "Femme"))
        poids = st.sidebar.number_input("Poids (en kg)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
        taille = st.sidebar.number_input("Taille (en cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
        bmi = round(poids / ((taille / 100) ** 2), 2) if taille > 0 else 0  # Calcul automatique du BMI
        children = st.sidebar.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0, step=1)
        smoker = st.sidebar.radio("Fumeur", ("Non", "Oui"))
        region = st.sidebar.selectbox("Région", ("Sud-Ouest", "Sud-Est", "Nord-Ouest", "Nord-Est"))

        # Calcul automatique du BMI
        bmi = round(poids / ((taille / 100) ** 2), 2) if taille > 0 else 0

        def reset_form():
            st.age_input = 0
            st.poids_input = 0.0
            st.taille_input = 0.0
            st.enfants_input = 0
            st.sex_radio = "male"
            st.smoker_radio = "no"
            st.region_radio = "southwest"

    # Affichage des informations
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

    # Boutons pour calculer ou réinitialiser
    col1, col2 = st.columns(2)
    if col1.button("Calculer"):
        if age == 0 or poids <= 0 or taille <= 0 or children < 0:
            st.error("Veuillez remplir tous les champs avec des valeurs valides.")
        else:
            try:
                # Créer le dataframe pour la prédiction
                valeurs = {
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                    "region": region
                }
                df_valeurs = pd.DataFrame([valeurs])

                # Prédiction
                predicted_premium = model.predict(df_valeurs)[0]
                st.success(f"Le prix d'assurance estimé est de : **{predicted_premium:.2f} $**")
            except Exception as e:
                st.error(f"Une erreur s'est produite lors du calcul : {e}")
    if st.button("Réinitialiser"):
        reset_form()

with st.container():
    st.markdown(
        """
        **📝 Note :**
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

# Lancer l'application
appli()
