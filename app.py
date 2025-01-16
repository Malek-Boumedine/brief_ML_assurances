import streamlit as st
import pandas as pd
import joblib
import main



# Charger le modèle
def charger_modele() -> object : 
    """
    Charge le modèle préentraîné à partir d'un fichier.

    Ce bloc tente de charger le pipeline complet enregistré dans un fichier `.pkl`. 
    Si le fichier est introuvable, un message d'erreur est affiché à l'utilisateur, et l'exécution de l'application est arrêtée.

    Exceptions :
        FileNotFoundError: Levée si le fichier `complete_pipeline.pkl` est absent.
    """
    try:
        return joblib.load("complete_pipeline.pkl")
    except FileNotFoundError:
        st.error("Le fichier du modèle est introuvable. Assurez-vous qu'il est dans le même répertoire que l'application Streamlit.")
        st.stop()


def reset_form() -> None :
    """
    Réinitialise le formulaire Streamlit à ses valeurs par défaut.

    Cette fonction rétablit les champs de saisie dans l'état de session Streamlit (`st.session_state`) 
    avec leurs valeurs par défaut. Cela permet de vider ou de réinitialiser le formulaire pour une nouvelle saisie.

    Effets :
        - Réinitialise les champs : âge, poids, taille, nombre d'enfants, sexe, statut fumeur, et région.
        - Les valeurs par défaut utilisées sont celles définies lors de l'initialisation de l'état.

    Champs réinitialisés :
        - age_input : 0
        - poids_input : 0.0
        - taille_input : 0.0
        - enfants_input : 0
        - sex_radio : "male"
        - smoker_radio : "no"
        - region_radio : "southwest"
    """
    st.session_state.age_input = 0
    st.session_state.poids_input = 0.0
    st.session_state.taille_input = 0.0
    st.session_state.enfants_input = 0
    st.session_state.sex_radio = "male"
    st.session_state.smoker_radio = "no"
    st.session_state.region_radio = "southwest"
    

# application 
def appli() -> None :
    """
    Affiche l'interface utilisateur pour le calcul du prix d'une assurance.

    Cette section utilise Streamlit pour construire un formulaire interactif permettant à l'utilisateur de saisir 
    les informations nécessaires au calcul du prix d'assurance, notamment : âge, sexe, poids, taille, nombre d'enfants, 
    statut fumeur et région.

    Fonctionnalités :
        - Entrée des données utilisateur via des champs interactifs (`number_input`, `radio`).
        - Calcul automatique de l'IMC (indice de masse corporelle) basé sur le poids et la taille.
        - Prédiction du prix d'assurance à l'aide d'un modèle chargé.
        - Gestion des erreurs (champs invalides ou incomplets).
        - Réinitialisation des champs avec un bouton dédié.

    Formulaire :
        - Champs requis : âge, sexe, poids, taille, nombre d'enfants, statut fumeur, région.
        - Boutons d'action : "Calculer" pour effectuer la prédiction, "Réinitialiser" pour vider les champs.

    Validation et messages :
        - Affiche un message d'erreur si des champs sont manquants ou invalides.
        - Affiche un message de succès avec le prix d'assurance prédit.

    Exceptions :
        - Gère et affiche les erreurs éventuelles rencontrées lors du calcul ou de la prédiction.

    """
    model = charger_modele()
    st.title("Calcul du prix d'une assurance")

    with st.form("nouveau_form"):
        valeurs = {"age": None, "sex": None, "bmi": None, "children": None, "smoker": None, "region": None}
        
        valeurs["age"] = st.number_input("Age", min_value=0, step=1, key="age_input")
        valeurs["sex"] = st.radio("Sexe", ["male", "female"], key="sex_radio")
        poids = st.number_input("Poids en kg", min_value=0.0, step=1.0, key="poids_input")
        taille = st.number_input("Taille en cm", min_value=0.0, step=1.0, key="taille_input")
        valeurs["children"] = st.number_input("Nombre d'enfants", min_value=0, step=1, key="enfants_input") 
        valeurs["smoker"] = st.radio("Fumeur", ["no", "yes"], key="smoker_radio") 
        valeurs["region"] = st.radio("Region", ["southwest", "southeast", "northwest", "northeast"], key="region_radio") 

        col1, col2 = st.columns(2)
        with col1:
            calculer = st.form_submit_button("Calculer")
        with col2:
            reinitialiser = st.form_submit_button("Réinitialiser", on_click=reset_form)
            
        if calculer:
            if not all(valeurs[cle] is not None for cle in ["age", "sex", "children", "smoker", "region"]) or poids <= 0 or taille <= 0:
                st.error("Veuillez remplir tous les champs avec des valeurs valides.")
            else:
                try:
                    valeurs["bmi"] = poids / ((taille / 100) ** 2)
                    df_valeurs = pd.DataFrame([valeurs])
                    prix_assu_pred = model.predict(df_valeurs)[0]
                    st.success(f"Le prix d'assurance estimé est de : {prix_assu_pred:.2f} $")
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors du calcul : {e}")
        elif reinitialiser :
            st.rerun()

# lancement de l'appli avec streamlit run app.py
appli()