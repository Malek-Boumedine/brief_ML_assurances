from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importation des données
data = pd.read_csv("dataset_assurance.csv")
data = data.drop_duplicates()

X = data.drop("charges", axis=1)
y = data["charges"]


# fonction pour afficher les metriques
def metriques(model, X_test, y_test) :
    """
    Calcule et affiche les métriques de régression.

    Cette fonction calcule plusieurs métriques d'évaluation pour évaluer la performance d'un modèle de régression : 
    - MSE (Mean Squared Error) : Erreur quadratique moyenne.
    - RMSE (Root Mean Squared Error) : Racine carrée de l'erreur quadratique moyenne.
    - MAE (Mean Absolute Error) : Erreur absolue moyenne.
    - R² (Coefficient de détermination) : Mesure la proportion de variance expliquée par le modèle.
    - MedAE (Median Absolute Error) : Médiane des erreurs absolues.

    Args:
        model: Le modèle de régression déjà entraîné.
        X_test: Les caractéristiques des données de test.
        y_test: La variable cible correspondante pour les données de test.

    Returns:
        tuple: Contient les valeurs des métriques calculées (MSE, MAE, R², MedAE).
    """

    pred_y = model.predict(X_test)
    mse = mean_squared_error(y_test, pred_y)
    mae = mean_absolute_error(y_test, pred_y)
    r2 = r2_score(y_test, pred_y)
    med_ae = median_absolute_error(y_test, pred_y)

    print(f"MSE : {mse}")
    print(f"RMSE : {np.sqrt(mse)}")
    print(f"MAE : {mae}")
    print(f"R2 : {r2}")
    print(f"MedAE : {med_ae}")

    return mse, mae, r2, med_ae

# fonction pour transformer les catégories d"age
def categorie_age(age) :
    """
    Catégorise les âges en groupes.

    Divise l'âge en catégories comme : "24 et moins", "25-34", etc., selon des seuils pré-définis.

    Args:
        age (int ou float): L'âge à catégoriser.

    Returns:
        str: La catégorie d'âge correspondante.
    """

    if age < 25:
        return "24 et moins"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    else:
        return "55 et plus"

# fonction pour transformer les catégories de bmi
def categorie_bmi(bmi):
    """
    Catégorise l'indice de masse corporelle (IMC), Body Mass Index (BMI) en anglais.
    Cette fonction prend un IMC en entrée et retourne une chaîne de caractères représentant la catégorie d'IMC correspondante.
    Le BMI est catégorisé en insuffisance pondérale, normal, surpoids, et différents niveaux d'obésité.

    Args:
        bmi (float): L'indice de masse corporelle à catégoriser.

    Returns:
        str: La catégorie de BMI correspondante.
    """

    if bmi < 18.5 :
        return "insuffisance pondérale"
    elif bmi < 24.5:
        return "normal"
    elif bmi < 30:
        return "surpoids"
    elif bmi < 35:
        return "obésité I"
    elif bmi < 40:
        return "obésité II"
    else:
        return "obésité III"


# preprocesseur personnalisé
class Preprocessor_personnalise(BaseEstimator, TransformerMixin):
    """
    Préprocesseur personnalisé pour les données d'assurance.

    Ce préprocesseur effectue les transformations suivantes :
    1. Encodage binaire : 
    - `smoker` (oui/non)
    - `sex` (homme/femme)
    2. Encodage one-hot : 
    - `region` (sud-ouest, nord-est, etc.)
    - Catégories `bmi` (IMC) et `age` (groupes d'âges)
    3. Renommage de la colonne `sex` en `sex_male`.

    Convient pour un pipeline dans scikit-learn.

    Attributes:
        encodeur_smoker (LabelBinarizer): Encodeur binaire pour la variable `smoker`.
        encodeur_sex (LabelBinarizer): Encodeur binaire pour la variable `sex`.
        encodeur_region (OneHotEncoder): Encodeur one-hot pour la variable `region`.
        encodeur_bmi_cat (OneHotEncoder): Encodeur one-hot pour les catégories de BMI.
        encodeur_age_cat (OneHotEncoder): Encodeur one-hot pour les catégories d'âge.
        region_columns (list): Noms des colonnes générées après l'encodage de `region`.
        bmi_cat_columns (list): Noms des colonnes générées après l'encodage de `bmi_cat`.
        age_cat_columns (list): Noms des colonnes générées après l'encodage de `age_cat`.
    """

    def __init__(self):
        """
        Initialise le préprocesseur avec les encodeurs nécessaires.

        Les encodeurs pour les variables catégorielles (`region`, `bmi_cat`, `age_cat`) 
        et binaires (`smoker`, `sex`) sont configurés.
        """

        self.encodeur_smoker = LabelBinarizer(pos_label=1, neg_label=0)
        self.encodeur_sex = LabelBinarizer(pos_label=1, neg_label=0)
        self.encodeur_region = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Ajout de handle_unknown
        self.encodeur_bmi_cat = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') # Ajout pour bmi_cat
        self.encodeur_age_cat = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') # Ajout pour age_cat

        # classes des encodeurs
        self.encodeur_smoker.classes_ = ["yes", "non"]
        self.encodeur_sex.classes_ = np.array(["male", "female"])

        self.region_columns = None
        self.bmi_cat_columns = None
        self.age_cat_columns = None

    def fit(self, X, y=None):
        """
        Adapte les encodeurs du préprocesseur aux données.

        Identifie les catégories uniques dans les variables catégorielles pour préparer l'encodage.

        Args:
            X (pd.DataFrame): Données d'entraînement (caractéristiques).
            y (array, optionnel): Données cibles (non utilisées ici).

        Returns:
            Preprocessor_personnalise: L'instance du préprocesseur ajustée.
        """

        # encodage des X cats
        self.encodeur_smoker.fit(X["smoker"])
        self.encodeur_sex.fit(X["sex"])
        self.encodeur_region.fit(X[["region"]])
        self.region_columns = self.encodeur_region.get_feature_names_out(["region"])

        # Création des catégories avant le fit des encodeurs one-hot
        X_copie = X.copy()
        X_copie["bmi_cat"] = X_copie["bmi"].apply(categorie_bmi)
        X_copie["age_cat"] = X_copie["age"].apply(categorie_age) 

        self.encodeur_bmi_cat.fit(X_copie[["bmi_cat"]])
        self.bmi_cat_columns = self.encodeur_bmi_cat.get_feature_names_out(["bmi_cat"])

        self.encodeur_age_cat.fit(X_copie[["age_cat"]])
        self.age_cat_columns = self.encodeur_age_cat.get_feature_names_out(["age_cat"])

        return self

    def transform(self, X):
        """
        Applique les transformations définies aux données.

        Effectue l'encodage binaire et one-hot sur les colonnes pertinentes et 
        ajoute les nouvelles colonnes aux données.

        Args:
            X (pd.DataFrame): Données à transformer.

        Returns:
            pd.DataFrame: Données transformées avec les colonnes encodées.
        """

        X_copie = X.copy()

        # encodage smoker et sex
        X_copie["smoker"] = self.encodeur_smoker.transform(X_copie["smoker"])
        X_copie["sex"] = self.encodeur_sex.transform(X_copie["sex"])
        X_copie.rename(columns={"sex": "sex_male"}, inplace=True)

        # encodage region
        region_encodee = self.encodeur_region.transform(X_copie[["region"]])
        df_region_encodee = pd.DataFrame(
            region_encodee,
            columns=self.region_columns,
            index=X_copie.index
        )

        # concat et drop
        X_copie = pd.concat([X_copie, df_region_encodee], axis=1)
        X_copie.drop("region", axis=1, inplace=True)

        # création categories bmi et age
        X_copie["bmi_cat"] = X_copie["bmi"].apply(categorie_bmi)
        X_copie["age_cat"] = X_copie["age"].apply(categorie_age) # Correction ici

        # onehot encodage bmi_cat
        bmi_cat_encodee = self.encodeur_bmi_cat.transform(X_copie[["bmi_cat"]])
        df_bmi_cat_encodee = pd.DataFrame(
            bmi_cat_encodee,
            columns=self.bmi_cat_columns,
            index=X_copie.index
        )
        X_copie = pd.concat([X_copie, df_bmi_cat_encodee], axis=1)
        X_copie.drop("bmi_cat", axis=1, inplace=True)

        # onehot encodage age_cat
        age_cat_encodee = self.encodeur_age_cat.transform(X_copie[["age_cat"]])
        df_age_cat_encodee = pd.DataFrame(
            age_cat_encodee,
            columns=self.age_cat_columns,
            index=X_copie.index
        )
        X_copie = pd.concat([X_copie, df_age_cat_encodee], axis=1)
        X_copie.drop("age_cat", axis=1, inplace=True)

        return X_copie
    
    
# pipeline 

"""
Pipeline complet pour le traitement des données d'assurance.

Ce pipeline effectue :
1. Prétraitement personnalisé avec `Preprocessor_personnalise`.
2. Création de nouvelles caractéristiques polynomiales avec `PolynomialFeatures`.
3. Mise à l'échelle des données avec `StandardScaler`.
4. Sélection des 50 meilleures caractéristiques avec `SelectKBest`.
5. Régression avec Lasso pour ajuster les données au modèle.

"""
pipeline_complete = Pipeline([
    ("preprocessor", Preprocessor_personnalise()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_regression, k=50)),
    ("regressor", Lasso(alpha=238, random_state=42))
])

# split trainset testset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.85, random_state=42, stratify=X["smoker"])

# entrainement
pipeline_complete.fit(X_train, y_train)


# sauvegarde pipeline
"""
Sauvegarde le pipeline entraîné dans un fichier .pkl pour un usage ultérieur.
"""
import joblib
joblib.dump(pipeline_complete, "complete_pipeline.pkl")


##################################################################

if __name__ == "__main__" :    
    # evaluation
    metriques(model=pipeline_complete, X_test=X_test, y_test=y_test)
    valeurs1 = {"age" : 34, "sex" : "male", "bmi" : 24.5, "children" : 0, "smoker" : "no", "region" : "southwest"}
    print("#"*20)
    df1 = pd.DataFrame([valeurs1])
    print(pipeline_complete.predict(df1))
    
    features_preprocess = pipeline_complete.named_steps["preprocessor"].transform(X_train).columns
    
    # récupérer features après transformation poly
    poly_features = pipeline_complete.named_steps["poly"].get_feature_names_out(features_preprocess)
    # poly_features

    # récupérer masque des features sélectionnées
    feature_mask = pipeline_complete.named_steps["feature_selection"].get_support()
    # feature_mask

    # récupérer les noms des features sélectionnées
    selected_features = poly_features[feature_mask]
    # selected_features

    # récupérer les coefficients
    coefficients = pipeline_complete.named_steps["regressor"].coef_
    # coefficients

    # df avec les features et leurs coefficients
    pipeline_coef_df = pd.DataFrame({
        "feature": selected_features,
        "coefficient": coefficients
    }).sort_values("coefficient", key=abs, ascending=True)  # trier selon la valeur absolue 
    # pipeline_coef_df

    plt.figure(figsize=(20, 12))
    plt.barh(pipeline_coef_df["feature"], pipeline_coef_df["coefficient"])
    plt.title("coefficients des variables sélectionnées dans la pipeline")
    plt.xlabel("Coefficient")
    plt.grid()
    plt.tight_layout()
    plt.show()

    
    
    
    
    
    
    
    
    
    
