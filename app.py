# Configuration et importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# Titre de l'application
st.title("Analyse Biostatistique avec Streamlit")
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Étapes", [
    "Chargement des données",
    "Exploration et prétraitement",
    "Visualisations",
    "Préparation pour la modélisation",
    "Modélisation",
    "Courbes ROC"
])

# Section : Chargement des données
if menu == "Chargement des données":
    st.header("Chargement des données")
    uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
    if uploaded_file:
        try:
            # Lire les données Excel
            data = pd.read_excel(uploaded_file, header=1)
            st.session_state['data'] = data
            st.success("Données chargées avec succès !")
            st.write("Dimensions des données :", data.shape)
            st.write("Aperçu des données :")
            st.write(data.head())
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
    else:
        st.warning("Veuillez téléverser un fichier Excel pour continuer.")

# Section : Exploration et prétraitement des données
if menu == "Exploration et prétraitement":
    st.header("Exploration et prétraitement")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Résumé statistique :")
        st.write(data.describe())
        st.write("Valeurs manquantes :")
        st.write(data.isnull().sum())
    else:
        st.warning("Veuillez d'abord charger les données dans la section 'Chargement des données'.")

# Section : Visualisations
if menu == "Visualisations":
    st.header("Visualisations")
    if 'data' in st.session_state:
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Sélectionnez une colonne numérique pour l'histogramme", numeric_columns)
        if col:
            st.write(f"Distribution de {col} :")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Veuillez d'abord charger les données dans la section 'Chargement des données'.")

# Section : Préparation des données pour la modélisation
if menu == "Préparation pour la modélisation":
    st.header("Préparation des données pour la modélisation")
    if 'data' in st.session_state:
        data = st.session_state['data']
        target = st.selectbox("Sélectionnez la variable cible (y)", data.columns)
        features = st.multiselect("Sélectionnez les variables explicatives (X)", data.columns)
        if target and features:
            X = data[features]
            y = data[target]

            # Convertir les colonnes catégoriques en numériques
            X = pd.get_dummies(X, drop_first=True)
            if y.dtype == 'object':
                y = y.map({'OUI': 1, 'NON': 0})  # Adapter selon vos données

            # Remplacer les valeurs manquantes par 0
            X = X.fillna(0)
            y = y.fillna(0)

            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vérification des types et affichage des données
            st.write("Types des colonnes dans X_train :")
            st.write(X_train.dtypes)
            st.write("Aperçu des premières lignes de X_train :")
            st.write(X_train.head())
            st.write("Aperçu des premières valeurs de y_train :")
            st.write(y_train.head())

            # Stocker les données pour la modélisation
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.success("Données préparées pour la modélisation.")
        else:
            st.warning("Veuillez sélectionner la cible et les variables explicatives.")
    else:
        st.warning("Veuillez d'abord charger les données dans la section 'Chargement des données'.")

# Section : Modélisation
if menu == "Modélisation":
    st.header("Modélisation")
    if 'X_train' in st.session_state and 'y_train' in st.session_state:
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']

        try:
            # Régression logistique
            log_model = LogisticRegression()
            log_model.fit(X_train, y_train)
            y_pred_log = log_model.predict(X_test)
            st.session_state['log_model'] = log_model  # Stocker le modèle

            # Forêt aléatoire
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            st.session_state['rf_model'] = rf_model  # Stocker le modèle

            # Afficher les résultats
            st.subheader("Régression Logistique")
            st.write(classification_report(y_test, y_pred_log))

            st.subheader("Forêt Aléatoire")
            st.write(classification_report(y_test, y_pred_rf))
        except ValueError as e:
            st.error(f"Erreur lors de la modélisation : {e}")
    else:
        st.warning("Veuillez préparer les données dans la section 'Préparation pour la modélisation'.")

# Section : Courbes ROC
if menu == "Courbes ROC":
    st.header("Courbes ROC")
    if 'X_test' in st.session_state and 'y_test' in st.session_state:
        if 'log_model' in st.session_state and 'rf_model' in st.session_state:
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            log_model = st.session_state['log_model']
            rf_model = st.session_state['rf_model']
            
            # Régression Logistique ROC
            y_prob_log = log_model.predict_proba(X_test)[:, 1]
            fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
            roc_auc_log = auc(fpr_log, tpr_log)

            # Forêt Aléatoire ROC
            y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
            roc_auc_rf = auc(fpr_rf, tpr_rf)

            # Affichage des courbes ROC
            fig, ax = plt.subplots()
            ax.plot(fpr_log, tpr_log, label=f"Régression Logistique (AUC = {roc_auc_log:.2f})", color='blue')
            ax.plot(fpr_rf, tpr_rf, label=f"Forêt Aléatoire (AUC = {roc_auc_rf:.2f})", color='green')
            ax.plot([0, 1], [0, 1], 'k--', label="Modèle aléatoire (AUC = 0.50)", color='red')
            ax.set_xlabel("Taux de faux positifs (FPR)")
            ax.set_ylabel("Taux de vrais positifs (TPR)")
            ax.set_title("Courbes ROC")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Veuillez entraîner les modèles dans la section 'Modélisation'.")
    else:
        st.warning("Veuillez préparer les données dans la section 'Préparation pour la modélisation'.")
