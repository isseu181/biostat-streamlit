# Configuration et importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

# Titre de l'application
st.title("Analyse Biostatistique avec Streamlit")
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Étapes", [
    "Configuration et importation",
    "Chargement des données",
    "Exploration et prétraitement",
    "Visualisations",
    "Préparation pour la modélisation",
    "Modélisation",
    "Comparaison des modèles",
    "Courbes ROC"
])

# Chargement des données
if menu == "Chargement des données":
    st.header("Chargement des données")
    uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
    if uploaded_file:
        data = pd.read_excel(uploaded_file, header=1)
        st.write("Aperçu des données chargées :")
        st.write(data.head())
    else:
        st.warning("Veuillez téléverser un fichier Excel.")

# Exploration et prétraitement des données
if menu == "Exploration et prétraitement":
    st.header("Exploration et prétraitement")
    if 'data' in locals():
        st.write("Résumé statistique :")
        st.write(data.describe())
        st.write("Valeurs manquantes :")
        st.write(data.isnull().sum())
    else:
        st.warning("Veuillez d'abord charger les données.")

# Visualisations
if menu == "Visualisations":
    st.header("Visualisations")
    if 'data' in locals():
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Sélectionnez une colonne numérique pour l'histogramme", numeric_columns)
        if col:
            st.write(f"Distribution de {col} :")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Veuillez d'abord charger les données.")

# Préparation des données pour la modélisation
if menu == "Préparation pour la modélisation":
    st.header("Préparation des données pour la modélisation")
    if 'data' in locals():
        target = st.selectbox("Sélectionnez la variable cible (y)", data.columns)
        features = st.multiselect("Sélectionnez les variables explicatives (X)", data.columns)
        if target and features:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.success("Données préparées pour la modélisation.")
        else:
            st.warning("Veuillez sélectionner la cible et les variables explicatives.")
    else:
        st.warning("Veuillez d'abord charger les données.")

# Modélisation
if menu == "Modélisation":
    st.header("Modélisation")
    if 'X_train' in locals() and 'y_train' in locals():
        # Régression logistique
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)

        # Forêt aléatoire
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        # Afficher les résultats
        st.subheader("Régression Logistique")
        st.write(classification_report(y_test, y_pred_log))

        st.subheader("Forêt Aléatoire")
        st.write(classification_report(y_test, y_pred_rf))
    else:
        st.warning("Veuillez préparer les données pour la modélisation.")

# Comparaison des modèles et Courbes ROC
if menu == "Courbes ROC":
    st.header("Courbes ROC")
    if 'X_test' in locals() and 'y_test' in locals():
        # Régression Logistique ROC
        y_prob_log = log_model.predict_proba(X_test)[:, 1]
        fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
        roc_auc_log = auc(fpr_log, tpr_log)

        # Forêt Aléatoire ROC
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)

        # Plot
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
        st.warning("Veuillez entraîner les modèles pour afficher les courbes ROC.")
