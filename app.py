# Configuration et importation des bibliothèques
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

# Titre de l'application
st.title("Étude pronostique des décès après traitement")

# Navigation dans l'application
menu = st.sidebar.radio("Étapes", [
    "Chargement des données",
    "Analyse descriptive",
    "Préparation des données",
    "Modélisation et comparaison",
    "Courbes ROC et déploiement"
])

# Section : Chargement des données
if menu == "Chargement des données":
    st.header("Chargement des données")
    uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
    if uploaded_file:
        try:
            # Charger les données
            data = pd.read_excel(uploaded_file, header=0)
            st.session_state['data'] = data
            st.success("Données chargées avec succès !")
            st.write("Aperçu des données :", data.head())
            st.write(f"Dimensions des données : {data.shape}")
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
    else:
        st.warning("Veuillez téléverser un fichier Excel pour continuer.")

# Section : Analyse descriptive
if menu == "Analyse descriptive":
    st.header("Analyse descriptive")
    if 'data' in st.session_state:
        data = st.session_state['data']

        # Conversion des données textuelles en numériques
        data = data.replace({
            'OUI': 1, 'NON': 0, 'Homme': 1, 'Femme': 0,
            'Deces': 1, 'Vivant': 0, 'Thrombolyse': 1, 'Chirurgie': 2
        })
        data = data.fillna(0)  # Remplacer les valeurs manquantes par 0
        st.session_state['data'] = data

        # Résumé statistique
        st.write("Résumé statistique des données :")
        st.write(data.describe(include='all'))

        # Visualisation : Relation entre Traitement et Décès
        st.subheader("Analyse de la relation entre Traitement et Décès")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Traitement', hue='Evolution', data=data, ax=ax1)
        ax1.set_title("Évolution par type de traitement")
        ax1.legend(title='Évolution', labels=['Vivant (0)', 'Décès (1)'])
        st.pyplot(fig1)

        # Visualisation : Distribution des âges
        st.subheader("Distribution des âges")
        fig2, ax2 = plt.subplots()
        sns.histplot(data['AGE'], kde=True, ax=ax2)
        ax2.set_title("Distribution des âges")
        ax2.set_xlabel("Âge")
        st.pyplot(fig2)

        # Corrélations
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.write("Matrice de corrélation :")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)
        else:
            st.warning("Aucune colonne numérique disponible pour calculer la corrélation.")
    else:
        st.warning("Veuillez charger les données dans la section précédente.")

# Section : Préparation des données
if menu == "Préparation des données":
    st.header("Préparation des données pour la modélisation")
    if 'data' in st.session_state:
        data = st.session_state['data']
        target = st.selectbox("Sélectionnez la variable cible (y)", data.columns)
        features = st.multiselect("Sélectionnez les variables explicatives (X)", data.columns)

        if target and features:
            # Préparer X et y
            X = data[features]
            y = data[target]

            # Convertir en numérique et remplir les valeurs manquantes
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            y = y.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Afficher les tailles des ensembles
            st.write(f"Taille de l'ensemble d'entraînement : {X_train.shape}")
            st.write(f"Taille de l'ensemble de test : {X_test.shape}")

            # Stocker les données
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test

            st.success("Données préparées avec succès.")
            st.write("Aperçu des données d'entraînement :", X_train.head())
        else:
            st.warning("Veuillez sélectionner la cible et les variables explicatives.")
    else:
        st.warning("Veuillez charger les données dans la section précédente.")

# Section : Modélisation et comparaison
if menu == "Modélisation et comparaison":
    st.header("Modélisation et comparaison des modèles")
    if 'X_train' in st.session_state and 'y_train' in st.session_state:
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']

        # Modèle de régression logistique
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        st.session_state['log_model'] = log_model

        # Modèle de forêt aléatoire
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        st.session_state['rf_model'] = rf_model

        # Afficher les résultats
        st.subheader("Régression Logistique")
        st.text(classification_report(y_test, y_pred_log))

        st.subheader("Forêt Aléatoire")
        st.text(classification_report(y_test, y_pred_rf))
    else:
        st.warning("Veuillez préparer les données avant de continuer.")

# Section : Courbes ROC et déploiement
if menu == "Courbes ROC et déploiement":
    st.header("Courbes ROC et comparaison finale")
    if 'X_test' in st.session_state and 'y_test' in st.session_state:
        if 'log_model' in st.session_state and 'rf_model' in st.session_state:
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            log_model = st.session_state['log_model']
            rf_model = st.session_state['rf_model']

            # Courbe ROC - Régression Logistique
            y_prob_log = log_model.predict_proba(X_test)[:, 1]
            fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
            roc_auc_log = auc(fpr_log, tpr_log)

            # Courbe ROC - Forêt Aléatoire
            y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
            roc_auc_rf = auc(fpr_rf, tpr_rf)

            # Tracer les courbes ROC
            fig, ax = plt.subplots()
            ax.plot(fpr_log, tpr_log, label=f"Régression Logistique (AUC = {roc_auc_log:.2f})", color="blue")
            ax.plot(fpr_rf, tpr_rf, label=f"Forêt Aléatoire (AUC = {roc_auc_rf:.2f})", color="green")
            ax.plot([0, 1], [0, 1], 'k--', label="Modèle aléatoire (AUC = 0.50)", color="red")
            ax.set_xlabel("Taux de faux positifs (FPR)")
            ax.set_ylabel("Taux de vrais positifs (TPR)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Veuillez entraîner les modèles avant de continuer.")
    else:
        st.warning("Veuillez préparer les données avant de continuer.")
