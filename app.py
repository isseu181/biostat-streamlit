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
    "Prédictions pour nouveaux patients",
    "Courbes ROC et recommandations"
])

# Section : Chargement de la base de données
if menu == "Chargement des données":
    st.header("Chargement de la base de données")
    try:
        # Charger directement la base intégrée
        data = pd.read_excel("Donnnées_Projet_M2SID2023_2024.xlsx", header=0)
        st.success("Base de données chargée avec succès !")
        st.write("Aperçu des données :", data.head(10))
        st.write(f"Dimensions des données : {data.shape}")

        # Stocker les données dans la session
        st.session_state['data'] = data
    except Exception as e:
        st.error(f"Erreur lors du chargement de la base de données : {e}")

# Section : Analyse descriptive
if menu == "Analyse descriptive":
    st.header("Analyse descriptive")
    if 'data' in st.session_state:
        data = st.session_state['data']

        # Colonnes binaires et traitement spécial
        colonnes_binaires = [
            'SEXE', 'Hypertension Arterielle', 'Diabete', 'Cardiopathie',
            'hémiplégie', 'Paralysie faciale', 'Aphasie', 'Hémiparésie',
            'Engagement Cerebral', 'Inondation Ventriculaire', 'Evolution'
        ]
        colonne_traitement = 'Traitement'

        # Remplacer les valeurs textuelles par des valeurs numériques
        data = data.replace({
            'OUI': 1, 'NON': 0, 'Homme': 1, 'Femme': 0,
            'Deces': 1, 'Vivant': 0, 'Thrombolyse': 1, 'Chirurgie': 2
        })

        # Forcer les colonnes binaires à être 0 ou 1
        for col in colonnes_binaires:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: 1 if x == 1 else 0).fillna(0)

        # Forcer "Traitement" à être 1 ou 2
        if colonne_traitement in data.columns:
            data[colonne_traitement] = data[colonne_traitement].apply(
                lambda x: 1 if x == 1 else 2
            ).fillna(1)

        st.session_state['data'] = data

        # Résumé statistique
        st.write("Résumé statistique des données :")
        st.write(data.describe(include='all'))

        # Visualisation : Répartition des traitements par évolution
        st.subheader("Répartition des traitements par évolution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Traitement', hue='Evolution', data=data, ax=ax1)
        ax1.set_title("Répartition des traitements par évolution")
        ax1.set_xlabel("Type de traitement (1 = Thrombolyse, 2 = Chirurgie)")
        ax1.set_ylabel("Nombre de patients")
        ax1.legend(title="Évolution", labels=["Vivant (0)", "Décès (1)"])
        st.pyplot(fig1)

        # Visualisation : Distribution des âges
        st.subheader("Distribution des âges")
        fig2, ax2 = plt.subplots()
        sns.histplot(data['AGE'], kde=True, ax=ax2)
        ax2.set_title("Distribution des âges")
        ax2.set_xlabel("Âge")
        st.pyplot(fig2)
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
        log_model = LogisticRegression(max_iter=1000)
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

# Section : Prédictions pour nouveaux patients
if menu == "Prédictions pour nouveaux patients":
    st.header("Prédictions pour nouveaux patients")
    if 'rf_model' in st.session_state and 'X_test' in st.session_state:
        rf_model = st.session_state['rf_model']
        feature_names = st.session_state['X_test'].columns

        # Interface utilisateur pour entrer les caractéristiques du patient
        st.subheader("Entrez les caractéristiques du patient :")
        input_data = {}

        # Colonnes avec des choix fixes
        choix_binaires = {
            'SEXE': [0, 1],  # 0 = Femme, 1 = Homme
            'Hypertension Arterielle': [0, 1],
            'Diabete': [0, 1],
            'Cardiopathie': [0, 1],
            'hémiplégie': [0, 1],
            'Paralysie faciale': [0, 1],
            'Aphasie': [0, 1],
            'Hémiparésie': [0, 1],
            'Engagement Cerebral': [0, 1],
            'Inondation Ventriculaire': [0, 1],
            'Traitement': [1, 2]  # 1 = Thrombolyse, 2 = Chirurgie
        }

        # Générer l'interface utilisateur
        for feature in feature_names:
            if feature in choix_binaires:
                input_data[feature] = st.selectbox(f"{feature} (Choisissez une option)", choix_binaires[feature])
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)

        input_df = pd.DataFrame([input_data])

        # Prédiction
        prediction = rf_model.predict(input_df)
        probas = rf_model.predict_proba(input_df)[:, 1]
        st.write(f"Prédiction : {'Décès (1)' if prediction[0] == 1 else 'Vivant (0)'}")
        st.write(f"Probabilité de décès : {probas[0]:.2f}")
    else:
        st.warning("Veuillez entraîner le modèle avant d'effectuer des prédictions.")

# Section : Courbes ROC 
if menu == "Courbes ROC ":
    st.header("Courbes ROC")
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
