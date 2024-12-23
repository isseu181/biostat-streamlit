# Configuration et importation des bibliothèques
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
    "Modélisation et validation croisée",
    "Prédictions finales",
    "Recommandations pratiques"
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

# Section : Modélisation et validation croisée
if menu == "Modélisation et validation croisée":
    st.header("Modélisation et validation croisée")
    if 'X_train' in st.session_state and 'y_train' in st.session_state:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']

        # Modèle de régression logistique
        log_model = LogisticRegression()
        scores_log = cross_val_score(log_model, X_train, y_train, cv=5, scoring='accuracy')
        st.write(f"Validation croisée - Régression Logistique : {scores_log.mean():.2f} ± {scores_log.std():.2f}")
        log_model.fit(X_train, y_train)

        # Modèle de forêt aléatoire
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
        st.write(f"Validation croisée - Forêt Aléatoire : {scores_rf.mean():.2f} ± {scores_rf.std():.2f}")
        rf_model.fit(X_train, y_train)

        # Stocker les modèles
        st.session_state['log_model'] = log_model
        st.session_state['rf_model'] = rf_model
    else:
        st.warning("Veuillez préparer les données avant de continuer.")

# Section : Prédictions finales
if menu == "Prédictions finales":
    st.header("Prédictions finales pour de nouveaux patients")
    if 'rf_model' in st.session_state:
        rf_model = st.session_state['rf_model']
        X_test = st.session_state['X_test']

        st.subheader("Entrez les caractéristiques du patient pour prédiction")
        input_data = {col: st.number_input(f"{col}", value=float(X_test[col].mean())) for col in X_test.columns}
        input_df = pd.DataFrame([input_data])

        # Prédiction
        prediction = rf_model.predict(input_df)
        probas = rf_model.predict_proba(input_df)[:, 1]
        st.write(f"Prédiction : {'Décès (1)' if prediction[0] == 1 else 'Vivant (0)'}")
        st.write(f"Probabilité de décès : {probas[0]:.2f}")
    else:
        st.warning("Veuillez entraîner les modèles avant de continuer.")

# Section : Recommandations pratiques
if menu == "Recommandations pratiques":
    st.header("Recommandations pratiques")
    st.write("""
    - Pour réduire le risque de décès, privilégiez les traitements ayant montré une efficacité élevée dans les modèles.
    - Analysez les caractéristiques des patients pour ajuster les traitements, comme :
        - Type de traitement (Thrombolyse ou Chirurgie).
        - Surveillance étroite pour les patients présentant des signes de gravité.
    - Intégrez les prédictions pour optimiser les décisions cliniques.
    """)
