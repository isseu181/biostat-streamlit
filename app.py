import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc

# Fonction pour charger les données
def charger_donnees():
    uploaded_file = st.file_uploader("Choisir un fichier Excel", type="xlsx")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("Données chargées avec succès!")
        return data
    else:
        return None

# Fonction pour explorer les données
def explorer_donnees(data):
    st.write("### Aperçu des données")
    st.write(data.head())
    
    st.write("### Statistiques descriptives")
    st.write(data.describe())
    
    st.write("### Vérification des valeurs manquantes")
    st.write(data.isnull().sum())

# Fonction pour visualiser les données
def visualiser_donnees(data):
    st.write("### Visualisation des données")
    
    fig, ax = plt.subplots()
    sns.countplot(x='Evolution', data=data, ax=ax)
    ax.set_title('Distribution de la survenue de décès')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x='Evolution', y='Age', data=data, ax=ax)
    ax.set_title('Impact de l\'âge sur la survenue de décès')
    st.pyplot(fig)

# Fonction pour préparer les données
def preparer_donnees(data):
    # Conversion des variables catégoriques en numériques
    data = pd.get_dummies(data, drop_first=True)
    
    # Mise à l'échelle des variables numériques
    scaler = StandardScaler()
    data[['Age', 'Pression_artérielle']] = scaler.fit_transform(data[['Age', 'Pression_artérielle']])
    
    return data, scaler

# Fonction pour séparer les données en ensembles d'entraînement et de test
def separer_donnees(data):
    X = data.drop('Evolution', axis=1)  # Variables explicatives
    y = data['Evolution']  # Variable cible
    
    # Séparation des données (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Fonction pour entraîner et évaluer le modèle
def entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test):
    # Modélisation : Régression logistique
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = log_model.predict(X_test)

    # Rapport de classification
    st.write("### Rapport de classification")
    st.text(classification_report(y_test, y_pred))

    # Courbe ROC pour évaluer la performance du modèle
    y_prob = log_model.predict_proba(X_test)[:, 1]  # Probabilités de prédiction pour la classe 1 (décès)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'Régression Logistique (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label="Modèle aléatoire (AUC = 0.50)")
    ax.set_xlabel('Taux de faux positifs (FPR)')
    ax.set_ylabel('Taux de vrais positifs (TPR)')
    ax.set_title('Courbe ROC')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    return log_model

# Fonction pour effectuer une prédiction pour un nouveau patient
def predire_pour_nouveau_patient(log_model, scaler):
    st.write("### Entrez les données pour un nouveau patient")
    age = st.number_input("Âge", min_value=0, max_value=100, value=50)
    pression_art = st.number_input("Pression artérielle", min_value=0, max_value=200, value=120)
    traitement = st.radio("Traitement", [0, 1], index=1)
    
    # Préparation des données du patient
    new_patient = pd.DataFrame({
        'Age': [age],
        'Pression_artérielle': [pression_art],
        'Traitement': [traitement]
    })
    
    # Mise à l'échelle des données
    new_patient = pd.get_dummies(new_patient, drop_first=True)
    new_patient = scaler.transform(new_patient)
    
    # Prédiction de la probabilité de décès
    prob_deces = log_model.predict_proba(new_patient)[:, 1]
    st.write(f"Probabilité de décès pour ce patient : {prob_deces[0]:.2f}")

# Création de l'interface Streamlit
def app():
    st.title("Étude pronostique de la survenue de décès après traitement")

    data = charger_donnees()

    if data is not None:
        # Explorations des données
        st.sidebar.header("Exploration des données")
        if st.sidebar.button("Explorer les données"):
            explorer_donnees(data)
        
        # Visualisation des données
        st.sidebar.header("Visualisation des données")
        if st.sidebar.button("Visualiser les données"):
            visualiser_donnees(data)

        # Préparation et entraînement du modèle
        st.sidebar.header("Modélisation")
        if st.sidebar.button("Entraîner le modèle"):
            data, scaler = preparer_donnees(data)
            X_train, X_test, y_train, y_test = separer_donnees(data)
            log_model = entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test)

        # Prédiction pour un nouveau patient
        st.sidebar.header("Prédiction pour un nouveau patient")
        if st.sidebar.button("Faire une prédiction"):
            if 'log_model' in locals() and 'scaler' in locals():
                predire_pour_nouveau_patient(log_model, scaler)
            else:
                st.write("Entraînez d'abord le modèle avant de faire une prédiction.")

if __name__ == '__main__':
    app()
