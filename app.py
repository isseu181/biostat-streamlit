import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Fonction de chargement des données
def charger_donnees():
    uploaded_file = st.file_uploader("Choisir un fichier Excel", type="xlsx")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("Données chargées avec succès!")
        return data
    else:
        return None

# Fonction de prétraitement des données
def preparer_donnees(data):
    if data.isnull().sum().sum() > 0:
        st.write("Des valeurs manquantes ont été détectées, nous allons les remplir.")
        data['Age'].fillna(data['Age'].mean(), inplace=True)
        data['Pression_artérielle'].fillna(data['Pression_artérielle'].mean(), inplace=True)
        data['Traitement'].fillna(data['Traitement'].mode()[0], inplace=True)
    
    # Encodage des variables catégorielles
    data = pd.get_dummies(data, drop_first=True)
    
    # Mise à l'échelle des variables numériques
    scaler = StandardScaler()
    data[['Age', 'Pression_artérielle']] = scaler.fit_transform(data[['Age', 'Pression_artérielle']])
    
    return data, scaler

# Fonction de séparation des données
def separer_donnees(data):
    X = data.drop('Evolution', axis=1)
    y = data['Evolution']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Fonction d'entraînement et d'évaluation du modèle
def entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
    
    # Calcul de la courbe ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('Courbe ROC')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    return model

# Fonction de prédiction pour un nouveau patient
def predire_pour_nouveau_patient(model, scaler):
    st.write("Entrez les données pour un nouveau patient")
    age = st.number_input("Âge", min_value=0, max_value=100, value=50)
    pression_art = st.number_input("Pression artérielle", min_value=0, max_value=200, value=120)
    traitement = st.radio("Traitement", [0, 1], index=1)
    
    new_patient = pd.DataFrame({'Age': [age], 'Pression_artérielle': [pression_art], 'Traitement': [traitement]})
    new_patient = pd.get_dummies(new_patient, drop_first=True)
    new_patient = scaler.transform(new_patient)
    
    prob_deces = model.predict_proba(new_patient)[:, 1]
    st.write(f"Probabilité de décès pour ce patient : {prob_deces[0]:.2f}")

# Création de l'application Streamlit avec un menu
def app():
    st.title("Pronostic sur la survenue de décès")
    
    # Menu latéral
    menu = ["Accueil", "Prétraitement des données", "Entraînement du modèle", "Prédiction"]
    choix = st.sidebar.selectbox("Choisissez une option", menu)
    
    # Affichage des pages selon le menu sélectionné
    if choix == "Accueil":
        st.write("Bienvenue dans l'application de pronostic sur la survenue de décès. "
                 "Veuillez charger un fichier Excel contenant les données pour commencer.")
    
    elif choix == "Prétraitement des données":
        st.write("### Prétraitement des données")
        data = charger_donnees()
        if data is not None:
            if st.button("Prétraiter les données"):
                data, scaler = preparer_donnees(data)
                st.write("Prétraitement terminé avec succès !")
                st.write(data.head())
    
    elif choix == "Entraînement du modèle":
        st.write("### Entraînement du modèle")
        data = charger_donnees()
        if data is not None:
            if st.button("Entraîner le modèle"):
                data, scaler = preparer_donnees(data)
                X_train, X_test, y_train, y_test = separer_donnees(data)
                model = entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.write("Modèle entraîné avec succès !")

    elif choix == "Prédiction":
        st.write("### Prédiction pour un nouveau patient")
        if 'model' in st.session_state and 'scaler' in st.session_state:
            model = st.session_state.model
            scaler = st.session_state.scaler
            predire_pour_nouveau_patient(model, scaler)
        else:
            st.write("Veuillez entraîner d'abord le modèle avant de faire une prédiction.")

if __name__ == '__main__':
    app()
