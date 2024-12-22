import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Fonction de prétraitement des données avec gestion des colonnes
def preparer_donnees(data):
    # Afficher les noms de colonnes pour débogage
    st.write("Colonnes disponibles dans les données :", data.columns)

    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = ['AGE', 'Hypertension Arterielle', 'Evolution']  # Colonnes correctes
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"Les colonnes suivantes sont manquantes dans les données : {', '.join(missing_columns)}")
        return None, None  # Retourner None si les colonnes sont manquantes
    
    # Traitement des valeurs manquantes pour les autres colonnes (si nécessaire)
    data = data.fillna({
        'Evolution': data['Evolution'].mode()[0]  # Remplacer les valeurs manquantes dans 'Evolution' par la modalité la plus fréquente
    })
    
    # Encodage de la colonne 'Evolution' (Décès -> 1, Vivant -> 0)
    data['Evolution'] = data['Evolution'].apply(lambda x: 1 if x == 'Décès' else 0)
    
    # Encodage des variables catégorielles (si elles existent)
    data = pd.get_dummies(data, drop_first=True)
    
    # Mise à l'échelle des variables numériques
    scaler = StandardScaler()
    data[['AGE', 'Hypertension Arterielle']] = scaler.fit_transform(data[['AGE', 'Hypertension Arterielle']])
    
    return data, scaler

# Fonction principale de l'application Streamlit
def app():
    # Titre de l'application
    st.title("Analyse de données de décès après traitement")

    # Menu pour sélectionner les différentes sections
    options = ["Télécharger les données", "Analyse des données", "Modélisation"]
    choix = st.sidebar.selectbox("Choisissez une option", options)
    
    # Si l'utilisateur choisit de télécharger un fichier
    if choix == "Télécharger les données":
        uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type=["xlsx"])
        
        if uploaded_file is not None:
            # Charger les données
            data = pd.read_excel(uploaded_file)
            
            # Afficher un aperçu des données brutes
            st.write("Aperçu des données brutes :", data.head())
            
            # Afficher les noms de colonnes
            st.write("Noms des colonnes dans les données :", data.columns)
            
            # Afficher la dimension des données
            st.write("Dimension des données : ", data.shape)
            
            # Afficher un résumé statistique des données
            st.write("Résumé statistique des données :")
            st.write(data.describe())
            
            # Prétraiter les données
            data, scaler = preparer_donnees(data)
            
            if data is not None:
                # Afficher un aperçu des données après prétraitement
                st.write("Données après prétraitement :", data.head())
            else:
                st.warning("Les données n'ont pas pu être traitées en raison de colonnes manquantes.")
    
    # Si l'utilisateur choisit d'analyser les données
    elif choix == "Analyse des données":
        if 'data' in locals() and data is not None:
            st.write("Analyse des données :")
            
            # Exemple de visualisation de distribution
            st.write("Distribution de l'âge des patients")
            plt.hist(data['AGE'], bins=10)
            st.pyplot()
            
            # Exemple de visualisation de la pression artérielle
            st.write("Distribution de l'hypertension artérielle")
            plt.hist(data['Hypertension Arterielle'], bins=10, color='orange')
            st.pyplot()
            
            # Calcul de corrélation entre les variables
            st.write("Matrice de corrélation :")
            corr = data.corr()
            st.write(corr)
        else:
            st.warning("Les données n'ont pas encore été chargées ou traitées.")
        
    # Si l'utilisateur choisit de faire de la modélisation
    elif choix == "Modélisation":
        if 'data' in locals() and data is not None:
            st.write("Modélisation :")
            
            # Séparer les caractéristiques (X) et la variable cible (y)
            X = data[['AGE', 'Hypertension Arterielle']]
            y = data['Evolution']
            
            # Séparer les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Créer et entraîner le modèle de régression logistique
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            # Prédictions sur l'ensemble de test
            y_pred = model.predict(X_test)
            
            # Afficher les résultats de la prédiction
            st.write("Prédictions du modèle :")
            st.write(y_pred)
            
            # Calcul de la précision du modèle
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Précision du modèle : {accuracy:.2f}")
            
            # Affichage du rapport de classification
            st.write("Rapport de classification :")
            st.text(classification_report(y_test, y_pred))
            
            # Affichage de la matrice de confusion
            st.write("Matrice de confusion :")
            conf_matrix = confusion_matrix(y_test, y_pred)
            st.write(conf_matrix)
            
            # Affichage d'une matrice de confusion avec un graphique
            st.write("Matrice de confusion (graphique) :")
            fig, ax = plt.subplots()
            ax.matshow(conf_matrix, cmap='Blues')
            for (i, j), val in np.ndenumerate(conf_matrix):
                ax.text(j, i, f'{val}', ha='center', va='center', color='red')
            st.pyplot(fig)
        else:
            st.warning("Les données n'ont pas encore été chargées ou traitées.")

# Lancer l'application
if __name__ == "__main__":
    app()
