import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Fonction de prétraitement des données avec gestion des colonnes
def preparer_donnees(data):
    # Afficher les noms de colonnes pour débogage
    st.write("Colonnes disponibles dans les données :", data.columns)

    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = ['Age', 'Pression_artérielle']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"Les colonnes suivantes sont manquantes dans les données : {', '.join(missing_columns)}")
        return None, None  # Retourner None si les colonnes sont manquantes
    
    # Traitement des valeurs manquantes
    if data.isnull().sum().sum() > 0:
        st.write("Des valeurs manquantes ont été détectées, nous allons les remplir.")
        data['Age'].fillna(data['Age'].mean(), inplace=True)
        data['Pression_artérielle'].fillna(data['Pression_artérielle'].mean(), inplace=True)
    
    # Encodage des variables catégorielles (si elles existent)
    data = pd.get_dummies(data, drop_first=True)
    
    # Mise à l'échelle des variables numériques
    scaler = StandardScaler()
    data[['Age', 'Pression_artérielle']] = scaler.fit_transform(data[['Age', 'Pression_artérielle']])
    
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
            
            # Prétraiter les données
            data, scaler = preparer_donnees(data)
            
            if data is not None:
                # Afficher un aperçu des données après prétraitement
                st.write("Données après prétraitement :", data.head())
            else:
                st.warning("Les données n'ont pas pu être traitées en raison de colonnes manquantes.")
    
    # Si l'utilisateur choisit d'analyser les données
    elif choix == "Analyse des données":
        st.write("Analyse des données :")
        st.write("Ajoutez ici des analyses exploratoires, des graphiques, etc.")
        
        # Vous pouvez ajouter des visualisations, comme des histogrammes, des corrélations, etc.
        # Exemple :
        # st.write("Distribution de l'âge des patients")
        # st.histogram(data['Age'])
        
    # Si l'utilisateur choisit de faire de la modélisation
    elif choix == "Modélisation":
        st.write("Modélisation :")
        st.write("Ajoutez ici des modèles de prédiction, comme une régression logistique.")
        
        # Exemple : un simple modèle de régression logistique pour prédire la survenue de décès
        # Modélisez et évaluez un modèle ici, par exemple avec `sklearn.linear_model.LogisticRegression`
    
# Lancer l'application
if __name__ == "__main__":
    app()
