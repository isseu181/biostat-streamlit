import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc

# Titre de l'application
st.title("Étude pronostique des décès après traitement")

# Chargement des données
uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.write("Aperçu des données :", data.head())

    # Préparation des données
    target = st.selectbox("Sélectionnez la variable cible", data.columns)
    features = st.multiselect("Sélectionnez les variables explicatives", data.columns)

    if target and features:
        X = pd.get_dummies(data[features], drop_first=True)
        y = data[target].map({'OUI': 1, 'NON': 0})  # Adapter selon vos données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modélisation
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Rapport de classification :", classification_report(y_test, y_pred))

        # Courbe ROC
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        st.write(f"AUC : {roc_auc:.2f}")

        # Affichage de la courbe ROC
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.legend()
        st.pyplot(fig)
