# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13Py8cRvW9vH6dnB0vQYvXJJl0LHfokrc
"""

import streamlit as st

"""
**Configuration et importation des bibliothèques**"""

pip install lifelines

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from lifelines import CoxPHFitter
from lifelines.plotting import plot_lifetimes

""" **Chargement des données**"""

# Charger les données
from google.colab import files
uploaded = files.upload()
# Charger le fichier Excel dans un DataFrame
data = pd.read_excel("Donnnées_Projet_M2SID2023_2024.xlsx", header=1)
# Afficher les premières lignes des données
data.head(10)

#  Renommer la colonne 'Unnamed: 13' en 'Traitement'
data.rename(columns={'Unnamed: 13': 'Traitement'}, inplace=True)
# Afficher les 10 premières lignes des données
data.head(10)

"""**Exploration et prétraitement** **des** **données**"""

# Dimensions du jeu de données
print("Dimensions des données :", data.shape)

# Résumé statistique
print("\nStatistiques descriptives :")
data.describe()

# Afficher le type des colonnes
print(data.dtypes)

# Vérification des valeurs manquantes
print("Valeurs manquantes par colonne :")
print(data.isnull().sum())

#  Gérer les valeurs catégoriques
# Convertir les valeurs textuelles ('OUI', 'NON', etc.) en numériques
data = data.replace({'OUI': 1, 'NON': 0, 'Homme': 1, 'Femme': 0, 'Deces': 1, 'Vivant': 0, 'Thrombolyse': 1, 'Chirurgie': 2})
data.head(10)

# Afficher le type des colonnes
print(data.dtypes)

"""**Visualisations**"""

import matplotlib.pyplot as plt
import seaborn as sns
# Distribution des âges
sns.histplot(data['AGE'], kde=True)
plt.title("Distribution des âges")
plt.show()

# Analyse de la relation entre traitement et décès
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Traitement', hue='Evolution', data=data)
plt.title("Évolution par type de traitement")
# Légende personnalisée pour indiquer 1 = Décès, 0 = Vivant
plt.legend(title='Évolution', labels=['Vivant (0)', 'Décès (1)'])
plt.show()

# Matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

"""**Préparation des données pour la modélisation**"""

from sklearn.model_selection import train_test_split

# Définir les variables explicatives et la cible
X = data.drop(columns=['Evolution'])
y = data['Evolution']
# Diviser les données en ensembles d'entraînement et de test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Taille de l'ensemble d'entraînement :", X_train.shape)
print("Taille de l'ensemble de test :", X_test.shape)

"""**Modélisation**

**1.Régression logistique**
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Entraîner un modèle de régression logistique
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Prédictions
y_pred = log_model.predict(X_test)
# Évaluation
print("Rapport de classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.show()

"""**2.Random Forest**"""

from sklearn.ensemble import RandomForestClassifier
# Modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédictions
y_pred_rf = rf_model.predict(X_test)
# Évaluation
print("Rapport de classification pour Random Forest :")
print(classification_report(y_test, y_pred_rf))

"""**Comparaison des modèles**

**Courbes ROC**
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Entraîner un modèle de régression logistique
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Entraîner un modèle de forêt aléatoire
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Évaluation du modèle de régression logistique
print("Rapport de classification - Régression Logistique :")
print(classification_report(y_test, y_pred_log))
print("Matrice de confusion - Régression Logistique :")
print(confusion_matrix(y_test, y_pred_log))

# Évaluation du modèle de forêt aléatoire
print("Rapport de classification - Forêt Aléatoire :")
print(classification_report(y_test, y_pred_rf))
print("Matrice de confusion - Forêt Aléatoire :")
print(confusion_matrix(y_test, y_pred_rf))

# Prédictions de probabilité pour la régression logistique
y_prob_log = log_model.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
roc_auc_log = auc(fpr_log, tpr_log)

# Prédictions de probabilité pour la forêt aléatoire
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Affichage des courbes ROC
plt.figure(figsize=(12, 8))
plt.plot(fpr_log, tpr_log, label=f"Régression Logistique (AUC = {roc_auc_log:.2f})", color="blue", lw=2)
plt.plot(fpr_rf, tpr_rf, label=f"Forêt Aléatoire (AUC = {roc_auc_rf:.2f})", color="green", lw=2)
plt.plot([0, 1], [0, 1], 'k--', label="Ligne aléatoire (AUC = 0.50)", lw=1, color="red")

# Personnalisation des axes et du titre
plt.title("Courbes ROC des modèles", fontsize=16)
plt.xlabel("Taux de faux positifs (FPR)", fontsize=12)
plt.ylabel("Taux de vrais positifs (TPR)", fontsize=12)

# Ajustement de la légende
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.show()