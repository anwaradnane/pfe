import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Fonction pour le modèle de la méthode Chain Ladder
def chain_ladder_method(Triangle):
    st.subheader("Méthode Chain Ladder")

    # Manipulation des données
    Triangle.set_index(Triangle['origine'], inplace=True)
    del Triangle['origine']

    # Affichage du Triangle
    st.write("Triangle:")
    st.write(Triangle)
    fig, ax = plt.subplots(figsize=(12, 6))
    Triangle.T.plot(ax=ax)
    st.pyplot(fig)

    # Utilisation de Chain_Ladder
    facteurs = []
    for col in Triangle.columns[:-1]:
        facteurs.append(Triangle[str(int(col) + 1)].sum() / Triangle[col][:-int(col)].sum())
    facteurs = np.array(facteurs)

    # Affichage des facteurs
    st.write("Facteurs:")
    per_dev = np.array([(i+1) for i in range(9)])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(per_dev, facteurs)
    ax.set_xlabel('Période du Développement')
    ax.set_ylabel('Facteurs')
    st.pyplot(fig)

    # Plot des facteurs (Scatter après application de la régression linéaire)
    st.write("Plot des facteurs (régression linéaire):")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Plot des facteurs')
    ax.set_xlabel('Période de Développement')
    ax.set_ylabel('Facteur')
    sns.regplot(x=per_dev, y=np.log(facteurs-1), ax=ax)
    st.pyplot(fig)

    # Fitting du modèle
    model = LinearRegression()
    model.fit(per_dev.reshape(-1, 1), np.log(facteurs-1))
    delta = np.array([(i+10) for i in range(101)])
    delta = np.exp(model.intercept_ + model.coef_ * delta) + 1

    # Compléter le Triangle
    for i, col in enumerate(Triangle.columns[1:]):
        for j in range(i+1):
            Triangle[col].at[2021-j] = facteurs[i] * Triangle[str(int(col)-1)].at[2021-j]
    

    # Affichage du Triangle complété
    st.write("Triangle complété:")
    st.write(Triangle)
    fig, ax = plt.subplots(figsize=(12, 6))
    Triangle.T.plot(ax=ax)
    st.pyplot(fig)

    # Calcul d'ultim et IBNR
    Triangle['ultim'] = Triangle['10']*delta.prod()
    Triangle['IBNR'] = Triangle['ultim'].subtract(Triangle['10'])

    # Affichage des résultats finaux
    st.write("Résultats finaux:")
    st.write(Triangle)
    st.write("Somme:")
    st.write(Triangle.sum())

# Fonction pour le modèle du Mack Chain Ladder
def mack_chain_ladder_model(Triangle):
    st.subheader("Modèle du Mack Chain Ladder")

    # Manipulation des données
    Triangle.set_index(Triangle['origine'], inplace=True)
    del Triangle['origine']

    # Affichage du Triangle
    st.write("Triangle:")
    st.write(Triangle)
    fig, ax = plt.subplots(figsize=(12, 6))
    Triangle.T.plot(ax=ax)
    st.pyplot(fig)

    # Utilisation de Mack Chain Ladder
    facteurs = []
    for col in Triangle.columns[:-1]:
        facteurs.append(Triangle[str(int(col) + 1)].sum() / Triangle[col][:-int(col)].sum())
    facteurs = np.array(facteurs)

    # Affichage des facteurs
    st.write("Facteurs:")
    per_dev = np.array([(i+1) for i in range(9)])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(per_dev, facteurs)
    ax.set_xlabel('Période du Développement')
    ax.set_ylabel('Facteurs')
    st.pyplot(fig)

    # Plot des facteurs (Scatter après application de la régression linéaire)
    st.write("Plot des facteurs (régression linéaire):")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Plot des facteurs')
    ax.set_xlabel('Période de Développement')
    ax.set_ylabel('Facteur')
    sns.regplot(x=per_dev, y=np.log(facteurs-1), ax=ax)
    st.pyplot(fig)
    
    # Compléter le Triangle
    for i, col in enumerate(Triangle.columns[1:]):
        for j in range(i+1):
            Triangle[col].at[2021-j] = facteurs[i] * Triangle[str(int(col)-1)].at[2021-j]
    Triangle1 = Triangle
    
    # Fitting du modèle
    model = LinearRegression()
    model.fit(per_dev.reshape(-1, 1), np.log(facteurs-1))
    delta = np.array([(i+10) for i in range(101)])
    delta = np.exp(model.intercept_ + model.coef_ * delta) + 1

    # Calcul de sigma
    t = np.array(Triangle)
    sigma = []
    for j in range(9):
        s = 0
        for i in range(10):
            s += t[i, j] * (((t[i, j+1] / t[i, j]) - facteurs[j])**2)
        sigma.append(np.sqrt(s / (9-j)))

    # Compléter le Triangle
    for i, col in enumerate(Triangle.columns[1:]):
        for j in range(i+1):
            Triangle[col].at[2021-j] = Triangle[str(int(col)-1)].at[2021-j] + (sigma[j] * np.sqrt(Triangle[str(int(col)-1)].at[2021-j]))

    # Affichage des résultats
    st.write("Résultats du modèle du Mack Chain Ladder:")
    st.write("Coefficients de régression:")
    st.write(f"Intercept: {model.intercept_}, Coefficient: {model.coef_}")
    st.write("Sigma:")
    st.write(sigma)
    st.write("Delta:")
    st.write(delta)
    st.write("Produit de Delta:")
    st.write(delta.prod())

    # Calcul d'ultim et IBNR
    Triangle['ultim'] = Triangle['10']*delta.prod()
    Triangle['IBNR'] = Triangle['ultim'].subtract(Triangle['10'])

    # Affichage des résultats finaux
    st.write("Résultats finaux:")
    st.write(Triangle)
    st.write("Somme:")
    st.write(Triangle.sum())

# Code principal
st.title("Étude des Méthodes Chain Ladder et Mack Chain Ladder")
st.sidebar.title("Sélection de la Méthode")

# Importation des données du Scor
  Data_Scor = {
        'origine': [2012+i for i in range(10)],
        '1': [508700, 20100, 342200, 573600, 117700, 156300, 55800, 142600, 314300, 206800],
        '2': [836800, 433600, 906000, 1159200, 962400, 644700, 404200, 682400, 539700, np.nan],
        '3': [1094400, 543800, 1391100, 1581800, 1587100, 1172300, 1095500, 1314800, np.nan, np.nan],
        '4': [1189900, 1073300, 1623400, 2133200, 2226600, 1296400, 1237900, np.nan, np.nan, np.nan],
        '5': [1358700, 1380400, 1875700, 2352100, 2603300, 1589600, np.nan, np.nan, np.nan, np.nan],
        '6': [1619300, 1569300, 2224100, 2610400, 2619400, np.nan, np.nan, np.nan, np.nan, np.nan],
        '7': [1801200, 1559000, 2287600, 2709700, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        '8': [1865700, 1622800, 2347600, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        '9': [1866200, 1675700, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        '10': [1886100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    Triangle = pd.DataFrame(Data_Scor)
    Triangle.set_index(Triangle['origine'], inplace=True)
    del Triangle['origine']

# Affichage des données du Scor
st.subheader("Données du Scor:")
st.write(data_scor)

# Sélection de la méthode
method = st.sidebar.selectbox("Sélectionnez une méthode", ["Chain Ladder", "Mack Chain Ladder"])

if method == "Chain Ladder":
    chain_ladder_method(data_scor)
elif method == "Mack Chain Ladder":
    mack_chain_ladder_model(data_scor)
