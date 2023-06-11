import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.title("Triangle d'IBNR")

def chain_ladder_method(Triangle):
    st.subheader("Méthode Chain Ladder")

    st.write("Triangle:")
    st.write(Triangle)
    fig, ax = plt.subplots(figsize=(12, 6))
    Triangle.T.plot(ax=ax)
    st.pyplot(fig)

    facteurs = []
    for col in Triangle.columns[:-1]:
        facteurs.append(Triangle[str(int(col) + 1)].sum() / Triangle[col][:-int(col)].sum())
    facteurs = np.array(facteurs)

    st.write("Facteurs:")
    per_dev = np.array([(i+1) for i in range(Triangle.shape[1]-1)])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(per_dev, facteurs)
    ax.set_xlabel('Période du Développement')
    ax.set_ylabel('Facteurs')
    st.pyplot(fig)

    st.write("Plot des facteurs (régression linéaire):")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Plot des facteurs')
    ax.set_xlabel('Période de Développement')
    ax.set_ylabel('Facteur')
    sns.regplot(x=per_dev, y=np.log(facteurs-1), ax=ax)
    st.pyplot(fig)

    model = LinearRegression()
    model.fit(per_dev.reshape(-1, 1), np.log(facteurs-1))
    delta = np.array([(i+10) for i in range(101)])
    delta = np.exp(model.intercept_ + model.coef_ * delta) + 1

    for i, col in enumerate(Triangle.columns[1:]):
        for j in range(i+1):
            Triangle[col].at[2021-j] = facteurs[i] * Triangle[str(int(col)-1)].at[2021-j]

    st.write("Triangle complété:")
    st.write(Triangle)
    fig, ax = plt.subplots(figsize=(12, 6))
    Triangle.T.plot(ax=ax)
    st.pyplot(fig)

    Triangle['ultim'] = Triangle['10']*delta.prod()
    Triangle['IBNR'] = Triangle['ultim'].subtract(Triangle['10'])

    st.write("Résultats finaux:")
    st.write(Triangle)
    st.write("Somme:")
    st.write(Triangle.sum())

def mack_chain_ladder_model(Triangle):
    st.subheader("Modèle du Mack Chain Ladder")

    st.write("Triangle:")
    st.write(Triangle)
    fig, ax = plt.subplots(figsize=(12, 6))
    Triangle.T.plot(ax=ax)
    st.pyplot(fig)

    facteurs = []
    for col in Triangle.columns[:-1]:
        facteurs.append(Triangle[str(int(col) + 1)].sum() / Triangle[col][:-int(col)].sum())
    facteurs = np.array(facteurs)

    st.write("Facteurs:")
    per_dev = np.array([(i+1) for i in range(Triangle.shape[1]-1)])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(per_dev, facteurs)
    ax.set_xlabel('Période du Développement')
    ax.set_ylabel('Facteurs')
    st.pyplot(fig)

    st.write("Plot des facteurs (régression linéaire):")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Plot des facteurs')
    ax.set_xlabel('Période de Développement')
    ax.set_ylabel('Facteur')
    sns.regplot(x=per_dev, y=np.log(facteurs-1), ax=ax)
    st.pyplot(fig)

    for i, col in enumerate(Triangle.columns[1:]):
        for j in range(i+1):
            Triangle[col].at[2021-j] = facteurs[i] * Triangle[str(int(col)-1)].at[2021-j]
    Triangle1 = Triangle

    model = LinearRegression()
    model.fit(per_dev.reshape(-1, 1), np.log(facteurs-1))
    delta = np.array([(i+10) for i in range(101)])
    delta = np.exp(model.intercept_ + model.coef_ * delta) + 1

    t = np.array(Triangle)
    sigma = []
    for j in range(Triangle.shape[1]-1):
        s = 0
        for i in range(Triangle.shape[0]):
            s += t[i, j] * (((t[i, j+1] / t[i, j]) - facteurs[j])**2)
        sigma.append(np.sqrt(s / (Triangle.shape[1]-2-j)))

    for i, col in enumerate(Triangle.columns[1:]):
        for j in range(i+1):
            Triangle[col].at[2021-j] = Triangle[str(int(col)-1)].at[2021-j] + (sigma[j] * np.sqrt(Triangle[str(int(col)-1)].at[2021-j]))

    st.write("Résultats du modèle du Mack Chain Ladder:")
    st.write("Coefficients de régression:")
    st.write(f"Intercept: {model.intercept_}, Coefficient: {model.coef_}")
    st.write("Sigma:")
    st.write(sigma)
    st.write("Delta:")
    st.write(delta)
    st.write("Produit de Delta:")
    st.write(delta.prod())

    st.write("Triangle complété:")
    st.write(Triangle)
    fig, ax = plt.subplots(figsize=(12, 6))
    Triangle.T.plot(ax=ax)
    st.pyplot(fig)

    Triangle['ultim'] = Triangle['10']*delta.prod()
    Triangle['IBNR'] = Triangle['ultim'].subtract(Triangle['10'])

    st.write("Résultats finaux:")
    st.write(Triangle)
    st.write("Somme:")
    st.write(Triangle.sum())

# Création des données
data_choice = st.sidebar.selectbox("Sélectionnez un jeu de données", ("Données 1", "Données 2"))

if data_choice == "Données 1":
    Triangle_data = {
        '10': [5000, 4000, 3000, 2000, 1000, 0, 0, 0, 0, 0],
        '9': [6000, 5000, 4000, 3000, 2000, 1000, 0, 0, 0, 0],
        '8': [7000, 6000, 5000, 4000, 3000, 2000, 1000, 0, 0, 0],
        '7': [8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0, 0],
        '6': [9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0],
        '5': [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000],
        '4': [11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000],
        '3': [12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000],
        '2': [13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000],
        '1': [14000, 13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000],
    }

    Triangle = pd.DataFrame(Triangle_data)

elif data_choice == "Données 2":
    Triangle_data = {
        '10': [5000, 4000, 3000, 2000, 1000, 0, 0, 0, 0, 0],
        '9': [6000, 5000, 4000, 3000, 2000, 1000, 0, 0, 0, 0],
        '8': [7000, 6000, 5000, 4000, 3000, 2000, 1000, 0, 0, 0],
        '7': [8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0, 0],
        '6': [9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0],
        '5': [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000],
        '4': [11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000],
        '3': [12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000],
        '2': [13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000],
        '1': [14000, 13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000],
    }

    Triangle = pd.DataFrame(Triangle_data)

# Affichage des résultats
chain_ladder_method(Triangle)
mack_chain_ladder_model(Triangle)
