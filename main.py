import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

Data_Scor = {'origine': [2012+i for i in range(10)],
             '1': [508700, 20100, 342200, 573600, 117700, 156300, 55800, 142600, 314300, 206800],
             '2': [836800, 433600, 906000, 1159200, 962400, 644700, 404200, 682400, 539700, np.nan],
             '3': [1094400, 543800, 1391100, 1581800, 1587100, 1172300, 1095500, 1314800, np.nan, np.nan],
             '4': [1189900, 1073300, 1623400, 2133200, 2226600, 1296400, 1237900, np.nan, np.nan, np.nan],
             '5': [1358700, 1380400, 1875700, 2352100, 2603300, 1589600, np.nan, np.nan, np.nan, np.nan],
             '6': [1619300, 1569300, 2224100, 2610400, 2619400, np.nan, np.nan, np.nan, np.nan, np.nan],
             '7': [1801200, 1559000, 2287600, 2709700, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             '8': [1865700, 1622800, 2347600, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             '9': [1866200, 1675700, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             '10': [1886100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}

# Manipulation des Donnees:
Triangle = pd.DataFrame(Data_Scor)
Triangle.set_index(Triangle['origine'], inplace=True)
del Triangle['origine']

# Affichage du Triangle:
st.write("Triangle:")
st.write(Triangle)

# Plot du Triangle:
st.write("Plot du Triangle:")
fig, ax = plt.subplots(figsize=(12, 6))
Triangle.plot(ax=ax)
st.pyplot(fig)

# Utilisation de Chain_Ladder:
facteurs = []
for col in Triangle.columns[:-1]:
    facteurs.append(Triangle[str(int(col) + 1)].sum() / Triangle[col][:-int(col)].sum())
facteurs = np.array(facteurs)

# Plot des Facteurs en utilisant des scatter plots:
st.write("Plot des Facteurs:")
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Plot des Facteurs')
ax.set_xlabel('Periode_Developpement')
ax.set_ylabel('Facteur')
sns.scatterplot(x=np.arange(1, 10), y=facteurs)
st.pyplot(fig)

# Fitting du Model:
model = LinearRegression()
model.fit(np.arange(1, 10).reshape(-1, 1), np.log(facteurs - 1))
delta = np.array([(i + 10) for i in range(101)])
delta = np.exp(model.intercept_ + model.coef_ * delta) + 1

# Affichage du Fitting du Model - Delta:
st.write("Fitting du Model - Delta:")
st.write(delta)
st.write("Produit de Delta:")
st.write(delta.prod())

# Compléter le Triangle:
for i, col in enumerate(Triangle.columns[1:]):
    for j in range(i + 1):
        Triangle[col].at[2021 - j] = facteurs[i] * Triangle[str(int(col) - 1)].at[2021 - j]

# Affichage du Triangle Complet:
st.write("Triangle Complet:")
st.write(Triangle)

# Plot du Triangle Complet:
st.write("Plot du Triangle Complet:")
fig, ax = plt.subplots(figsize=(12, 6))
Triangle.T.plot(ax=ax)
st.pyplot(fig)

# Calcul de l'ultim et de l'IBNR:
Triangle['ultim'] = Triangle['10'] * delta.prod()
Triangle['IBNR'] = Triangle['ultim'].subtract(Triangle['10'])

# Affichage du Triangle avec ultim et IBNR:
st.write("Triangle avec ultim et IBNR:")
st.write(Triangle)

# Affichage des indices:
st.write("Indices:")
indices = pd.DataFrame({
    'Indice': ['Somme IBNR', 'Somme ultim'],
    'Valeur': [Triangle['IBNR'].sum(), Triangle['ultim'].sum()]
})
st.write(indices)
