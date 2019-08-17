# RÃ©gression Polynomiale

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

### Modèle de Ré©gression Polynomiale ###

# Build Model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Ally our modèle with new matrice X_poly to transform them in a RL model non-linear.
regressor = LinearRegression()
regressor.fit(X_poly, y)

# How to predict a new variable outside the scope of our dataset
regressor.predict([[15]])

# Visualize results
plt.scatter(X, y, color = 'red') # Les points d'observation du training set.
plt.plot(X, regressor.predict(X_poly), color = 'blue') # La courbe de prÃ©diction de notre modÃ¨le RLS.
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
