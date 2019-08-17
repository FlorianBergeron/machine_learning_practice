# Régression Linéaire simple

# Data Preprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("50_Startups.csv")
#dataset.drop("State", axis = 1, inplace=True)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# No Data Missing, so we don't manage it.

# Manage catégoriques variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Divid dataset between Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

### Modèle de Régression Linéaire Multiple ###

# Build Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Do new predictions
y_pred = regressor.predict(X_test)
regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))
