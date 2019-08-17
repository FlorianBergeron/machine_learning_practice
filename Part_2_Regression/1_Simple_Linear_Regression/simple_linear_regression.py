# Régression Linéaire simple

### Data Preprocessing ###

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# =============================================================================
#   
#  NOTE:
#
#  On devrait normalement supprimer ce passage,
#  car il n'est pas utile, nous n'avons pas de 
#  données manquantes dans ce dataset, n'y de
#  variables catégoriques et pas besoin de 
#  faire de feature scalling, car nous sommes 
#  dans une RLS.
#
# =============================================================================
# 
# Manage missing data
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# 
# imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
# 
# Manage categoric variables
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# 
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
# 
# Feature scalling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# =============================================================================

# Divid dataset between Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1.0/3, random_state = 0)

### Modèle de Régression Linéaire Simple ###

# Build Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Do new predictions
y_pred = regressor.predict(X_test)

# How to predict a new variable outside the scope of our dataset
regressor.predict([[15]])

# Visualize results
plt.scatter(X_test, Y_test, color = 'red') # Les points d'observation du training set.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # La courbe de prédiction de notre modèle RLS.
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
