### REGRESSION - SIMPLE LINEAR

# =============================================================================
### Data Preprocessing ###

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# =!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
#   
#  NOTE:
#
#   This passage should normally be deleted,
#   because it is not useful, we have no
#   missing data in this dataset, no
#   categorical data and no need for
#   do feature scalling, because we are
#   in an simple linear regression.
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
# =!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=

# Divid dataset between Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1.0/3, random_state = 0)
# =============================================================================

# =============================================================================
### Simple Linear Regression model ###

# Build Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Do new predictions
y_pred = regressor.predict(X_test)

# How to predict a new variable outside the scope of our dataset
regressor.predict([[15]])

# Visualize results
plt.scatter(X_test, Y_test, color = 'red') # Training set observation points.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Prediction courbe of our simple linear regression.
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
# =============================================================================
