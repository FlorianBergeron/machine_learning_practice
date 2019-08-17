# CLASSIFICATION - SVM

# =============================================================================
### Data Preprocessing ###

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
#dataset.drop("Country", axis = 1, inplace=True)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Divid dataset between Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# =============================================================================

# =============================================================================
### SVM Classification model ###

# Build Model
from sklearn.svm import SVC # Support Vector for Classification (SVC), because there is a class for regression too (SVR)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
# =============================================================================

# =============================================================================
### CONFUSION METRICS & DETERMINE ACCURACY (AR & ER) ###

# Do now predictions
y_pred = classifier.predict(X_test)

# Confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# =============================================================================

# =============================================================================
### VISUALIZE RESULTS

from matplotlib.colors import ListedColormap

# Set variables
X_set, y_set = X_train, y_train

# Draw on each pixel a color depending on model's prediction and determine 2 classification's areas (0 or 1 / No or Yes)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# Draw RLOG model's line between 2 classification's areas
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))

# Draw X & y limits
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Feature scalling to set OBS PTS from training set on graph
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

# Build and display graph
plt.title('TRAINING SET\'S RESULT')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()
# =============================================================================
