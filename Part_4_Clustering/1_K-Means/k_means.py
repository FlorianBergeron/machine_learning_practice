### CLUSTERING - K-MEANS ###

# =============================================================================
### Data Preprocessing ###

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Mall_Customers.csv")
#dataset.drop("Country", axis = 1, inplace=True)
X = dataset.iloc[:, [3,4]].values

# =============================================================================
### Search optimal cluster number ###

# Use elbox method to find optimal clusters number ("K")
from sklearn.cluster import KMeans

# Withing Cluster Sum of squares
wcss = []

# Search optimal clusters number with a loop
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Display 2D Graph
plt.plot(range(1,11), wcss)
plt.title('ELBOW METHOD')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# =============================================================================
### K-MEANS clustering model ###

# Build model
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Get IVs & VD with scatter method for each cluster (K = 5)
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], c = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'black', label = 'Cluster 5')

# Display 2D Graph
plt.title('CLIENTS\' CLUSTERS')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 to 100)')
plt.legend()

# =============================================================================