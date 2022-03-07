# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
print(dataset.head())
len(dataset)

dataset.columns
# average income vs spending score
plt.plot(dataset['A'])

data = dataset.iloc[:, [3, 4]].values
data

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

distance = []

K=range(3,11)
for i in K:
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(data)
    distance.append(kmeans.inertia_)
    

plt.plot(K, distance)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, random_state = 42)

y_kmeans = kmeans.fit_predict(data)
y_kmeans.shape

dataset['result']=y_kmeans

cluster_0_points = dataset[dataset['result'] == 0].index
cluster_1_points = dataset[dataset['result'] == 1].index
cluster_2_points = dataset[dataset['result'] == 2].index
cluster_3_points = dataset[dataset['result'] == 3].index
cluster_4_points = dataset[dataset['result'] == 4].index

kmeans.cluster_centers_   # CENTROIDS VALUES

# Visualising the clusters
plt.scatter(data[cluster_0_points, 0],
            data[cluster_0_points, 1], 
            s = 100, 
            c = 'red',
            label = 'Customer 1')


plt.scatter(data[cluster_1_points, 0], 
            data[cluster_1_points, 1], 
            s = 100, 
            c = 'blue',
            label = 'Customer 2')


plt.scatter(data[cluster_2_points, 0], data[cluster_2_points, 1], s = 100, c = 'green', label = 'Customer 3')
plt.scatter(data[cluster_3_points, 0], data[cluster_3_points, 1], s = 100, c = 'cyan', label = 'Customer 4')
plt.scatter(data[cluster_4_points, 0], data[cluster_4_points, 1], s = 100, c = 'magenta', label = 'Customer 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()