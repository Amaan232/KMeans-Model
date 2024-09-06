# K-means Clustering for Customer Segmentation

## Project Overview
This project applies the K-means clustering algorithm to segment customers of a shopping mall based on their annual income and spending score. The goal is to group customers into distinct segments to help the mall management better understand their customer base and make informed marketing and business decisions.

## Dataset
The dataset used in this project is `Mall_Customers.csv`, which contains the following columns:
- **CustomerID**: Unique ID for each customer
- **Gender**: Customer gender
- **Age**: Customer age
- **Annual Income**: Customer annual income (k$)
- **Spending Score**: Customer spending score (1-100)

For the clustering analysis, only the `Annual Income` and `Spending Score` columns are used.

## Clustering Methodology
The K-means algorithm is used to group customers into clusters based on their similarities in annual income and spending score.

### Steps:
1. **Data Preprocessing**: 
    - Load the dataset.
    - Inspect for null values and basic information about the data.
2. **Selecting the Number of Clusters (Elbow Method)**:
    - The within-cluster sum of squares (WCSS) is calculated for 1 to 10 clusters.
    - The elbow graph is plotted to determine the optimal number of clusters.
3. **Customer Segmentation**:
    - K-means clustering is applied with the selected number of clusters (5).
    - The clusters are visualized using a scatter plot.

### Visualization:
- The clusters are plotted in a 2D scatter plot where each point represents a customer.
- The plot includes:
    - **Customer groups** represented by different colors.
    - **Centroids** of the clusters.

## Code

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
customer_data = pd.read_csv("Mall_Customers.csv")

# Select relevant features for clustering
x = customer_data.iloc[:,[3,4]].values

# Elbow Method to determine optimal number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=41)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
sns.set()
plt.plot(range(1,11),wcss)
plt.title('Elbow Graph')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-means with optimal number of clusters (5)
kmeans = KMeans(n_clusters=5,init='k-means++',random_state=41)
Y = kmeans.fit_predict(x)

# Visualizing the clusters
plt.figure(figsize=(8,8))
plt.scatter(x[Y==0,0],x[Y==0,1], s=50, c= 'green', label = 'Cluster 1')
plt.scatter(x[Y==1,0],x[Y==1,1], s=50, c= 'red', label = 'Cluster 2')
plt.scatter(x[Y==2,0],x[Y==2,1], s=50, c= 'blue', label = 'Cluster 3')
plt.scatter(x[Y==3,0],x[Y==3,1], s=50, c= 'grey', label = 'Cluster 4')
plt.scatter(x[Y==4,0],x[Y==4,1], s=50, c= 'orange', label = 'Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label = 'Centroid')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
