# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans


# Loading the dataset

data = pd.read_csv("Mall_Customers.csv")


# Displaying first five rows

print("FIRST FIVE ROWS:")
print(data.head())


# Displaying dataset information

print("\nDATASET INFORMATION:")
print(data.info())


# Displaying number of rows and columns

print("\nDATASET SHAPE:")
print(data.shape)


# Checking missing values

print("\nMISSING VALUES:")
print(data.isnull().sum())


print("\nDUPLICATE RECORDS:")
print(data.duplicated().sum())

data = data.drop_duplicates()

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

plt.figure(figsize=(8, 5))

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)']
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Customer Distribution")

plt.show()

# Finding the optimal number of clusters

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)


# Plotting the Elbow Graph

plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()

data.to_csv("Customer_Segmentation_Results.csv", index=False)

print("Results saved successfully!")

print("\nCUSTOMER GROUP ANALYSIS:")

cluster_analysis = data.groupby('Cluster')[
    ['Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print(cluster_analysis)

cluster_names = {
    0: "Standard Customers",
    1: "Careful Customers",
    2: "Low Spending Customers",
    3: "Premium Customers",
    4: "Budget Customers"
}

data['Customer Type'] = data['Cluster'].map(cluster_names)

print("\nCUSTOMER TYPES:")
print(data[['CustomerID', 'Cluster', 'Customer Type']].head(10))

data.to_csv("Customer_Segmentation_Results.csv", index=False)

print("\nFinal results saved successfully!")