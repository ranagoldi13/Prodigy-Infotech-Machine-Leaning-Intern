# Customer Segmentation using K-Means

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Loading of Dataset

df = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Selecting Features

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Elbow Method

wcss = []   # Within cluster sum of squares plot

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=40,
        n_init=10
    )

    model.fit(X)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()


kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

y_pred = kmeans.fit_predict(X)

# Add cluster labels
df["Cluster"] = y_pred

# Print Cluster Counts

print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Visualize Clusters

plt.figure(figsize=(8, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_pred
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker="X"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.show()

# Save Output

df.to_csv("Customer_Segments.csv", index=False)

print("\nClustered dataset saved as Customer_Segments.csv")
