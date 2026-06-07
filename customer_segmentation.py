import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================================
# STEP 1: load and inspect the data
# =====================================================================
df = pd.read_csv('customer_segmentation_data.csv')
print("--- Data Snapshot ---")
print(df.head())

# =====================================================================
# STEP 2: Preprocessing & Scaling (The Math Pre-req)
# =====================================================================
# We select the numerical attributes we want our algorithm to look at
features = ['Age', 'Annual_Income', 'Recency_Days', 'Frequency_Orders', 'Monetary_Spend']
X = df[features]

# Scaling data ensures no single column dominates due to its measurement scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================================
# STEP 3: The Elbow Method (Finding the Perfect 'K')
# =====================================================================
# We don't guess 'K' (number of clusters). We mathematically find it.
wcss = [] # Within-Cluster Sum of Squares (measures how tight clusters are)

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method to Find Optimal Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# =====================================================================
# STEP 4: Train the Final K-Means Model
# =====================================================================
# Based on the Elbow graph, the "bend" happens sharply at K = 3.
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# =====================================================================
# STEP 5: Profile and Analyze the Segments (Business Context)
# =====================================================================
# Group by the new Cluster column and find the averages of our metrics
cluster_profile = df.groupby('Cluster')[features].mean().round(2)
cluster_profile['Customer_Count'] = df.groupby('Cluster')['Customer_ID'].count()

print("\n--- Final Customer Segment Profiles ---")
print(cluster_profile)

# =====================================================================
# STEP 6: Visualize the Segments
# =====================================================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='Frequency_Orders', 
    y='Monetary_Spend', 
    hue='Cluster', 
    palette='Set1', 
    s=100, 
    alpha=0.8
)
plt.title('Customer Segments: Frequency vs total Monetary Spend')
plt.xlabel('Total Orders placed (Frequency)')
plt.ylabel('Total Spend ($) (Monetary)')
plt.legend(title='Customer Cluster')
plt.grid(True)
plt.show()