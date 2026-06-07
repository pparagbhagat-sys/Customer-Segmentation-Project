# Customer-Segmentation-Project
Customer behavioral segmentation using K-Means clustering in Python.


## 📊 Project Overview
This project applies Unsupervised Machine Learning to segment a corporate customer base into distinct operational groups. By evaluating behavioral purchase patterns—specifically **Recency, Frequency, and Monetary (RFM)** variables—this data pipeline allows businesses to pivot from mass-marketing to highly optimized, targeted marketing campaigns.

## 🛠️ Data Science Pipeline
1. **Feature Engineering:** Extracted foundational demographic variables and calculated core RFM performance metrics per customer.
2. **Data Feature Scaling:** Implemented `StandardScaler` to normalize distance measurements, ensuring high-value monetary dimensions did not mathematically drown out small-scale frequencies.
3. **Hyperparameter Tuning:** Utilized the Within-Cluster Sum of Squares (WCSS) via the **Elbow Method** to calculate the mathematically optimal cluster break-point ($K=3$).
4. **Model Execution:** Trained the final `KMeans` algorithm to group and tag the customer base.

## 📈 Model Performance & Visualizations

### 1. The Elbow Optimization Curve
<img width="800" height="400" alt="elbow_method" src="https://github.com/user-attachments/assets/0658509b-82d6-4fd4-b7d1-b5b363236c15" />


### 2. Final Cluster Distinctions
<img width="1000" height="600" alt="customer_cluster" src="https://github.com/user-attachments/assets/059c7670-d4ba-4dfc-90d3-5bafe2441cd2" />


---

## 💡 Discovered Customer Profiles & Targeted Business Actions

Based on the computed cluster matrix averages, the customer base separates into three strategic personas:

### 🎯 Cluster 0: Churned / At-Risk Accounts (Count: 257)
* **Profile:** High Recency (Avg 239 days since last order), Low Frequency (1.49 orders), Low Spend ($84.79 total).
* **Corporate Strategy:** Deploy high-value win-back offers and "We Miss You" incentive discounts via automated email triggers.

### 🎯 Cluster 1: Core Steady Consumers (Count: 551)
* **Profile:** Moderate Recency (55 days), Consistent Frequency (6.74 orders), Steady Spend ($674.19 total).
* **Corporate Strategy:** Feed regular cross-selling recommendations, loyalty milestones, and bundle deals to prevent them from drifting into dormancy.

### 🎯 Cluster 2: Executive VIP Elites (Count: 192)
* **Profile:** Ultra-low Recency (7.55 days since last check-out), Extreme Frequency (20.56 orders), Massive Spend ($3,298.08 total).
* **Corporate Strategy:** Provide dedicated customer support channels, early product drop previews, and automatic premium-tier loyalty upgrades.
