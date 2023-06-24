# Emircan Karagöz
# Onur Doğan
# Emre Karataş

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pickle
import seaborn as sns


# Load the dataset
dfVisual = pd.read_csv("BTCUSDT.csv")

# Convert 'time' column to datetime type
dfVisual['time'] = pd.to_datetime(dfVisual['time'])

# Exclude non-numeric columns
numeric_cols = dfVisual.select_dtypes(include='number').columns
dfVisual_numeric = dfVisual[numeric_cols]

# Histograms
dfVisual_numeric.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Line plots
plt.plot(dfVisual['time'], dfVisual['open'], label='Open')
plt.plot(dfVisual['time'], dfVisual['close'], label='Close')
plt.plot(dfVisual['time'], dfVisual['next close'], label='Next Close')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Cross-correlation table
corr_matrix = dfVisual_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Cross-correlation Matrix')
plt.show()


KMeansClusteringPath = "Saved_Models/KMeansClustering"

# Load the dataset
df = pd.read_csv("BTCUSDT.csv", sep=",")

# Check for null values
null_values = df.isnull().sum()

# Print the columns with null values
# Print the number of null values for each column
for column, count in null_values.items():
    print(f"Column '{column}' has {count} null value(s).")

# Preprocess the data
X = df.drop(['time', 'Buy/Sell'], axis=1)

#print statistical values
print(X.describe())

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Hyperparameter tuning
best_train_silhouette = -1
best_n_clusters = None
best_random_state = None

# Try different values for n_clusters and random_state
for n_clusters in range(2, 6):
    for random_state in range(1, 5):
        # Perform clustering with k-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(X_train)

        # Predict cluster labels for the training set
        train_labels = kmeans.predict(X_train)

        # Calculate silhouette score for the training set
        train_silhouette = silhouette_score(X_train, train_labels)

        # Update the best silhouette score and hyperparameters if necessary
        if train_silhouette > best_train_silhouette:
            best_train_silhouette = train_silhouette
            best_n_clusters = n_clusters
            best_random_state = random_state

# Perform clustering with K-Means using the best hyperparameters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=best_random_state)
kmeans.fit(X_train)

# Predict cluster labels for the training set
train_labels = kmeans.predict(X_train)

# Predict cluster labels for the test set
test_labels = kmeans.predict(X_test)

# Calculate silhouette score and inertia for training set
train_silhouette = silhouette_score(X_train, train_labels)
train_inertia = kmeans.inertia_

# Calculate silhouette score and inertia for test set
test_silhouette = silhouette_score(X_test, test_labels)
test_inertia = kmeans.score(X_test) * -1  # Multiply by -1 to get positive inertia

# Print the results
print("K-Means Training Set:")
print("Number of Clusters:", best_n_clusters)
print("Random State:", best_random_state)
print("K-Means Silhouette Score:", train_silhouette)
print("K-Means Inertia:", train_inertia)

print("\nK-Means Test Set:")
print("K-Means Silhouette Score:", test_silhouette)
print("K-Means Inertia:", test_inertia)



# Create a new DataFrame with the cluster labels
cluster_df = pd.concat([X_train, pd.DataFrame({'Cluster': train_labels})], axis=1)

# Visualize the clusters on train data
plt.scatter(cluster_df['open'], cluster_df['close'], c=cluster_df['Cluster'])
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('Clustering Results')
plt.show()



# Perform clustering with DBSCAN
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Perform clustering with DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
train_clusters = dbscan.fit_predict(X_train)


# Calculate the silhouette score for training clusters
train_silhouette_score = silhouette_score(X_train, train_clusters)

# Predict clusters on the test set
test_clusters = dbscan.fit_predict(X_test)

# Calculate the silhouette score for test clusters
test_silhouette_score = silhouette_score(X_test, test_clusters)



# Print the evaluation scores
print("\n\nDBSCAN Training Silhouette Score:", train_silhouette_score)
print("DBSCAN Test Silhouette Score:", test_silhouette_score)

# Visualize the training clusters
plt.scatter(X_train[:, 0], X_train[:, 1], c=train_clusters, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('DBSCAN Clustering (Training Set)')
plt.show()


# Visualize the test clusters
# since we dropped time and, buy/sell columns X_test[:, 0] represents "open" and X_test[:, 1] represents "close" columns
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_clusters, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('DBSCAN Clustering (Test Set)')
plt.show()


# saving best performed model
pickle.dump(kmeans, open(KMeansClusteringPath, 'wb'))








