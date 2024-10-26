import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway

# Load data
df = pd.read_csv('wine-clustering_Normalisasi_Z-Standardization.csv')

# Drop the 'class' column
df = df.drop(columns=['class'])
print("Data setelah menghapus kolom 'class': \n", df.head())

# Check data shape and statistics
print("\nBentuk Data : \n", df.shape)
print("\nStatistik Deskriptif : \n", df.describe())
print("\nData Null : \n", df.isnull().sum())

# Visualisasi distribusi data pada setiap fitur
num_columns = len(df.columns)
num_rows = (num_columns + 2) // 3  # Automatically calculate rows for 3 columns per row
plt.figure(1, figsize=(15, num_rows * 2))
for n, col in enumerate(df.columns):
    plt.subplot(num_rows, 3, n + 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[col], kde=True, stat="density", kde_kws={"cut": 3}, bins=20)
    plt.title(f'Distplot of {col}')
plt.show()

# Determine optimal k with Elbow Method
X = df.values
inertia = []
for n in range(1, 11):
    algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=0)
    algorithm.fit(X)
    inertia.append(algorithm.inertia_)

# Calculate optimal k based on relative inertia change
inertia_diff = np.diff(inertia)
inertia_diff_ratio = inertia_diff[:-1] / inertia_diff[1:]
optimal_k = np.argmax(inertia_diff_ratio) + 2
print(f"\nNilai k Paling Optimal berdasarkan Elbow Method (Perubahan Relatif Inertia): {optimal_k}")

# Plot Elbow graph
plt.figure(figsize=(15, 6))
plt.plot(range(1, 11), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Build k-means model on the full dataset
algorithm = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=0)
algorithm.fit(X)
labels = algorithm.labels_
centroids = algorithm.cluster_centers_

# Tambahkan hasil clustering ke dalam DataFrame
df['cluster'] = labels

# Visualize clusters (only for the first two features for 2D plot)
X_2d = df.iloc[:, :2].values  # Use only first two features for visualization
kmeans_2d = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans_2d.fit(X_2d)
labels_2d = kmeans_2d.labels_
centroids_2d = kmeans_2d.cluster_centers_

# Plotting decision boundary using first two features
step = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
Z = kmeans_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(15, 7))
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_2d, s=100)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=300, c='red', alpha=0.5)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('Cluster Visualization (2D Projection)')
plt.show()

# Display Silhouette Score for the main clustering model
score = silhouette_score(X, labels)
print("\nSilhouette Score : ", score)

# Analisis Varians (ANOVA) untuk setiap fitur
print("\n========== Analisis Varians (ANOVA) ==========")
numeric_columns = df.columns[:-1]  # Semua kolom kecuali 'cluster'
for col in numeric_columns:
    # Hanya ambil data untuk cluster yang ada
    cluster_data = [df[df['cluster'] == i][col] for i in range(optimal_k) if i in df['cluster'].unique()]
    
    # Pastikan ada cukup data untuk ANOVA
    if len(cluster_data) > 1 and all(len(data) > 0 for data in cluster_data):
        f_value, p_value = f_oneway(*cluster_data)
        print(f"ANOVA {col}: F-value = {f_value:.3f}, P-value = {p_value:.3f}")
    else:
        print(f"ANOVA tidak dapat dilakukan untuk {col} karena tidak ada cukup cluster.")

# Analisis Fitur Dominan di Setiap Cluster
cluster_summary = df.groupby('cluster')[numeric_columns].mean()
dominant_features = cluster_summary.idxmax(axis=1)
dominant_values = cluster_summary.max(axis=1)

print("\nFitur dominan untuk setiap cluster:")
for cluster_num in dominant_features.index:
    feature = dominant_features[cluster_num]
    value = dominant_values[cluster_num]
    print(f"Cluster {cluster_num}: Fitur dominan adalah '{feature}' dengan rata-rata {value:.3f}")

# Simpan Hasil Clustering ke File CSV
output_path = 'wine-clustering_with_clusters_newProcess.csv'
df.to_csv(output_path, index=False)
print(f"\nData dengan cluster dan class disimpan ke: {output_path}")
