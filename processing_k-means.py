import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway  # Untuk ANOVA
from sklearn.metrics import silhouette_score

# 1. Load Data yang Sudah Dinormalisasi
file_path = 'wine-clustering_Normalisasi_Z-Standardization.csv'
data = pd.read_csv(file_path)

print("\n========== Data Awal ==========")
print("Data Awal : \n", data.head())
data.info()

print("========== Process Pelatihan K-Means Clustering ==========")
if 'class' in data.columns:
    class_column = data['class']
    print("\nKolom 'class' hanya dipisahkan sementara dari clustering, tapi tetap ada dalam data.")

# 3. Pilih kolom numerik yang tersisa untuk clustering
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
if 'class' in numeric_columns:
    numeric_columns = numeric_columns.drop('class')
X = data[numeric_columns]
print(f"\nKolom Numerik yang Digunakan untuk Clustering:\n{numeric_columns.tolist()}")

# 4. Visualisasi Elbow Method
inertia_list = []
k_values = range(1, 11)  # Uji k dari 1 hingga 10

for k in k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=0)
    kmeans_temp.fit(X)
    inertia_list.append(kmeans_temp.inertia_)

print("\nInertia untuk setiap K:")
for i in range(len(k_values)):
    print(f"K = {k_values[i]}: {inertia_list[i]}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_list, 'bx-')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.grid(True)
plt.show()

# 5. Visualisasi Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    kmeans_temp = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans_temp.fit_predict(X)
    sil_score = silhouette_score(X, cluster_labels)
    silhouette_scores.append(sil_score)

print("\nSilhouette Score untuk setiap K:")
for i in range(len(silhouette_scores)):
    print(f"K = {i+2}: {silhouette_scores[i]}")

optimal_k = np.argmax(silhouette_scores) + 2
print(f"\nNilai k Paling Optimal berdasarkan Silhouette Score: {optimal_k}")

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, 'go-')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score untuk Menentukan k Optimal')
plt.grid(True)
plt.show()

# 6. Melatih K-Means dengan k = 3 (sesuai paper)
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(X)
print("Prediksi Cluster:\n", clusters)

data['cluster'] = clusters
print("\nHasil K-Means Clustering:\n", data['cluster'].value_counts())

# 7. Visualisasi Hasil Clustering (2D Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.title(f'K-Means Clustering dengan k=3')
plt.xlabel(numeric_columns[0])
plt.ylabel(numeric_columns[1])
plt.legend()
plt.grid(True)
plt.show()

# 9. Analisis Varians (ANOVA) untuk setiap fitur
print("\n========== Analisis Varians (ANOVA) ==========")
for col in numeric_columns:
    # Hanya ambil cluster yang memiliki data
    cluster_data = [data[data['cluster'] == i][col] for i in range(k) if i in data['cluster'].unique()]
    
    # Pastikan ada cukup data untuk ANOVA
    if len(cluster_data) > 1 and all(len(data) > 0 for data in cluster_data):
        f_value, p_value = f_oneway(*cluster_data)
        print(f"ANOVA {col}: F-value = {f_value:.3f}, P-value = {p_value:.3f}")
    else:
        print(f"ANOVA tidak dapat dilakukan untuk {col} karena tidak ada cukup cluster.")

# 10. Analisis Fitur Dominan di Setiap Cluster
cluster_summary = data.groupby('cluster')[numeric_columns].mean()
dominant_features = cluster_summary.idxmax(axis=1)
dominant_values = cluster_summary.max(axis=1)

print("\nFitur dominan untuk setiap cluster:")
for cluster_num in dominant_features.index:
    feature = dominant_features[cluster_num]
    value = dominant_values[cluster_num]
    print(f"Cluster {cluster_num}: Fitur dominan adalah '{feature}' dengan rata-rata {value:.3f}")

# Gabungkan kembali kolom class dengan hasil clustering
if class_column is not None:
    data['class'] = class_column.reset_index(drop=True)

# 8. Simpan Hasil Clustering ke File CSV
output_path = 'wine-clustering_with_clusters.csv'
data.to_csv(output_path, index=False)
print(f"\nData dengan cluster dan class disimpan ke: {output_path}")