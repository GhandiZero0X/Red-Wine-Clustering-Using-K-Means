import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load data
file_path = 'wine-clustering_Clean_MissingValue_Outlier.csv'
data_normalisasi = pd.read_csv(file_path)

# 2. Data awal sebelum normalisasi
print("========== Data Awal Sebelum Normalisasi ==========")
print("Data Awal : \n", data_normalisasi.head())
print(data_normalisasi.info())

print("\n========== Proses Normalisasi ==========")

# 3. Pilih kolom numerik
data_numerik = data_normalisasi.select_dtypes(include=['float64', 'int64']).columns
print("\nNama Kolom Numerik:")
print(data_numerik.tolist())

# 4. Normalisasi dengan Z-Standardization
scaler = StandardScaler()
data_normalisasi[data_numerik] = scaler.fit_transform(data_normalisasi[data_numerik])

# 5. Data setelah normalisasi
print("\n========== Data Setelah Normalisasi ==========")
print("Data Setelah Normalisasi : \n", data_normalisasi.head())
print(data_normalisasi.info())

# 6. Cek distribusi data setelah normalisasi
for c in list(data_normalisasi):
    print(f"\nValue counts pada kolom '{c}':\n", data_normalisasi[c].value_counts())

# 7. Simpan data yang sudah dinormalisasi
output_path = 'wine-clustering_Normalisasi_Z-Standardization.csv'
data_normalisasi.to_csv(output_path, index=False)
print(f"\nData yang sudah bersih disimpan ke: {output_path}")
