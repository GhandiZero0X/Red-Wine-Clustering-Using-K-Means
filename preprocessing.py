import pandas as pd
import numpy as np

file_path = 'wine-clustering.csv'
data = pd.read_csv(file_path)

print("========== Data Awal ==========")
print("Data Awal : \n", data)

print("Data Info : \n")
data.info()

print("\nPersebaran data pada setiap kolom:")
for c in list(data):
    print(f"\nValue counts pada kolom '{c}':\n", data[c].value_counts())

print("\n========== Pengecekan Missing Value dan Outlier ==========")
print("Missing Value Setiap Kolom : \n", data.isnull().sum())

# Penanganan Missing Value dengan Menghapus Baris yang Memiliki Missing Value
data_clean_missingValue = data.dropna()
print(f"\nData setelah menghapus missing value: {data_clean_missingValue.shape}")

print("\n========== Pengecekan Outlier ==========")
numeric_columns = data_clean_missingValue.select_dtypes(include=['float64', 'int64'])

# Calculate IQR for each numeric column
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("\nBatas bawah IQR:\n", lower_bound)
print("\nBatas atas IQR:\n", upper_bound)

# Identify outliers (values outside 1.5 * IQR range)
outliers = ((numeric_columns < lower_bound) | (numeric_columns > upper_bound))

# Print number of outliers per column
outliers_summary = outliers.sum()
print("\nJumlah Outlier per Kolom:\n", outliers_summary)

# Remove rows with outliers
data_clean_outliers = data_clean_missingValue[~(outliers).any(axis=1)]
print(f"\nData setelah menghapus outlier: \n", data_clean_outliers.shape)
print("data bersih outliers:\n", data_clean_outliers)

# pengecekan ulang untuk outlier setelah di drop
numeric_columns_cleaned = data_clean_outliers.select_dtypes(include=['float64', 'int64'])
outliers_cleaned = ((numeric_columns_cleaned < lower_bound) | (numeric_columns_cleaned > upper_bound))
outliers_summary_cleaned = outliers_cleaned.sum()
print("\nJumlah Outlier per Kolom setelah di drop:\n", outliers_summary_cleaned)

# Step 5: Save the cleaned data
output_path = "wine-clustering_Clean_MissingValue_Outlier.csv"
data_clean_outliers.to_csv(output_path, index=False)

print(f"\nData yang sudah bersih disimpan ke: {output_path}")

