import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Extract data
X = wine.data.features
y = wine.data.targets

# Combine features and target into one DataFrame
wine_df = pd.concat([X, y], axis=1)

# Save to CSV
wine_df.to_csv('wine-clustering.csv', index=False)

print("Data saved to wine-clustering.csv")
