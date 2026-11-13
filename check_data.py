import pandas as pd

# Load dataset
df = pd.read_csv("data/water_potability.csv")

# Display first five rows
print(df.head())

# Display dataset shape
print("\nDataset Shape:", df.shape)

# Display column names
print("\nColumns:")
print(df.columns.tolist())