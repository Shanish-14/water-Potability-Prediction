import pandas as pd

# Load dataset
df = pd.read_csv("data/water_potability.csv")

# 1. Dataset information
print("\n========== DATASET INFORMATION ==========")
print(df.info())

# 2. Dataset shape
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 4. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 5. Duplicate records
print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

# 6. Target distribution
print("\n========== POTABILITY DISTRIBUTION ==========")
print(df["Potability"].value_counts())