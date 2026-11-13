import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/water_potability.csv")

# Set visualization style
sns.set_theme(style="whitegrid")

# 1. Potability distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Potability")
plt.title("Water Potability Distribution")
plt.xlabel("Potability (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Histograms of all features
df.hist(figsize=(14, 10), bins=30)
plt.suptitle("Distribution of Water Quality Features")
plt.tight_layout()
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()