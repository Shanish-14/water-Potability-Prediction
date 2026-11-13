
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model
model_package = joblib.load("models/water_potability_model.pkl")
model = model_package["model"]
features = model_package["features"]

# Get feature importance
importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df)

# Visualize feature importance
plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Water Quality Features")
plt.tight_layout()
plt.show()
