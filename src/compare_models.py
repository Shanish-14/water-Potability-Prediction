
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("data/water_potability.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# Same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load both models
baseline_package = joblib.load(
    "models/water_potability_model.pkl"
)

baseline_model = baseline_package["model"]
baseline_imputer = baseline_package["imputer"]

tuned_model = joblib.load(
    "models/tuned_water_potability_model.pkl"
)

# Prepare test data
X_test_imputed = baseline_imputer.transform(X_test)

# Predictions
baseline_pred = baseline_model.predict(X_test_imputed)
tuned_pred = tuned_model.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0)
    }

# Compare results
results = pd.DataFrame([
    evaluate_model("Baseline Random Forest", y_test, baseline_pred),
    evaluate_model("Tuned Random Forest", y_test, tuned_pred)
])

print("\n========== MODEL COMPARISON ==========")
print(results.to_string(index=False))

# Best model based on F1 score
best_model_name = results.loc[
    results["F1 Score"].idxmax(), "Model"
]

print("\nBest model based on F1 Score:", best_model_name)
