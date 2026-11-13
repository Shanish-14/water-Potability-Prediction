
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

# 2. Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3. Recreate the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Load saved model package
model_package = joblib.load("models/water_potability_model.pkl")

model = model_package["model"]
imputer = model_package["imputer"]

# 5. Handle missing values in test data
X_test_imputed = imputer.transform(X_test)

# 6. Make predictions
y_pred = model.predict(X_test_imputed)

# 7. Calculate evaluation metrics
print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

# 8. Classification report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Confusion matrix
print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))
