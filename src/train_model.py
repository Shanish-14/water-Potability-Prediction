import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

# 2. Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Handle missing values
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 5. Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# 6. Train model
rf_model.fit(X_train_imputed, y_train)

print("Model training completed successfully!")

# 7. Save model and imputer
model_package = {
    "model": rf_model,
    "imputer": imputer,
    "features": list(X.columns)
}

joblib.dump(model_package, "models/water_potability_model.pkl")

print("Model saved successfully!")