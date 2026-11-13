import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("data/water_potability.csv")

# Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Handle missing values
imputer = SimpleImputer(strategy="median")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Display results
print("========== PREPROCESSING RESULTS ==========")

print("Original dataset:", df.shape)
print("Training data:", X_train_imputed.shape)
print("Testing data:", X_test_imputed.shape)

print("\nMissing values in training data:")
print(X_train_imputed.isnull().sum())

print("\nMissing values in testing data:")
print(X_test_imputed.isnull().sum())