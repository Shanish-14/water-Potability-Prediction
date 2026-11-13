
import pandas as pd
import joblib

# Load final model
model = joblib.load(
    "models/final_water_potability_model.pkl"
)

# Prediction function
def predict_water(sample):

    # Convert measurements into a DataFrame
    sample_df = pd.DataFrame(
        [sample],
        columns=model.feature_names_in_
    )

    # Make prediction
    prediction = model.predict(sample_df)[0]

    # Get prediction probability
    probabilities = model.predict_proba(sample_df)[0]

    potable_probability = probabilities[
        list(model.classes_).index(1)
    ]

    # Display result
    if prediction == 1:
        result = "Predicted Potable"
    else:
        result = "Predicted Not Potable"

    return {
        "Prediction": result,
        "Potable Probability": round(
            potable_probability * 100, 2
        )
    }


# Test using one sample from the dataset
df = pd.read_csv("data/water_potability.csv")


# Enter your own water-quality measurements
sample = {
    "ph": 7.2,
    "Hardness": 180.0,
    "Solids": 15000.0,
    "Chloramines": 7.0,
    "Sulfate": 350.0,
    "Conductivity": 400.0,
    "Organic_carbon": 12.0,
    "Trihalomethanes": 60.0,
    "Turbidity": 3.5
}

result = predict_water(sample)

print("\n========== MANUAL WATER PREDICTION ==========")
print(result)

