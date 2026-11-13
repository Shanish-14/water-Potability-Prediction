
import os
import joblib

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load the tuned model
final_model = joblib.load(
    "models/tuned_water_potability_model.pkl"
)

# Save it as our final model
joblib.dump(
    final_model,
    "models/final_water_potability_model.pkl"
)

print("Final model saved successfully!")
