
import streamlit as st
import pandas as pd
import joblib

# Load final model
model = joblib.load(
    "models/final_water_potability_model.pkl"
)

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧"
)

st.title("💧 Water Potability Prediction")
st.write(
    "Enter water-quality measurements to get an experimental ML prediction."
)

st.subheader("Water Quality Measurements")

# Input fields
ph = st.number_input("pH", 0.0, 14.0, 7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=180.0)
solids = st.number_input("Solids", min_value=0.0, value=15000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=350.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=12.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=3.5)

if st.button("Predict Water Potability"):

    sample = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }])

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]
    potable_probability = probabilities[
        list(model.classes_).index(1)
    ]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Model Prediction: Potable")
    else:
        st.warning("Model Prediction: Not Potable")

    st.write(
        f"Estimated potable probability: {potable_probability * 100:.2f}%"
    )

st.info(
    "This application is for educational purposes only. "
    "A machine-learning prediction cannot certify drinking-water safety. "
    "Use accredited laboratory testing for actual safety decisions."
)
