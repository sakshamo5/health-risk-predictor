import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names
feature_names = ["Age", "BMI", "Blood Pressure", "Cholesterol", "Glucose"]

# Define healthy ranges (approximate)
healthy_ranges = {
    "Age": "No specific range, but higher age can increase risk.",
    "BMI": "18.5 - 24.9 (Normal)",
    "Blood Pressure": "90-120 (Normal), 120-140 (Prehypertension), 140+ (High)",
    "Cholesterol": "Below 200 (Normal), 200-240 (Borderline), 240+ (High)",
    "Glucose": "70-99 (Normal), 100-125 (Prediabetes), 126+ (Diabetes)"
}

# Streamlit UI
st.title("\U0001F3E5 Health Risk Prediction App")
st.markdown("Enter medical values to predict the risk factor.")

# User inputs with info tooltips
input_data = []
for feature in feature_names:
    value = st.number_input(
        f"Enter {feature}:",
        value=0,  # Default value as an integer
        format="%d",  # Ensures integer input
        help=healthy_ranges[feature]  # Adds hover information
    )
    input_data.append(value)

# Predict on user input
if st.button("Predict"):
    X_input = np.array(input_data).reshape(1, -1)
    X_input_scaled = scaler.transform(X_input)

    # Make prediction
    prediction = model.predict(X_input_scaled)[0]

    # Categorize the prediction
    if prediction < 140:
        category = "🟢 No Risk"
        color = "green"
    elif 140 <= prediction < 180:
        category = "🟡 Low Risk"
        color = "green"
    elif 180 <= prediction < 200:
        category = "🟠 Intermediate Risk"
        color = "orange"
    else:
        category = "🔴 High Risk"
        color = "red"

    # Display results
    st.subheader("🔍 Prediction Result:")
    st.markdown(f"**Predicted Health Risk Score:** {int(prediction)}")  # Display as integer
    st.markdown(f"<h3 style='color:{color};'>{category}</h3>", unsafe_allow_html=True)
    
    st.markdown("### ℹ️ How to Interpret the Result:")
    st.write("- **Score < 140:** No Risk")
    st.write("- **140 ≤ Score < 180:** Low Risk")
    st.write("- **180 ≤ Score < 200:** Intermediate Risk")
    st.write("- **Score ≥ 200:** High Risk")