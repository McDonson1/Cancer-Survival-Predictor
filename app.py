import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# Load the trained model relative to this file so the app works on Streamlit Cloud.
MODEL_PATH = Path(__file__).resolve().parent / "GradientBoosting.pkl"


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(page_title="Cancer Survival Prediction", page_icon="🩺", layout="centered")
st.title("Cancer Survival Prediction App")
st.caption("Educational demonstration only — not a substitute for professional medical advice or diagnosis.")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "The model file 'GradientBoosting.pkl' was not found. "
        "Ensure it is included in the repository alongside app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()


# Country list
countries = [
    "UK", "Japan", "France", "USA", "China", "South Korea", "Brazil",
    "Germany", "Canada", "Pakistan", "Italy", "New Zealand",
    "South Africa", "India", "Nigeria", "Australia",
]

# Patient inputs
alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
cancer_stage = st.selectbox("Cancer Stage", ["Localized", "Regional", "Metastatic"])
country = st.selectbox("Country", countries)
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
diabetes_history = st.selectbox("Diabetes History", ["Yes", "No"])
diet_risk = st.selectbox("Diet Risk", ["Low", "Moderate", "High"])
early_detection = st.selectbox("Early Detection", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
mutation = st.selectbox("Genetic Mutation", ["Yes", "No"])
healthcare_access = st.selectbox("Healthcare Access", ["Low", "Moderate", "High"])
healthcare_costs = st.number_input("Healthcare Costs (USD)", min_value=0.0)
heart_disease = st.selectbox("Heart Disease History", ["Yes", "No"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
incidence_rate = st.number_input("Incidence Rate per 100K", min_value=0.0)
ibd = st.selectbox("Inflammatory Bowel Disease", ["Yes", "No"])
insurance_cost = st.selectbox("Insurance Costs", ["No insurance", "Basic", "Extended"])
insurance_status = st.selectbox("Insurance Status", ["Insured", "Uninsured"])
mortality_rate = st.number_input("Mortality Rate per 100K", min_value=0.0)
non_smoker = st.selectbox("Non-Smoker", ["Yes", "No"])
obesity_bmi = st.selectbox("Obesity BMI", ["Normal", "Overweight", "Obese"])
physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
screening_history = st.selectbox("Screening History", ["Never", "Irregular", "Regular"])
smoking_history = st.selectbox("Smoking History", ["Yes", "No"])
treatment_type = st.selectbox(
    "Treatment Type", ["Chemotherapy", "Radiation", "Surgery", "Combined"]
)
tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0)
urban_rural = st.selectbox("Urban or Rural", ["Urban", "Rural"])
age = st.number_input("Age", min_value=0)

# Keep the feature names and order identical to the model's training data.
columns = [
    "Alcohol Consumption", "Cancer Stage", "Country", "Diabetes",
    "Diabetes History", "Diet Risk", "Early Detection", "Family History",
    "Gender", "Genetic Mutation", "Healthcare Access", "Healthcare Costs",
    "Heart Disease History", "Hypertension", "Incidence Rate per 100K",
    "Inflammatory Bowel Disease", "Insurance Costs", "Insurance Status",
    "Mortality Rate per 100K", "Non Smoker", "Obesity BMI",
    "Physical Activity", "Screening History", "Smoking History",
    "Treatment Type", "Tumor Size (mm)", "Urban or Rural", "Age",
]

input_data = pd.DataFrame(
    [[
        alcohol, cancer_stage, country, diabetes, diabetes_history, diet_risk,
        early_detection, family_history, gender, mutation, healthcare_access,
        healthcare_costs, heart_disease, hypertension, incidence_rate, ibd,
        insurance_cost, insurance_status, mortality_rate, non_smoker,
        obesity_bmi, physical_activity, screening_history, smoking_history,
        treatment_type, tumor_size, urban_rural, age,
    ]],
    columns=columns,
)

if st.button("Predict Survival", type="primary"):
    try:
        prediction = model.predict(input_data)
        result = "Yes" if prediction[0] == 1 else "No"
        st.success(f"Prediction: {result}")
        st.caption(
            "This is a model-generated estimate for educational purposes, "
            "not a clinical assessment."
        )
    except Exception as exc:
        st.error(f"Prediction failed. Please check the input features and model: {exc}")
