import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("diabetes_model.pkl")

# App title
st.title("🩺 Diabetes Prediction App")
st.write("Fill the form to check diabetes risk.")

# Name input
name = st.text_input("Enter Patient Name")

# Gender input
gender = st.radio("Select Gender", ("Male", "Female"))
gender_val = 1 if gender == "Male" else 0

# Other inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age, gender_val]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error(f"⚠️ {name} is likely to have diabetes.")
    else:
        st.success(f"✅ {name} is not likely to have diabetes.")
