import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🏥")

st.title("Medical Insurance Cost Prediction")
st.write("Predict medical insurance charges using a Linear Regression model.")

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("model.pkl was not found. Please run train_model.py first.")
    st.stop()

st.subheader("Enter Patient Information")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex_male": [1 if sex == "male" else 0],
    "smoker_yes": [1 if smoker == "yes" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0]
})

if st.button("Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Medical Insurance Charges: ${prediction:,.2f}")
