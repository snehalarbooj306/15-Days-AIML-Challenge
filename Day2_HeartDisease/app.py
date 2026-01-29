import streamlit as st
import numpy as np
import joblib
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Load or Train Model
# -----------------------------
@st.cache_resource
def load_or_train():
    if os.path.exists("heart_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("heart_model.pkl")
        scaler = joblib.load("scaler.pkl")
    else:
        data = fetch_openml(name="heart-disease", version=1, as_frame=True)
        df = data.frame

        X = df.drop("target", axis=1)
        y = df["target"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        joblib.dump(model, "heart_model.pkl")
        joblib.dump(scaler, "scaler.pkl")

    return model, scaler


model, scaler = load_or_train()

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(page_title="Heart Disease Risk Checker", layout="centered")

# -----------------------------
# Simple Medical UI Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}

h1, h2, h3, p, label {
    color: #1f2933 !important;
    font-family: Arial, sans-serif;
}

input, select {
    border-radius: 6px !important;
    padding: 6px !important;
}

button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Risk Checker")
st.write("Enter basic health details to estimate heart disease risk.")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🧾 Patient Information")

age = st.number_input("Age", 20, 100, 45)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

restecg = st.selectbox("Rest ECG Result (0–2)", [0, 1, 2])

thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])

thal = st.selectbox("Thalassemia (1–3)", [1, 2, 3])

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🩺 Risk Assessment")

if st.button("Check Heart Disease Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1] * 100

    if probability < 30:
        risk = "Low Risk"
        color = "green"
    elif probability < 70:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    st.markdown(
        f"### 🧠 Risk Level: <span style='color:{color}'>{risk}</span>",
        unsafe_allow_html=True
    )

    st.markdown(f"### 📊 Probability of Heart Disease: **{probability:.2f}%**")

    st.info("⚠️ This tool is for learning purposes only and not a medical diagnosis.")
