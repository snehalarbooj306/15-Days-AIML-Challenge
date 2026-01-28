import streamlit as st
import joblib
import numpy as np
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# -------------------------------
# Load or Train Model
# -------------------------------
@st.cache_resource
def load_or_train_model():
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
    else:
        data = fetch_california_housing()
        X = data.data
        y = data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        joblib.dump(model, "model.pkl")

    return model


model = load_or_train_model()

# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

# -------------------------------
# Pink UI Styling
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe6f0;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: black !important;
    }

    input {
        background-color: #1f1f2e !important;
        color: white !important;
        border-radius: 10px !important;
        font-size: 18px !important;
    }

    button {
        background-color: #ff69b4 !important;
        color: white !important;
        border-radius: 12px !important;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# App UI
# -------------------------------
st.title("🏡 House Price Predictor 💗")
st.write("Fill the details and click Predict")

MedInc = st.number_input("Median Income", 0.0, 15.0, 5.0)
HouseAge = st.number_input("House Age", 1.0, 50.0, 20.0)
AveRooms = st.number_input("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.number_input("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.number_input("Population", 1.0, 5000.0, 1000.0)
AveOccup = st.number_input("Average Occupancy", 1.0, 5.0, 3.0)
Latitude = st.number_input("Latitude", 32.0, 42.0, 36.0)
Longitude = st.number_input("Longitude", -125.0, -114.0, -120.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("💖 Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                             Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(input_data)

    usd_price = prediction[0] * 100000
    inr_price = usd_price * 83

    st.success(f"🏠 Predicted House Price: ₹ {inr_price:,.0f}")
