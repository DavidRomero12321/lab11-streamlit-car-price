
import streamlit as st
import pandas as pd
import pickle


# ===== PRICE PREDICTOR HEADER =====
st.markdown("""
<h1 style='font-size:36px; margin-bottom:0;'>💰 Price Predictor</h1>
<p style='font-size:17px; color:#444; margin-top:0;'>
Estimate the expected market price of a used car based on its characteristics.
</p>
<hr style='margin-top:5px; margin-bottom:15px;'>
""", unsafe_allow_html=True)

# ===== SHORT INSTRUCTIONS =====
st.markdown("""
<div style='background-color:#F3F4F6; padding:15px; border-radius:8px; border-left:4px solid #3c7edb;'>
<b>How it works:</b><br>
• Select the main features of the car you want to evaluate 🚗<br>
• Adjust mileage, engine volume, and year using the sliders<br>
• The model will estimate a realistic market price based on historical data 📈<br><br>
</div>
<br>
""", unsafe_allow_html=True)
# Load model + encoders
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

# Load dataset for options
df_original = pd.read_csv("car_ad_display.csv", encoding="ISO-8859-1", sep=";").drop(columns='Unnamed: 0')

st.subheader("Input Car Features")

brand = st.selectbox("Car Brand", df_original['car'].unique())
body = st.selectbox("Body Type", df_original['body'].unique())
mileage = st.slider("Mileage", min_value=0, max_value=600, value=100)
engV = st.slider("Engine Volume", min_value=0.5, max_value=7.5, value=2.0)
engType = st.selectbox("Engine Type", df_original['engType'].unique())
reg = st.selectbox("Registered?", ["yes", "no"])
year = st.slider("Car Year", min_value=1975, max_value=2023, value=2010)
drive = st.selectbox("Drive Type", df_original['drive'].unique())

if st.button("Predict Price"):
    try:
        X_sample = [[
            le_car.transform([brand])[0],
            le_body.transform([body])[0],
            mileage,
            engV,
            le_engType.transform([engType])[0],
            1 if reg == "yes" else 0,
            year,
            le_drive.transform([drive])[0]
        ]]

        pred = model.predict(X_sample)[0]
        st.success(f"Estimated Price: **${pred:,.2f}**")
    except:
        st.error("This car configuration includes unseen labels not present during training.")