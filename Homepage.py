import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="Car Price Prediction App", layout="wide")

# ======================
# INTRO SECTION 
# ======================
st.markdown("""
<div style="
    max-width: 900px; 
    margin:auto; 
    margin-top:10px; 
    margin-bottom:35px; 
    padding: 25px 30px;
    background-color: #f9fafc;
    border-radius: 12px;
    border-left: 6px solid #4c8bf5;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
">

<h1 style="text-align:center; font-size:48px; margin-bottom: 8px; color:#222;">
🚗 Car Price Prediction App
</h1>

<p style="font-size:18px; text-align:center; margin-top:0; color:#444;">
Welcome to the Car Price Prediction App!
Designed as a complete interactive environment to explore vehicle market data, analyze feature behaviour, predict car 
prices using a trained machine learning model, and understand why the model makes its predictions through 
explainability techniques such as SHAP.
</p>

<p style="font-size:18px; text-align:center; margin-top:12px; color:#444;">
Use the dashboards below to navigate through each part of the project.
</p>

</div>
""", unsafe_allow_html=True)

# ======================
# DASHBOARD BUTTONS
# ======================

st.write("")  

# 4 dashboard buttons in a 2x2 layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# ---- BUTTON 1 ----
with col1:
    st.markdown("""
    <div style="
        padding:20px; 
        background-color:white; 
        border-radius:12px; 
        border:1px solid #e6e6e6; 
        box-shadow:0 3px 8px rgba(0,0,0,0.05); 
        margin-bottom:15px;">
        <h3 style='margin-top:0;'>📁 Data Explorer</h3>
        <p>Explore the dataset structure, distributions, cleaning steps and descriptive insights.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Data Explorer"):
        st.switch_page("Pages/1_Data_Explorer.py")

# ---- BUTTON 2 ----
with col2:
    st.markdown("""
    <div style="
        padding:20px; 
        background-color:white; 
        border-radius:12px; 
        border:1px solid #e6e6e6; 
        box-shadow:0 3px 8px rgba(0,0,0,0.05); 
        margin-bottom:15px;">
        <h3 style='margin-top:0;'>📊 Feature Relationships</h3>
        <p>Understand how numeric and categorical features influence car prices.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Feature Relationships"):
        st.switch_page("Pages/2_Feature_Relationships.py")

# ---- BUTTON 3 ----
with col3:
    st.markdown("""
    <div style="
        padding:20px; 
        background-color:white; 
        border-radius:12px; 
        border:1px solid #e6e6e6; 
        box-shadow:0 3px 8px rgba(0,0,0,0.05); 
        margin-bottom:15px;">
        <h3 style='margin-top:0;'>💰 Price Predictor</h3>
        <p>Use our trained LightGBM model to estimate the price of a vehicle.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Price Predictor"):
        st.switch_page("Pages/3_Price_Predictor.py")

# ---- BUTTON 4 ----
with col4:
    st.markdown("""
    <div style="
        padding:20px; 
        background-color:white; 
        border-radius:12px; 
        border:1px solid #e6e6e6; 
        box-shadow:0 3px 8px rgba(0,0,0,0.05); 
        margin-bottom:15px;">
        <h3 style='margin-top:0;'>🔎 Explainability (SHAP)</h3>
        <p>Interpret model predictions and feature importance using SHAP values.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Explainability"):
        st.switch_page("Pages/4_Explainability.py")
