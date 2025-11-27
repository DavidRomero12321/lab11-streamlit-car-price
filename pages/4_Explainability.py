import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# =======================================
# PAGE CONFIG
# =======================================
st.set_page_config(layout="wide")
st.title("🔍 Model Explainability (SHAP)")

st.markdown("""
This page provides a complete explainability analysis of the final LightGBM model using **SHAP values**.

Understanding *why* the model predicts a certain price is essential for transparency and trust.
Below you will find:

- **Global explainability** → how features influence predictions across the entire dataset  
- **Local explainability** → a step-by-step explanation of one specific prediction  
- **Interaction effects** → how two features interact using dependence plots  
""")


# =======================================
# LOAD MODEL + ENCODERS
# =======================================
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]


# =======================================
# LOAD + CLEAN DATA
# =======================================
df = pd.read_csv("car_ad_display.csv", encoding="ISO-8859-1", sep=";").drop(columns='Unnamed: 0')
df = df.dropna()

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

df['car'] = df['car'].map(shorten_categories(df.car.value_counts(), 10))
df['model'] = df['model'].map(shorten_categories(df.model.value_counts(), 10))

df = df[(df["price"] <= 100000) & (df["price"] >= 1000)]
df = df[(df["mileage"] <= 600) & (df["engV"] <= 7.5)]
df = df[df["year"] >= 1975]

yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']
df['registration'] = np.where(df['registration'].isin(yes_l), 1, 0)

df = df.drop(columns='model')

df['car'] = le_car.transform(df['car'])
df['body'] = le_body.transform(df['body'])
df['engType'] = le_engType.transform(df['engType'])
df['drive'] = le_drive.transform(df['drive'])

X = df.drop(columns="price")


# =======================================
# COMPUTE SHAP VALUES
# =======================================

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)


# =======================================
# TABS
# =======================================
tab1, tab2, tab3 = st.tabs([
    "🌈 Global Explainability",
    "📌 Local Explainability",
    "📈 Feature Interaction (Dependence Plots)"
])


# =======================================
# TAB 1 — GLOBAL EXPLAINABILITY
# =======================================
with tab1:
    st.header("🌈 Global Feature Impact")

    colA, colB = st.columns([1.3, 1])

    with colA:
        st.subheader("SHAP Summary Plot")
        fig1 = plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig1)

    with colB:
        st.subheader("Mean Absolute SHAP Values")
        fig2 = plt.figure(figsize=(6, 4))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig2)

    st.markdown("""
    ### 🔍 **Insights**
    - **Year** is the strongest positive driver of price — newer cars are systematically valued higher.  
    - **Mileage** decreases predicted value sharply, matching real-world expectations.  
    - **Engine volume (engV)** has a strong positive effect at higher values.  
    - Other categorical features contribute less but have relevant effects in specific scenarios.  
    """)


# =======================================
# TAB 2 — LOCAL EXPLAINABILITY
# =======================================
with tab2:
    st.header("📌 Local Explainability for a Single Prediction")

    st.write("""
    Below we analyze **one specific car** from the dataset.
    These plots explain exactly *how* the model combines the feature effects to produce its prediction.
    """)

    idx = st.number_input("Select an index to explain:", min_value=0, max_value=len(X)-1, value=0)

    # -------- WATERFALL ----------
    st.subheader("📘 Waterfall Plot")
    figW = plt.figure(figsize=(7, 5))
    shap.plots.waterfall(shap_values[idx], show=False)
    st.pyplot(figW)


    # -------- FORCE PLOT ----------
    # Force plot
    st.subheader("🟩 Force Plot")

    figF = plt.figure(figsize=(9, 3))
    shap.force_plot(
    shap_values[idx].base_values,
    shap_values[idx].values,
    X.iloc[idx],
    matplotlib=True,
    show=False)
    st.pyplot(figF)


    # -------- DECISION PLOT ----------
    st.subheader("📙 Decision Plot")
    figD = plt.figure(figsize=(8, 4))
    shap.decision_plot(
        shap_values[idx].base_values,
        shap_values[idx].values,
        X.iloc[idx],
        show=False
    )
    st.pyplot(figD)


    st.markdown("""
    ### 🔍 **Insights**
    - Local explainability reveals individually how each feature increases or decreases the predicted price.  
    - The waterfall plot is ideal to see **feature-by-feature contributions**.  
    - The force plot shows whether the model is generally pushed upward or downward.  
    - The decision plot describes the model's reasoning **step-by-step**.  
    """)


# =======================================
# TAB 3 — DEPENDENCE PLOTS
# =======================================
with tab3:
    st.header("📈 Feature Interaction Effects")

    st.write("""
    These plots show how a feature’s SHAP value evolves depending on another feature’s value.
    """)

    feature = st.selectbox("Choose a feature:", ["mileage", "engV", "year"])
    interaction = st.selectbox("Interaction feature:", ["engV", "mileage", "year"])

    # Avoid same feature for both axes
    if feature == interaction:
        st.warning("Interaction feature must be different from the selected feature.")
        st.stop()

    # Draw plot AFTER choices are made
    if st.button("Generate Dependence Plot"):
        plt.clf()              # Clear previous SHAP fig
        plt.close('all')       # Close hidden matplotlib handles

        shap.dependence_plot(
            feature,
            shap_values.values,
            X,
            interaction_index=interaction,
            show=False
        )

        figDP = plt.gcf()  
        st.pyplot(figDP)

        st.markdown(f"### 🔍 **Insights**")
        st.markdown("""
        - SHAP dependence plots reveal how feature effects vary non-linearly.  
        - Higher **mileage** consistently reduces SHAP contribution and final price.  
        - Larger **engV** engines increase SHAP contribution, especially for low-mileage cars.  
        - Newer **year** values show strong positive impact, with SHAP climbing fast after 2010.  
        """)
    else:
        st.info("Select features and click the button to generate the dependence plot.")
