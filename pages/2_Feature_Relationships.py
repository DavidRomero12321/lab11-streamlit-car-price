import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ========================================
# GLOBAL PAGE STYLE 
# ========================================
st.set_page_config(layout="wide")
sns.set_style("whitegrid")

st.markdown("""
<style>
h1, h2, h3 {font-family: 'Georgia', serif;}
.section-title {
    font-size: 28px;
    font-weight: 600;
    padding-top: 10px;
    border-bottom: 2px solid #d9d9d9;
    margin-bottom: 15px;
}
.subheader-q {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin-top: 10px;
}
.insight-box {
    background-color: #f5f5f5;
    padding: 12px 16px;
    border-left: 4px solid #3c7edb;
    border-radius: 4px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Feature Relationships with Price")

# ========================================
# DATA PIPELINE 
# ========================================

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

car_map = shorten_categories(df.car.value_counts(), 10)
df['car'] = df['car'].map(car_map)

model_map = shorten_categories(df.model.value_counts(), 10)
df['model'] = df['model'].map(model_map)

df = df[df["price"] <= 100000]
df = df[df["price"] >= 1000]
df = df[df["mileage"] <= 600]
df = df[df["engV"] <= 7.5]
df = df[df["year"] >= 1975]

# ========================================
# TABS
# ========================================
tab_num, tab_cat, tab_corr = st.tabs([
    "📉 Numeric vs Price",
    "📦 Categorical vs Price",
    "🧮 Correlation Matrix"
])

# ========================================
# TAB 1: Numeric Relationships
# ========================================
with tab_num:
    st.markdown("<div class='section-title'>1. Numeric Features vs Price</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader-q'>How do continuous variables influence vehicle price?</div>", unsafe_allow_html=True)

    numeric_features = ["mileage", "engV", "year"]
    cols = st.columns(3)

    for col, feat in zip(cols, numeric_features):
        with col:
            fig, ax = plt.subplots(figsize=(4.3, 3))
            sns.scatterplot(data=df, x=feat, y="price", alpha=0.25, ax=ax)
            ax.set_title(f"{feat.capitalize()} vs Price", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("""
    <div class='subheader-q'>💡 Insights</div>
    <div class='insight-box'>
    • <b>Year</b> shows a strong positive effect: newer cars reliably cost more.<br>
    • <b>Mileage</b> displays a clear negative trend - fewer kilometers means higher market value.<br>
    • <b>Engine volume (engV)</b> increases price but in a noisier way (non-linear effects).<br>
    • These numeric variables are by far the strongest linear predictors of vehicle price.
    </div>
    """, unsafe_allow_html=True)

# ========================================
# TAB 2: Categorical Relationships
# ========================================
with tab_cat:
    st.markdown("<div class='section-title'>2. Categorical Features vs Price</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader-q'>How do body type, engine type and drivetrain affect prices?</div>", unsafe_allow_html=True)

    cat_features = ["body", "engType", "drive", "registration"]

    for i in range(0, len(cat_features), 2):
        cols = st.columns(2)
        for col, feat in zip(cols, cat_features[i:i+2]):
            with col:
                st.markdown(f"#### {feat.capitalize()}")
                fig, ax = plt.subplots(figsize=(4.8, 3))
                sns.boxplot(data=df, x=feat, y="price", ax=ax)
                plt.xticks(rotation=35)
                plt.tight_layout()
                st.pyplot(fig)

    st.markdown("""
<div class='subheader-q'>💡 Insights</div>
<div class='insight-box'>
• <b>Crossovers</b> are clearly the <b>most expensive</b> body type, while sedans, hatchbacks and vans stay in the <b>lower-price</b> range 🚙💰<br>
• <b>Petrol</b> and <b>diesel</b> engines show <b>large price spread</b>, meaning they include many premium models 🔧📈<br>
• <b>Full-wheel drive (AWD)</b> cars are <b>much more expensive</b> than <b>rear-wheel (RWD)</b> or <b>front-wheel (FWD)</b> vehicles <br>
• <b>Registered cars</b> cost <b>far more</b> than unregistered ones - expected in second-hand listings <br>
• Categorical features show <b>non-linear effects</b>, so simple correlations cannot capture their full impact 
</div>
""", unsafe_allow_html=True)



# ========================================
# TAB 3: Correlation Matrix
# ========================================
with tab_corr:
    st.markdown("<div class='section-title'>3. Correlation Matrix</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader-q'>Which variables are most linearly correlated with price?</div>", unsafe_allow_html=True)

    df_corr = df.copy()
    df_corr = df_corr.drop(columns=["model"])   

    for col in ["car", "body", "engType", "drive"]:
        le = LabelEncoder()
        df_corr[col] = le.fit_transform(df_corr[col])

    df_corr["registration"] = df_corr["registration"].map(
        {"yes": 1, "YES": 1, "Yes": 1, "y": 1, "Y": 1,
         "no": 0, "No": 0, "NO": 0}
    )

    corr = df_corr.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="icefire", annot=False, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=13)
    st.pyplot(fig)

    st.markdown("""
    <div class='subheader-q'>💡 Insights</div>
    <div class='insight-box'>
    • <b>Year</b> holds the strongest positive correlation with price.<br>
    • <b>Mileage</b> shows the strongest negative correlation.<br>
    • <b>engV</b> also correlates positively, though at a lower magnitude.<br>
    • Encoded categorical variables appear weak in linear correlation, but their impact is stronger in ML models (non-linear).<br>
    • Overall, numeric variables dominate linear predictive power.
    </div>
    """, unsafe_allow_html=True)
