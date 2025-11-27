import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# GLOBAL CONFIG & STYLE
# ========================================
st.set_page_config(layout="wide")
sns.set_style("whitegrid")

st.markdown("""
<style>
section.main > div {padding-top: 1rem;}
h1, h2, h3 {font-family: 'Georgia', serif;}
.insight-box {
    background-color: #f5f5f5;
    padding: 12px 16px;
    border-left: 4px solid #3c7edb;
    border-radius: 4px;
    margin-bottom: 10px;
}
.section-title {
    font-size: 26px;
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
</style>
""", unsafe_allow_html=True)

st.title("📊 Data Overview")

# ========================================
# DATA IMPORT & CLEANING
# ========================================
df = pd.read_csv("car_ad_display.csv", encoding="ISO-8859-1", sep=";").drop(columns='Unnamed: 0')
df = df.dropna()

df_raw = df.copy()   

def shorten_categories(categories, cutoff):
    mapping = {}
    for i in range(len(categories)):
        mapping[categories.index[i]] = (
            categories.index[i] if categories.values[i] >= cutoff else "Other"
        )
    return mapping

df['car'] = df['car'].map(shorten_categories(df['car'].value_counts(), 10))
df['model'] = df['model'].map(shorten_categories(df['model'].value_counts(), 10))

initial_rows = df.shape[0]

# Filters 
df = df[(df["price"] <= 100000) & (df["price"] >= 1000)]
df = df[(df["mileage"] <= 600) & (df["engV"] <= 7.5)]
df = df[df["year"] >= 1975]

cleaned_rows = df.shape[0]

# ========================================
# PAGE TABS
# ========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Dataset Overview",
    "📊 Numeric Distributions",
    "📦 Categorical Distributions",
    "💰 Top Expensive Cars",
    "🧹 Cleaning Summary"
])

# ========================================
# TAB 1 - DATASET OVERVIEW
# ========================================
with tab1:
    st.markdown("<div class='section-title'>1. Dataset Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader-q'>What does the dataset look like after basic cleaning?</div>", unsafe_allow_html=True)

    st.dataframe(df.head(), height=180)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", cleaned_rows)
    c2.metric("Columns", df.shape[1])
    c3.metric("Unique Brands", df['car'].nunique())
    c4.metric("Unique Models", df['model'].nunique())

    st.markdown("<div class='subheader-q'>Which car brands appear most frequently?</div>", unsafe_allow_html=True)
    
    brand_table = df['car'].value_counts().reset_index()
    brand_table.columns = ['Brand', 'Frequency']
    st.dataframe(brand_table, height=210)

    st.markdown("""
    <div class='subheader-q'>💡 Insights</div>
    <div class='insight-box'>
    • Brand presence is dominated by Volkswagen, Mercedes, BMW and Toyota.<br>
    • Grouping rare brands into <b>'Other'</b> avoids sparsity issues.<br>
    • The dataset shows solid diversity for price modeling tasks.
    </div>
    """, unsafe_allow_html=True)

# ========================================
# TAB 2 - NUMERIC DISTRIBUTIONS
# ========================================
with tab2:
    st.markdown("<div class='section-title'>2. Numeric Feature Distributions</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader-q'>How are price, mileage and engine characteristics distributed?</div>", unsafe_allow_html=True)

    numeric_cols = ["price", "mileage", "engV", "year"]

    for i in range(0, len(numeric_cols), 2):
        cols = st.columns(2)
        for col, feature in zip(cols, numeric_cols[i:i+2]):
            with col:
                fig, ax = plt.subplots(figsize=(4.2, 2.7))
                sns.histplot(df[feature], kde=True, bins=35, ax=ax)
                ax.set_title(feature.capitalize(), fontsize=11)
                st.pyplot(fig)

    st.markdown("""
    <div class='subheader-q'>💡 Insights</div>
    <div class='insight-box'>
    • <b>Price</b> shows strong right skewness.<br>
    • <b>Mileage</b> reflects the typical second-hand market spread.<br>
    • <b>Engine displacement</b> concentrates around common sizes (1.6–2.0 L).<br>
    • <b>Manufacturing years</b> mostly range from 1995-2015.
    </div>
    """, unsafe_allow_html=True)

# ========================================
# TAB 3 - CATEGORICAL DISTRIBUTIONS
# ========================================
with tab3:
    st.markdown("<div class='section-title'>3. Categorical Feature Distributions</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader-q'>How are body types, engine types and drivetrains distributed?</div>", unsafe_allow_html=True)

    cat_cols = ["body", "engType", "drive", "registration"]

    for i in range(0, len(cat_cols), 2):
        cols = st.columns(2)
        for col, cat in zip(cols, cat_cols[i:i+2]):
            with col:
                fig, ax = plt.subplots(figsize=(4.2, 2.4))
                sns.countplot(x=cat, data=df, ax=ax)
                ax.set_title(cat.capitalize(), fontsize=11)
                plt.xticks(rotation=35)
                st.pyplot(fig)

    st.markdown("""
    <div class='subheader-q'>💡 Insights</div>
    <div class='insight-box'>
    • Sedans and Crossovers dominate body type distribution.<br>
    • Petrol remains the most frequent fuel/engine type.<br>
    • Front-wheel drive is the most common drivetrain.<br>
    • The majority of cars are registered.
    </div>
    """, unsafe_allow_html=True)
# --------------------------------------------------------
# TAB 4 - TOP EXPENSIVE CARS 
# --------------------------------------------------------
with tab4:
    st.header("💰 Top 10 Most Expensive Cars")

    
    df_top10 = df_raw.copy().dropna()

    top_10 = (
        df_top10.sort_values(by="price", ascending=False)
        .drop_duplicates(subset=["car", "model"])
        .head(10)
    )

    top_10["label"] = top_10["car"] + " " + top_10["model"]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=top_10, x="label", y="price", palette="viridis", ax=ax)
    plt.xticks(rotation=35, ha="right")
    ax.set_title("Top 10 Most Expensive Cars (Notebook Identical)", fontsize=14)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)

    st.subheader("Detailed Prices of Top 10 Cars")
    st.dataframe(top_10, height=320)
    
    st.markdown("""
<div class='subheader-q'>💡 Insights</div>
<div class='insight-box'>
• Bentley models dominate the absolute high-class market.<br>
• Premium Mercedes-Benz and Land Rover variants consistently rank among the top priced vehicles.<br>
• A significant price gap exists between the top Bentley models and the rest.
</div>
""", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SECOND PLOT - TOP 10 BY BRAND MEAN PRICE 
    # ------------------------------------------------------------
    st.subheader("💵 Top 10 Most Expensive Brands (Mean Price)")

    mean_price = df.groupby("car")["price"].mean().sort_values(ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    sns.barplot(x=mean_price.index, y=mean_price.values, palette="magma", ax=ax2)
    plt.xticks(rotation=35)
    ax2.set_title("Top 10 Brands by Average Price", fontsize=12)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig2)

    st.write(mean_price)

    st.markdown("""
<div class='subheader-q'>💡 Insights</div>
<div class='insight-box'>
• Bentley shows the highest mean vehicle price by a wide margin.<br>
• Porsche, Land Rover and Infiniti maintain consistently elevated average prices across their models.<br>
• More diversified brands such as Toyota or Mercedes-Benz appear lower in the ranking due to their diverse model ranges.
</div>
""", unsafe_allow_html=True)

# ========================================
# TAB 5 - CLEANING SUMMARY
# ========================================
with tab5:
    st.markdown("<div class='section-title'>5. Cleaning Summary</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader-q'>How much data was removed, and why?</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Initial Rows", initial_rows)
    c2.metric("Final Rows", cleaned_rows)
    c3.metric("Removed Rows", initial_rows - cleaned_rows)

    
    col_center = st.columns([2, 1, 2])[1]

    with col_center:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ax.pie(
            [cleaned_rows, initial_rows - cleaned_rows],
            labels=["Kept", "Removed"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"],
            textprops={'fontsize': 20}
        )
        ax.set_title("", fontsize=1)
        plt.tight_layout(pad=0)
        st.pyplot(fig)

    st.markdown("""
    <div class='subheader-q'>💡 Insights</div>
    <div class='insight-box'>
    • Only a small proportion of rows required elimination.<br>
    • Most removals correspond to extreme outliers in price, engine capacity or mileage.<br>
    • The resulting dataset is clean and suitable for statistical modeling.
    </div>
    """, unsafe_allow_html=True)
