import streamlit as st
import pandas as pd

# Page Config

st.set_page_config(
    page_title="FitPulse",
    page_icon="💪",
    layout="wide"
)


#UI Styling

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.hero-container {
    text-align: center;
    padding: 40px;
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 30px;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #d1d1d1;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    text-align: center;
}

div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
    transition: 0.3s ease;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    transform: scale(1.05);
}

section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# Hero Section

st.markdown("""
<div class="hero-container">
    <div class="hero-title">💪 FitPulse</div>
    <div class="hero-subtitle">
        AI-Powered Health Anomaly Detection from Fitness Devices
    </div>
</div>
""", unsafe_allow_html=True)

st.title("💪 FitPulse – Health Anomaly Detection Dashboard")
st.markdown(
    "<h4 style='text-align: center; color: white;'>Upload fitness device data, clean it, and analyze null values instantly.</h4>",
    unsafe_allow_html=True
)

st.divider()


# File Upload

uploaded_file = st.file_uploader("📂 Upload Fitness CSV Data", type=["csv"])

if uploaded_file is not None:

    # Store original and working dataframe
    if "original_df" not in st.session_state:
        st.session_state["original_df"] = pd.read_csv(uploaded_file)
        st.session_state["df"] = st.session_state["original_df"].copy()

    df = st.session_state["df"]
    original_df = st.session_state["original_df"]

    st.success("File Uploaded Successfully ✅")

    # Metrics
  
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Rows", df.shape[0])
    col2.metric("📂 Total Columns", df.shape[1])
    col3.metric("⚠ Total Null Values", df.isnull().sum().sum())

    st.divider()

    # Buttons

    b1, b2, b3 = st.columns(3)

    # CLEAN DATA
    if b1.button("🧹 Clean Data"):
        with st.spinner("Cleaning Data... Please Wait ⏳"):

            df = original_df.copy()
            preprocessing_steps = []

            # Datetime handling
            for col in df.columns:
                if "date" in col.lower():
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[col] = df[col].ffill().bfill()
                    preprocessing_steps.append(f"{col} converted to datetime & missing filled")

            # Numeric handling
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            preprocessing_steps.append("Numeric columns filled using Forward/Backward Fill")

            # Object handling
            object_cols = df.select_dtypes(include=["object"]).columns
            df[object_cols] = df[object_cols].fillna("No Workout")
            preprocessing_steps.append("Categorical columns filled with 'No Workout'")

            st.session_state["df"] = df
            st.session_state["preprocessing"] = preprocessing_steps

        st.success("Data Cleaned Successfully ✅")

    # SHOW DATA
    if b2.button("👀 Show Data"):
        st.subheader("📋 Data Preview")
        st.dataframe(st.session_state["df"], use_container_width=True)

    # CHECK NULL VALUES
    if b3.button("🔍 Check Null Values"):

        cleaned_df = st.session_state["df"]

        st.subheader("📊 Null Values Before Cleaning")

        null_before = original_df.isnull().sum()
        percent_before = (null_before / len(original_df)) * 100

        df_before = pd.DataFrame({
            "Null Count": null_before,
            "Null Percentage (%)": percent_before.round(2)
        })

        st.dataframe(df_before)
        st.bar_chart(null_before)

        st.divider()

        st.subheader("📊 Null Values After Cleaning")

        null_after = cleaned_df.isnull().sum()
        percent_after = (null_after / len(cleaned_df)) * 100

        df_after = pd.DataFrame({
            "Null Count": null_after,
            "Null Percentage (%)": percent_after.round(2)
        })

        st.dataframe(df_after)
        st.bar_chart(null_after)

        if null_after.sum() == 0:
            st.success("All Null Values Removed Successfully ✅")
        else:
            st.warning("Some null values still exist ⚠️")

    
    # Show Preprocessing Steps
  
    if "preprocessing" in st.session_state:
        st.divider()
        st.subheader("⚙️ Pre-Processing Steps Applied")

        for step in st.session_state["preprocessing"]:
            st.write("✔", step)