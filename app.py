# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Data & Models
# ------------------------------
@st.cache_data

def load_data():
    return pd.read_csv("data/attrition_cleaned.csv")
def load_raw():
    return pd.read_csv("data/Emp_A.csv") 

@st.cache_resource
def load_models():
    try:
        attrition_model = joblib.load("model/attrition_model.pkl")
    except:
        attrition_model = None
    try:
        performance_model = joblib.load("model/performance_model.pkl")
    except:
        performance_model = None
    return attrition_model, performance_model

df = load_data()
attrition_model, performance_model = load_models()
df1 = load_raw()
# ------------------------------
# Streamlit Layout
# ------------------------------
st.set_page_config(page_title="Employee Attrition & Performance Dashboard", layout="wide")
st.title("📊 Employee Attrition & Performance Prediction Dashboard")
st.markdown("This dashboard provides **actionable HR insights** on attrition risk, performance ratings, and workforce trends.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Attrition Prediction", "Performance Prediction", "HR Insights & Reports"])

# ------------------------------
# Overview Page
# ------------------------------
if page == "Overview":
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.metric("Total Employees", len(df))
    predicted_attrition_rate = df["Attrition"].value_counts(normalize=True).get(1, 0)
    st.metric("Predicted Attrition Rate:", round(predicted_attrition_rate*100, 2), "%")

    # print("Predicted Attrition Rate:", round(predicted_attrition_rate*100, 2), "%")
    #    predicted_attrition_rate = df["Attrition"].value_counts(normalize=True).get("Yes", 0)


    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(data=df1, x="Attrition", hue='Department', ax = ax)
    plt.legend()
    ax.set_title("Attrition Count by Department")
    st.pyplot(fig)

# ------------------------------
# Attrition Prediction
# ------------------------------
elif page == "Attrition Prediction":
    st.subheader("🔮 Predict Employee Attrition Risk")

    if attrition_model is None:
        st.error("Attrition model not found! Please upload attrition_model.pkl.")
    else:
        # Use only the columns present during model training
        features = df.loc[:, attrition_model.feature_names_in_]
        preds = attrition_model.predict(features)
        probs = attrition_model.predict_proba(features)[:, 1]

        df1["Attrition_Prediction"] = preds
        df1["Attrition_Probability"] = probs  # ✅ add this back

        st.write("### At-Risk Employees (Top 10)")
        risky = df1.sort_values("Attrition_Probability", ascending=False).head(10)
        st.dataframe(risky[["EmployeeNumber", "JobRole", "Department", 
                            "YearsAtCompany", "Attrition_Prediction", 
                            "Attrition_Probability"]])

        fig, ax = plt.subplots()
        sns.histplot(df1["Attrition_Probability"], bins=20, kde=True, ax=ax)
        ax.set_title("Distribution of Attrition Risk Probability")
        st.pyplot(fig)

# ------------------------------
# Performance Prediction
# ------------------------------
elif page == "Performance Prediction":
    st.subheader("📈 Predict Employee Performance Rating")

    if performance_model is None:
        st.error("Performance model not found! Please upload performance_model.pkl.")
    else:
        features = df.loc[:, performance_model.feature_names_in_]
        preds = performance_model.predict(features)
        df["Performance_Prediction"] = preds
        
        st.write("### Sample Predictions")
        # st.write("Columns in dataframe:", df.columns.tolist())

        cols_to_show = [col for col in ["EmployeeNumber", "YearsAtCompany", "Performance_Prediction"] if col in df.columns]
        st.dataframe(df[cols_to_show].head(10))


        fig, ax = plt.subplots()
        sns.countplot(x=df["Performance_Prediction"], hue= None,palette="viridis", ax=ax)
        ax.set_title("Distribution of Predicted Performance Ratings")
        st.pyplot(fig)

# ------------------------------
# HR Insights & Reports
# ------------------------------
elif page == "HR Insights & Reports":
    st.subheader("📌 Actionable Insights for HR")

    st.markdown("""
    - **At-Risk Employees** → Identify employees likely to leave and intervene with retention strategies.  
    - **Performance Risk** → Flag employees with consistently low predicted performance for training/upskilling.  
    - **Department Trends** → Monitor attrition & performance trends across departments.  
    - **Business Impact** → Estimate potential savings by reducing attrition.  
    """)

    # Business impact
    attrition_rate = (df["Attrition"].value_counts(normalize=True)[1]) * 100
    cost_per_employee = 5000  # Example cost for turnover
    potential_savings = len(df) * (attrition_rate / 100) * cost_per_employee

    st.metric("Current Attrition Rate", f"{attrition_rate:.2f}%")
    st.metric("Estimated Cost of Attrition", f"${potential_savings:,.0f}")

    # Department-wise attrition
    fig, ax = plt.subplots()
    dept_attrition = df1.groupby("Department")["Attrition"].value_counts(normalize=True).unstack().fillna(0)
    dept_attrition["Yes"].plot(kind="bar", ax=ax, color="salmon")
    ax.set_ylabel("Attrition Rate")
    ax.set_title("Attrition Rate by Department")
    st.pyplot(fig)

    # Download report
    st.download_button(
        label="📥 Download HR Report (CSV)",
        data=df.to_csv(index=False),
        file_name="hr_report.csv",
        mime="text/csv"
    )
