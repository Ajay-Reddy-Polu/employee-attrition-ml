import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("Employee Attrition Prediction Dashboard")

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

# ===============================
# 1️⃣ Full Dataset Prediction
# ===============================

st.subheader("Dataset Prediction")

df = pd.read_csv("data/Palo Alto Networks.csv")

X = df.drop("Attrition", axis=1)
X = pd.get_dummies(X, drop_first=True)

# Add missing columns
for col in model_columns:
    if col not in X.columns:
        X[col] = 0

X = X[model_columns]

df["Attrition_Probability"] = model.predict_proba(X)[:, 1]

# Create Risk Level
def get_risk(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Level"] = df["Attrition_Probability"].apply(get_risk)

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))
col2.metric("High Risk Employees", (df["Risk_Level"] == "High Risk").sum())
col3.metric("Low Risk Employees", (df["Risk_Level"] == "Low Risk").sum())

# Show High Risk Employees Only
st.subheader("High Risk Employees")

high_risk_df = df[df["Risk_Level"] == "High Risk"]
st.dataframe(high_risk_df)

# ===============================
# 2️⃣ Manual Employee Entry
# ===============================

st.subheader("Add New Employee")

with st.form("employee_form"):
    col1, col2 = st.columns(2)

    age = col1.number_input("Age", 18, 60, 30)
    income = col2.number_input("Monthly Income", 1000, 20000, 5000)
    job_sat = col1.slider("Job Satisfaction", 1, 4, 3)
    overtime = col2.selectbox("OverTime", ["Yes", "No"])
    gender = col1.selectbox("Gender", ["Male", "Female"])

    submit = st.form_submit_button("Predict")

if submit:

    input_dict = {
        "Age": age,
        "MonthlyIncome": income,
        "JobSatisfaction": job_sat,
        "OverTime_Yes": 1 if overtime == "Yes" else 0,
        "Gender_Male": 1 if gender == "Male" else 0
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    probability = model.predict_proba(input_df)[0][1]

    st.metric("Attrition Probability", f"{probability*100:.2f}%")

    if probability > 0.7:
        st.error("High Risk Employee ⚠")
    elif probability > 0.4:
        st.warning("Medium Risk Employee")
    else:
        st.success("Low Risk Employee ✅")
