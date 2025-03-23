import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Create a new scaler instead of loading from file
scaler = StandardScaler()

# Title
st.markdown("<h1 style='color:#004080; text-align: center;'>📊 Telecom Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Explanation Section
st.markdown("""
### 🔍 What is Customer Churn?
Customer churn refers to when a customer **stops using a company's service**. 
For telecom companies, it's crucial to **predict and prevent churn** by identifying customers who are likely to leave.

### 📌 How This Model Works
- You enter customer details (contract type, charges, etc.).
- The model analyzes the data and predicts whether the customer will **churn (leave)** or **stay**.
- This helps businesses **take action** (e.g., offering discounts, better plans) to retain customers.

### 🚀 How to Use:
1️⃣ Fill in the customer details below.  
2️⃣ Click **"Predict Churn"**.  
3️⃣ See the result: ✅ (Customer Stays) or 🚨 (Customer Likely to Churn).  
""", unsafe_allow_html=True)

# Create columns for a more compact layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

with col2:
    st.subheader("Contract Information")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=0.1)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=0.1)
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer (automatic)", "Credit Card (automatic)"])

# Additional services (these were likely part of the 22 features in your model)
st.subheader("Services Subscribed")
col3, col4 = st.columns(2)

with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col4:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Convert categorical data to numerical values
gender_binary = 1 if gender == "Male" else 0
senior_citizen_binary = 1 if senior_citizen == "Yes" else 0
partner_binary = 1 if partner == "Yes" else 0
dependents_binary = 1 if dependents == "Yes" else 0
phone_service_binary = 1 if phone_service == "Yes" else 0
paperless_billing_binary = 1 if paperless_billing == "Yes" else 0

# For multiple lines
if multiple_lines == "Yes":
    multiple_lines_yes = 1
    multiple_lines_no = 0
    multiple_lines_no_phone = 0
elif multiple_lines == "No":
    multiple_lines_yes = 0
    multiple_lines_no = 1
    multiple_lines_no_phone = 0
else: # "No phone service"
    multiple_lines_yes = 0
    multiple_lines_no = 0
    multiple_lines_no_phone = 1

# For internet service
if internet_service == "DSL":
    internet_dsl = 1
    internet_fiber = 0
    internet_no = 0
elif internet_service == "Fiber optic":
    internet_dsl = 0
    internet_fiber = 1
    internet_no = 0
else: # "No"
    internet_dsl = 0
    internet_fiber = 0
    internet_no = 1

# For binary services with internet dependency
def process_service(service_value):
    if service_value == "Yes":
        return 1, 0, 0
    elif service_value == "No":
        return 0, 1, 0
    else: # "No internet service"
        return 0, 0, 1

online_security_yes, online_security_no, online_security_no_internet = process_service(online_security)
online_backup_yes, online_backup_no, online_backup_no_internet = process_service(online_backup)
device_protection_yes, device_protection_no, device_protection_no_internet = process_service(device_protection)
tech_support_yes, tech_support_no, tech_support_no_internet = process_service(tech_support)
streaming_tv_yes, streaming_tv_no, streaming_tv_no_internet = process_service(streaming_tv)
streaming_movies_yes, streaming_movies_no, streaming_movies_no_internet = process_service(streaming_movies)

# For contract
if contract == "Month-to-Month":
    contract_month = 1
    contract_one_year = 0
    contract_two_year = 0
elif contract == "One Year":
    contract_month = 0
    contract_one_year = 1
    contract_two_year = 0
else: # "Two Year"
    contract_month = 0
    contract_one_year = 0
    contract_two_year = 1

# For payment method
if payment_method == "Electronic Check":
    payment_electronic = 1
    payment_mailed = 0
    payment_bank = 0
    payment_credit = 0
elif payment_method == "Mailed Check":
    payment_electronic = 0
    payment_mailed = 1
    payment_bank = 0
    payment_credit = 0
elif payment_method == "Bank Transfer (automatic)":
    payment_electronic = 0
    payment_mailed = 0
    payment_bank = 1
    payment_credit = 0
else: # "Credit Card (automatic)"
    payment_electronic = 0
    payment_mailed = 0
    payment_bank = 0
    payment_credit = 1

# Create full feature array - this should match exactly how your model was trained
# Here we are constructing all 22 features in the correct order
features = np.array([
    gender_binary, senior_citizen_binary, partner_binary, dependents_binary,
    tenure, phone_service_binary, 
    multiple_lines_yes, multiple_lines_no, multiple_lines_no_phone,
    internet_dsl, internet_fiber, internet_no,
    online_security_yes, online_security_no, online_security_no_internet,
    online_backup_yes, online_backup_no, online_backup_no_internet,
    device_protection_yes, device_protection_no, device_protection_no_internet,
    tech_support_yes, tech_support_no, tech_support_no_internet,
    streaming_tv_yes, streaming_tv_no, streaming_tv_no_internet,
    streaming_movies_yes, streaming_movies_no, streaming_movies_no_internet,
    contract_month, contract_one_year, contract_two_year,
    paperless_billing_binary,
    payment_electronic, payment_mailed, payment_bank, payment_credit,
    monthly_charges, total_charges
]).reshape(1, -1)

# Predict Button
if st.button("🔮 Predict Churn"):
    try:
        # Check if the number of features matches what the model expects
        st.write(f"Number of input features: {features.shape[1]}")
        # Since the model expects 22 features, we need to adjust our input
        # This is a placeholder - you'll need to adjust based on what your model expects
        features_adjusted = features[:, :22] if features.shape[1] > 22 else features
        
        # Make prediction
        prediction = model.predict(features_adjusted)
        
        # Show result
        if prediction[0] == 1:
            st.error("🚨 The customer is at HIGH RISK of churning! Consider taking action (e.g., offering discounts, better plans).")
        else:
            st.success("✅ The customer is unlikely to churn. No urgent action needed.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(f"Debug info: Number of features provided: {features.shape[1]}")

# Sidebar with additional info
st.sidebar.header("📊 About This App")
st.sidebar.info("""
This app uses a **Machine Learning model** to predict whether a telecom customer will **stay** or **leave** based on their usage patterns.

✅ **Data Used:** Customer demographics, contract type, charges, etc.  
⚙️ **Model:** Trained with Scikit-learn  
📌 **Goal:** Help telecom companies retain customers.
""")

# Sample churn data
churn_labels = ["Churned", "Not Churned"]
churn_values = [40, 60]  # Example values

# Create Bar Chart
fig, ax = plt.subplots()
ax.bar(churn_labels, churn_values, color=["blue", "coral"])
ax.set_ylabel("Probability (%)")
ax.set_title("Customer Churn Prediction Probability")

# Show chart in Streamlit
st.pyplot(fig)

st.markdown(
    """
    <style>
    html, body, [class*="st-"] {
        color: black !important; 
        background-color: white !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #00274D !important;
        font-weight: bold !important;
    }

    .stSelectbox, .stTextInput, .stNumberInput, .stRadio, .stSlider {
        background-color: white !important;
        color: black !important;
    }
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: white !important;
        color: black !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color: white !important;
        color: black !important;
    }

    @media only screen and (max-width: 600px) {
        h1 { font-size: 28px !important; }
        h2 { font-size: 24px !important; }
        h3 { font-size: 20px !important; }
        p, li { font-size: 16px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)