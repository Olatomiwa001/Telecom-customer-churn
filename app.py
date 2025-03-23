import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Custom CSS for better UI
st.markdown(
    """
    <style>
        body {
            background-color: #f0f5ff; /* Light Blue Background */
        }
        .stApp {
            background-color: #f0f8ff;
        }
        .big-font {
            font-size:20px !important;
            font-weight: bold;
            color: #004080;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='color:#004080; text-align: center;'>📊 Telecom Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Explanation Section
st.markdown("""
### 🔍 What is Customer Churn?
Customer churn refers to when a customer **stops using a company's service**. 
For telecom companies, it's crucial to **predict and prevent churn** by identifying customers who are likely to leave.

### 📌 How This Model Works
- You enter customer details (gender, contract type, charges, etc.).
- The model analyzes the data and predicts whether the customer will **churn (leave)** or **stay**.
- This helps businesses **take action** (e.g., offering discounts, better plans) to retain customers.

### 🚀 How to Use:
1️⃣ Fill in the customer details below.  
2️⃣ Click **"Predict Churn"**.  
3️⃣ See the result: ✅ (Customer Stays) or 🚨 (Customer Likely to Churn).  
""", unsafe_allow_html=True)

st.write("Enter customer details below to predict churn:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"], key="senior_citizen")
partner = st.selectbox("Has Partner?", ["Yes", "No"], key="partner")
dependents = st.selectbox("Has Dependents?", ["Yes", "No"], key="dependents")
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1, key="tenure")
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.1, key="monthly_charges")
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.1, key="total_charges")
contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"], key="contract")
payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], key="payment")

# Convert categorical data
gender = 1 if gender == "Male" else 0
senior_citizen = 1 if senior_citizen == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0

# Create feature array
features = np.array([gender, senior_citizen, partner, dependents, tenure, monthly_charges, total_charges]).reshape(1, -1)

# Predict Button
if st.button("Predict Churn"):
    try:
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Show result
        if prediction[0] == 1:
            st.error("🚨 The customer is at HIGH RISK of churning! Consider taking action (e.g., offering discounts, better plans).")
        else:
            st.success("✅ The customer is unlikely to churn. No urgent action needed.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar with additional info
st.sidebar.header("📊 About This App")
st.sidebar.info("""
This app uses a **Machine Learning model** to predict whether a telecom customer will **stay** or **leave** based on their usage patterns.

✅ **Data Used:** Customer demographics, contract type, charges, etc.  
⚙️ **Model:** Trained with Scikit-learn  
📌 **Goal:** Help telecom companies retain customers.
""")

# Sample churn data (you can replace this with real data)
churn_labels = ["Churned", "Not Churned"]
churn_values = [40, 60]  # Example: 40% churn rate

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
    .stSelectbox, .stTextInput, .stNumberInput, .stRadio, .stSlider {
        background-color: white !important;
        color: black !important;
    }
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: white !important;
        color: black !important;
    }
    div[data-testid="stNumberInput"] input {
        background-color: white !important;
        color: black !important;
    }
    div[data-testid="stTextInput"] input {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
