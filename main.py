#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os


# In[29]:


# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df


# In[30]:


df.head(40)


# In[31]:


# Convert 'TotalCharges' to numeric (handling errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


# In[32]:


# Encode categorical variables
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# In[33]:


# Feature Engineering
df['MonthlyChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
df['TotalChargesPerTenure'] = df['TotalCharges'] / (df['tenure'] + 1)


# In[34]:


# Define features and target variable
X = df.drop(columns=['Churn'])
y = df['Churn']


# In[ ]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[36]:


# Train multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "Logistic Regression": LogisticRegression()
}


# In[37]:


# Store model results
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))


# In[38]:


# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy: {results[best_model_name]:.4f}")


# In[42]:


# Create the "models" directory if it doesn't exist
os.makedirs("models", exist_ok=True)


# In[43]:


# Save the best model
joblib.dump(best_model, "models/customer_churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")


# In[44]:


# =========================
# Command Line Input for Prediction
# =========================
def predict_churn():
    model = joblib.load("models/customer_churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    print("\nEnter customer details for prediction:")
    user_input = []
    for col in X.columns:
        value = float(input(f"{col}: "))
        user_input.append(value)

    features_scaled = scaler.transform([user_input])
    prediction = model.predict(features_scaled)
    print("\n\U0001F534 Churn" if prediction[0] == 1 else "\U0001F7E2 Not Churn")

if __name__ == "__main__":
    while True:
        choice = input("\nDo you want to predict churn? (yes/no): ").strip().lower()
        if choice == "yes":
            predict_churn()
        else:
            print("Exiting...")
            break

