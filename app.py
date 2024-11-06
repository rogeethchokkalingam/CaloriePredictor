import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
df = pd.read_csv('calories.csv')
df.replace({'male': 0, 'female': 1}, inplace=True)

# Dropping and preprocessing data for training
to_remove = ['Weight', 'Duration', 'User_ID']
df.drop(to_remove, axis=1, inplace=True)
features = df.drop(['Calories'], axis=1)
target = df['Calories']

# Check the number of features after dropping columns
print(features.columns)  # Debugging line to check columns after drop

# Fit the scaler on the reduced feature set and train the model
scaler = StandardScaler()
X = scaler.fit_transform(features)
model = RandomForestRegressor()
model.fit(X, target)

# Streamlit UI
st.title("Calories Burnt Prediction App")
st.write("Enter details to estimate calories burned.")

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
gender = st.selectbox("Gender", ('Male', 'Female'))
activity_duration = st.number_input("Activity Duration (in minutes)", min_value=1, max_value=300, value=30)

# Convert gender to numerical format
gender = 0 if gender == 'Male' else 1

# Prediction button
if st.button("Predict"):
    # Re-format input data to match the scaled features
    input_data = scaler.transform([[age, height, gender, activity_duration, 0]])  # Ensure all features match
    calories_burnt = model.predict(input_data)
    st.success(f"Estimated Calories Burned: {calories_burnt[0]:.2f} kcal")