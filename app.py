import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
real_estate_data = pd.read_csv(r"C:\Users\febin\Downloads\Real_Estate.csv")

# Feature selection and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

# Splitting the data into training and testing sets
X = real_estate_data[features]
y = real_estate_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app interface
st.title("Real Estate Price Prediction")

# Input fields for the features
distance_to_mrt = st.number_input('Distance to the nearest MRT station', min_value=0.0)
convenience_stores = st.number_input('Number of convenience stores', min_value=0, step=1)
latitude = st.number_input('Latitude', min_value=0.0)
longitude = st.number_input('Longitude', min_value=0.0)

# Prediction button
if st.button('Predict'):
    # Collect the input data into a numpy array
    input_data = np.array([[distance_to_mrt, convenience_stores, latitude, longitude]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction result
    st.success(f'Predicted House Price of Unit Area: {prediction[0]:.2f}')

