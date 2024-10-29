import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
real_estate_data = pd.read_csv(r"dataset path")

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
distance_to_mrt = st.number_input('Distance to the nearest MRT station (in meters)', min_value=0.0)
convenience_stores = st.number_input('Number of convenience stores', min_value=0, step=1)
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0)

# Currency selection
currency = st.selectbox('Select the currency:', ('INR', 'USD', 'NTD'))

# Conversion rates
conversion_rates = {
    'INR': 1,       # Keeping INR as base
    'USD': 0.012,   # Example conversion rate for USD
    'NTD': 0.37     # Example conversion rate for NTD
}

# Average price per unit in INR (example values)
average_price_per_sqft_inr = 5000  # Average price per square foot in INR
average_price_per_cent_inr = 500000  # Average price per cent in INR

# Prediction button
if st.button('Predict'):
    # Collect the input data into a numpy array
    input_data = np.array([[distance_to_mrt, convenience_stores, latitude, longitude]])

    # Make prediction
    prediction = model.predict(input_data)

    # Adjust the predicted price based on the selected currency
    if currency in ['INR', 'NTD']:  # For India and Taiwan using cents
        predicted_price_in_inr = prediction[0] * average_price_per_cent_inr
    else:  # For the US using square feet
        predicted_price_in_inr = prediction[0] * average_price_per_sqft_inr

    final_price = predicted_price_in_inr * conversion_rates[currency]

    # Display the prediction result
    if currency in ['INR', 'NTD']:
        unit = "Cent"
    else:
        unit = "Square Foot"

    st.success(f'Predicted House Price per {unit}: {final_price:.2f} {currency}')
