import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
real_estate_data = pd.read_csv(r"C:\Users\febin\GitHubProjects\RealEstatePricePrediction\Real_Estate.csv")

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
distance_to_mrt_meters = st.number_input('Distance to the nearest MRT station (in meters)', min_value=0.0)
distance_to_mrt_km = distance_to_mrt_meters / 1000  # Convert to kilometers
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

# Price adjustment logic per cent for INR, USD, and NTD
cent_prices = {
    'INR': {'1_cent_min': 125000, '1_cent_max': 2700000},
    'USD': {'1_cent_min': 1500, '1_cent_max': 32400},
    'NTD': {'1_cent_min': 46250, '1_cent_max': 999000}
}

# Prediction button
if st.button('Predict'):
    # Collect the input data into a numpy array (distance in km)
    input_data = np.array([[distance_to_mrt_km, convenience_stores, latitude, longitude]])

    # Make prediction
    prediction = model.predict(input_data)

    # Adjust the predicted price based on cent prices (1 cent as base unit)
    predicted_price_in_inr = prediction[0] * cent_prices['INR']['1_cent_min']  # Using minimum price per cent for INR

    # Convert to selected currency
    final_price = predicted_price_in_inr * conversion_rates[currency]

    # Display the prediction result
    st.success(f'Predicted House Price per Cent: {final_price:.2f} {currency}')

# Streamlit description
st.info("""
This Real Estate Price Prediction app estimates property prices based on local cent-based pricing. 
It converts distance inputs from meters to kilometers for accuracy and supports predictions in INR, USD, and NTD.
The price ranges vary based on locality:
- INR: ₹1.25 lakh to ₹27 lakh per cent (India)
- USD: $1,500 to $32,400 per cent (USA)
- NTD: NT$46,250 to NT$999,000 per cent (Taiwan)
""")
