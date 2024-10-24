# Real Estate Price Prediction App

This is a Real Estate Price Prediction application developed using Python and Streamlit. The app predicts the price of a house's unit area based on various features such as distance to the nearest MRT station, the number of convenience stores, latitude, and longitude. The model used for prediction is a Linear Regression model trained on a dataset of real estate properties in Taiwan.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Model](#model)
- [App Interface](#app-interface)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Project Overview

This project is focused on building a machine learning model to predict house prices based on several factors. The app is built with Streamlit, which provides an easy-to-use web interface for users to input data and get predictions. The Linear Regression model used in this project is trained on real estate data and can predict house prices per unit area in both New Taiwan Dollars (NTD) and US Dollars (USD).

## Dataset Information

The dataset used for this project contains information about real estate properties in Taiwan. Key variables in the dataset include:
- **House age**: The age of the house.
- **Distance to the nearest MRT station**: The distance from the property to the nearest MRT station, measured in meters.
- **Number of convenience stores**: The number of convenience stores within walking distance from the property.
- **Latitude** and **Longitude**: The geographical coordinates of the property.
- **House price of unit area**: The actual price per unit area of the property.

## Features

The app allows users to input the following features to make predictions:
- **Distance to the nearest MRT station**: Enter the distance to the nearest MRT station (in meters).
- **Number of convenience stores**: Enter the number of convenience stores near the property.
- **Latitude**: Enter the latitude of the property.
- **Longitude**: Enter the longitude of the property.

### Predicted Output:
- The predicted house price of the unit area can be displayed in:
  - **New Taiwan Dollars (NTD)** or
  - **US Dollars (USD)**

The user can choose the currency in which they wish to see the predicted price.

## Model

The model used in this project is a **Linear Regression model**. The following steps were taken in training and evaluation:
- The dataset was split into training and testing sets using an 80/20 ratio.
- The **Linear Regression model** was fitted to the training data to learn the relationships between the features and the target variable (house price).
- After training, predictions were made on the test set, and the model's performance was evaluated using metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)**.

### Visualizations:

Several data visualizations were generated to help understand the dataset:
- **Histograms**: Distribution of house age, distance to MRT station, number of convenience stores, latitude, longitude, and house price.
- **Scatter plots**: Relationships between individual features and the house price of the unit area.
- **Correlation matrix**: A heatmap showing the correlation between different numeric features.

## App Interface

The Streamlit app provides a simple interface where users can input the required data to make predictions. Here is how it works:

1. **Input Fields**:
   - Distance to the nearest MRT station (in meters)
   - Number of convenience stores nearby
   - Latitude of the property
   - Longitude of the property

2. **Currency Selection**:
   - Users can choose whether they want the house price prediction in **New Taiwan Dollars (NTD)** or **US Dollars (USD)**.

3. **Prediction Button**:
   - After entering the values, the user can click the "Predict" button to get the predicted house price.

4. **Prediction Output**:
   - The predicted house price is displayed in the selected currency format.

## Installation Instructions

To run the project locally, follow these steps:

### Step 1: Clone the repository
Clone this GitHub repository to your local machine

### Step 2: Install dependencies
Navigate to the project directory and install the required Python packages.
Make sure you have all the necessary libraries installed:
- pandas
- numpy
- scikit-learn
- streamlit
- seaborn
- matplotlib

### Step 3 : Run the Streamlit app
After installing the dependencies, you can run the Streamlit app

### Usage

Once the app is running, you will see a web interface where you can input values for:

- Distance to the nearest MRT station
- Number of convenience stores
- Latitude
- Longitude
- Then, choose the currency in which you want the house price to be predicted (NTD or USD) and click "Predict." The app will display the predicted house price based on the input values.

**Example**

If the following inputs are provided:

- Distance to MRT station: 300 meters
- Number of convenience stores: 5
- Latitude: 25.033
- Longitude: 121.565

Output be :

Predicted House Price of Unit Area: 42.56 NTD
OR
Predicted House Price of Unit Area: 1.37 USD (if the USD option is selected:)

### Future Improvements

Here are some potential future improvements for this project:

- Add more features to improve the accuracy of the predictions (e.g., house age, floor area).
- Integrate more machine learning models and compare performance (e.g., Decision Trees, Random Forests).
- Implement model evaluation directly in the app, allowing users to see how well the model performs on the test data.
- Include additional visualizations, such as feature importance charts or geographic maps showing property prices.

