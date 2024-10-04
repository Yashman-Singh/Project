import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

# Title and description
st.title("Stock Market Prediction Using Machine Learning")
st.write("""
This app demonstrates a stock market prediction using Random Forest, LSTM, and Support Vector Machine (SVM).
The data is preprocessed with Z-score standardization and features like moving averages and rolling statistics are engineered to enhance the model performance.
""")

# Dataset loading
st.sidebar.title("Dataset")
st.sidebar.write("We will use the following dataset from Kaggle for stock prices:")
dataset_link = "https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset"
st.sidebar.write(f"[Dataset Link]({dataset_link})")

# Load dataset
@st.cache
def load_data():
    # Replace with actual dataset loading
    data = pd.read_csv('path_to_dataset.csv')  # You would point this to the real dataset file.
    return data

data = load_data()

# Preprocessing
st.sidebar.subheader("Preprocessing")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(data.head())

# Z-score Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Open', 'Close', 'High', 'Low', 'Volume']])

# Feature Engineering
st.sidebar.subheader("Feature Engineering")
if st.sidebar.checkbox("Apply Moving Averages"):
    window = st.sidebar.slider("Select window size", min_value=1, max_value=100, value=5)
    data['Moving Average'] = data['Close'].rolling(window=window).mean()
    st.line_chart(data[['Close', 'Moving Average']])

if st.sidebar.checkbox("Apply Rolling Statistics"):
    data['Rolling Std Dev'] = data['Close'].rolling(window=window).std()
    st.line_chart(data[['Close', 'Rolling Std Dev']])

# Train-Test Split
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Machine Learning Models
st.sidebar.title("Models")
model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "SVM"])

if model_choice == "Random Forest":
    st.subheader("Random Forest Model")
    # Random Forest Regressor
    rf = RandomForestRegressor()
    rf.fit(train[['Open', 'Close', 'High', 'Low', 'Volume']], train['Close'])
    predictions = rf.predict(test[['Open', 'Close', 'High', 'Low', 'Volume']])
elif model_choice == "SVM":
    st.subheader("Support Vector Machine Model")
    # Support Vector Regressor
    svm = SVR()
    svm.fit(train[['Open', 'Close', 'High', 'Low', 'Volume']], train['Close'])
    predictions = svm.predict(test[['Open', 'Close', 'High', 'Low', 'Volume']])

# Display Predictions vs Actual
st.subheader("Predicted vs Actual Stock Prices")
test['Predictions'] = predictions
st.line_chart(test[['Close', 'Predictions']])

# Evaluation Metrics
st.subheader("Model Evaluation")
mse = np.mean((test['Close'] - predictions) ** 2)
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
