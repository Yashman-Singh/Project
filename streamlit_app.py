import streamlit as st

# Title of the document
st.title("Stock Market Prediction Using Machine Learning")

# Introduction/Background
st.header("Introduction/Background")
st.write("""
There are two main ways for investors to analyze a stock. The first way is through a fundamental analysis, which considers the intrinsic value of stocks, the performance of the industry, the economy, and the political climate. The other main way is through technical analysis, which involves viewing market activity, including previous prices and volumes. 
Due to the volatility of the stock market, it is difficult to make accurate predictions. However, many researchers are attempting to predict price changes using various combinations of preprocessing and machine learning methods.
""")

# Dataset description
st.header("Dataset")
st.write("""
We will use the following dataset from Kaggle: 
[Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset). 
It includes the date, opening price, closing price, maximum price, minimum price, adjusted close price, and volume for each stock up until 04/01/2020. These features will help analyze price changes for different stocks.
""")

# Problem statement
st.header("Problem Statement")
st.write("""
The stock market is inherently volatile and non-linear, making stock price prediction a difficult task. The goal of this project is to use machine learning techniques to increase the accuracy of stock price prediction. This project is motivated by the potential benefits for investors and traders, allowing them to make more informed decisions.
""")

# Methods
st.header("Methods")

st.subheader("Preprocessing")
st.write("""
- **Z-score Standardization**: To account for varying stock prices across different stocks and remove outliers, the data for stock prices will be normalized using Z-score standardization (via `StandardScaler` in scikit-learn).
- **Feature Engineering**: New features such as moving averages and rolling statistics will be created to enhance model performance. Moving averages help smooth out price data to identify trends, while rolling statistics capture short-term variability in stock prices.
- **Train-Test Split**: A time-based train-test split will be used, with earlier data for training and more recent data for testing. This approach mimics real-world scenarios where future prices are predicted based on past data.
""")

st.subheader("ML Algorithms/Models")
st.write("""
- **Random Forest**: Resistant to noise and volatility, Random Forest averages the results of multiple decision trees made with random data points, reducing potential overfitting.
- **Long Short-Term Memory (LSTM)**: LSTM’s architecture allows it to ignore irrelevant short-term fluctuations and focus on long-term patterns, making it ideal for noisy stock market data.
- **Support Vector Machine (SVM)**: SVM is suited for stock market prediction due to its ability to model complex, non-linear relationships by projecting features into higher-dimensional spaces, capturing subtle trends in the data.
""")

# Evaluation metrics
st.header("Evaluation Metrics")
st.write("""
We will use the following quantitative metrics to evaluate model performance:
- **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual stock prices.
""")

# References
st.header("References")
st.write("""
1. K. Vanukuru, "Stock Market Prediction Using Machine Learning," International Research Journal of Engineering and Technology, 2018. [Online](https://doi.org/10.13140/RG.2.2.12300.77448).
2. A. Gupta, Akansha, K. Joshi, M. Patel, and V. Pratap, "Stock Market Prediction using Machine Learning Techniques: A Systematic Review," 2023 International Conference on Power, Instrumentation, Control, and Computing (PICC), Thrissur, India, 2023. DOI: 10.1109/PICC57976.2023.10142862.
""")

# Contributions section
st.header("Contributions")
st.write("""
- **Jake Wang**: Created the introduction/background, preprocessing method, and one ML model.
- **Yashman Singh**: Worked on two ML models, two preprocessing methods, and one quantitative metric.
- **Manya Jain**: Collected sources for the literature review, technical content, and visuals for the presentation, and worked on the Gantt chart.
""")
