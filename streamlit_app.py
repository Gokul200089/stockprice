import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Set the page title
st.title('Stock Price Prediction and Forecast Dashboard')

# Sidebar for stock selection and model configuration
st.sidebar.header('Stock Selection')
stock_symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AMZN, GOOG, TSLA)', 'AMZN')
forecast_period = st.sidebar.slider('Select Forecast Period (minutes)', min_value=60, max_value=1440, value=60, step=60)

# Fetch stock data
st.sidebar.header('Fetching Stock Data...')
stock = yf.Ticker(stock_symbol)
hist = stock.history(period="1d", interval="1m")
st.sidebar.success('Data Loaded Successfully!')

# Display raw data in the main section
st.write(f"Showing stock data for {stock_symbol}")
st.dataframe(hist)

# Train ARIMA model
st.sidebar.header('Training Model...')
close_prices = hist['Close']
model = ARIMA(close_prices, order=(5, 1, 0))
model_fit = model.fit()
st.sidebar.success('Model Trained Successfully!')

# Forecast for the next 'forecast_period' minutes
st.sidebar.header('Generating Forecast...')
forecast = model_fit.forecast(steps=forecast_period)

# Ensure forecast is 1-dimensional
if forecast.ndim > 1:
    forecast = forecast.flatten()

# Create forecast index
last_time = hist.index[-1]
forecast_index = pd.date_range(last_time, periods=forecast_period, freq='T')

# Plot actual vs forecasted values
st.subheader('Actual vs Forecasted Stock Prices')

plt.figure(figsize=(12, 6))
plt.plot(hist.index, close_prices, label='Actual (Historical)', color='blue')
plt.plot(forecast_index, forecast, label=f'Forecast (Next {forecast_period} Minutes)', color='orange')
plt.title(f'{stock_symbol} Stock Price: Actual vs Forecast')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Calculate accuracy metrics
st.subheader('Accuracy Metrics')
if len(close_prices) >= forecast_period:
    test_set = close_prices[-forecast_period:]
    # Ensure forecast matches length of test_set
    if len(forecast) > len(test_set):
        forecast = forecast[:len(test_set)]
    
    mae = mean_absolute_error(test_set, forecast)
    rmse = np.sqrt(mean_squared_error(test_set, forecast))
    mape = np.mean(np.abs((test_set - forecast) / test_set)) * 100

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
else:
    st.write("Not enough historical data to calculate accuracy metrics.")

st.sidebar.success('Forecast and Accuracy Calculated!')

# End of the dashboard
st.sidebar.write('Adjust forecast period using the slider.')
