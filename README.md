# MXN/USD Exchange Rate Forecasting: ARIMA vs LSTM

This project analyzes and forecasts the MXN/USD exchange rate using two different time-series approaches: ARIMA and Long Short-Term Memory (LSTM) neural networks.

## Project Overview

The analysis uses historical MXN/USD exchange rate data from 2016 to 2024. The objective is to analyze the behavior of the time series, build forecasting models, and compare the performance of traditional econometric methods with deep learning techniques.

## Methods

The project includes:

- Exploratory time-series analysis
- Augmented Dickey-Fuller test
- Phillips-Perron test
- First differencing
- Autocorrelation (ACF)
- Partial autocorrelation (PACF)
- Structural break analysis
- Zivot-Andrews test
- ARIMA modeling
- LSTM neural network
- 30, 60 and 90-day forecasts
- Model evaluation using MAE, MSE, RMSE, MAPE and R²
- Comparison of ARIMA and LSTM performance
- Analysis during structural shocks

## Models

### ARIMA

An ARIMA(1,1,0) model is used to forecast the MXN/USD exchange rate.

### LSTM

A multi-layer Long Short-Term Memory neural network is trained using historical exchange-rate observations and a 60-day lookback window.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn
- TensorFlow / Keras
- SciPy

## Data

Historical MXN/USD exchange-rate data obtained from Banco de México (Banxico).

## Objective

The main objective is to compare a traditional statistical time-series model with a deep-learning approach and evaluate their forecasting performance under normal conditions and periods of structural shocks.
