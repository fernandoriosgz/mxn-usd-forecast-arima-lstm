# MXN/USD Exchange Rate Forecasting — ARIMA vs LSTM

Time-series project that forecasts the Mexican Peso / US Dollar (MXN/USD)
exchange rate using two approaches: a classical **ARIMA** model and a deep
learning **LSTM** neural network. The goal is to compare a statistical model
against a neural network for daily exchange-rate forecasting and to test how
each behaves around major structural shocks.

## Dataset

Daily MXN/USD exchange rate from **Banxico** (series `SF63528`), covering
**January 4, 2016 – April 25, 2024** (~2,092 business days).

| Statistic | Value (pesos per dollar) |
|-----------|--------------------------|
| Mean      | 19.37                    |
| Median    | 19.22                    |
| Min       | 16.34                    |
| Max       | 25.12                    |
| Std. dev. | 1.46                     |

The series shows two major volatility events: the November 2016 US election
and the March 2020 COVID-19 peak (~25 pesos), followed by a sustained
appreciation of the peso from 2023 onward.

## Methodology

### Exploratory analysis & stationarity
- Log transform and rolling means (30 / 90 / 252 days) to inspect structure.
- **Stationarity tests**: ADF (p = 0.079) and Phillips-Perron (p = 0.060) both
  indicate a non-stationary series. After first differencing, ADF p = 0.000 →
  stationary, so the integration order is **d = 1**.
- **Kruskal-Wallis** (p = 0.000) confirms distributions differ across years.
- **ACF / seasonal decomposition**: slow ACF decay and no repeating pattern →
  no seasonality; the series behaves as long-memory.
- **Zivot-Andrews** (p = 0.350) → structural breaks with permanent shocks.

### ARIMA
- ACF/PACF suggested AR(1), MA(0), so the model tested is **ARIMA(1, 1, 0)**,
  selected by the lowest BIC (−1916.9; AIC −1928.3).
- Residual diagnostics: Ljung-Box p = 0.91 (no autocorrelation), with
  heteroskedasticity and non-normal residuals (typical of financial series).

### LSTM
- Data scaled with `MinMaxScaler`; sequences built with a **60-day lookback**.
- 80/20 train/test split.
- Architecture: `LSTM(128) → LSTM(64) → LSTM(32)` with `Dropout(0.2)` between
  layers, then `Dense(8, relu)` and `Dense(1)`. Optimizer: Adam, loss: MSE.
- `EarlyStopping` (patience = 50) stopped training at **136 epochs**; no
  significant overfitting.

Both models were used to forecast **30, 60, and 90 days** ahead.

## Results

| Metric | ARIMA(1,1,0) | LSTM   |
|--------|--------------|--------|
| MSE    | 0.1628       | 0.0359 |
| RMSE   | 0.4035       | 0.1895 |
| MAE    | 0.1126       | 0.1509 |
| MAPE   | 0.58%        | 0.86%  |
| R²     | 0.9232       | 0.9687 |

**Behavior around structural shocks (RMSE):**

| Period          | ARIMA  | LSTM   |
|-----------------|--------|--------|
| Trump, Nov 2016 | 0.2360 | 0.2408 |
| Pandemic, 2020  | 0.3742 | 0.3845 |
| Inflation, 2022 | 0.1264 | 0.1288 |

**Conclusion:** The LSTM achieves a better overall fit (higher R², lower
MSE/RMSE), while ARIMA is more precise on pointwise error (lower MAE/MAPE) and
edges out the LSTM in all three structural-shock periods. Both models struggle
most during the 2020 pandemic, confirming that abrupt exogenous shocks are the
hardest to predict.

## Tech Stack

- Python
- pandas, numpy
- statsmodels (ARIMA, ADF, Zivot-Andrews, seasonal decomposition, ACF/PACF)
- arch (Phillips-Perron), scipy (Kruskal-Wallis)
- scikit-learn (metrics, MinMaxScaler)
- TensorFlow / Keras (LSTM)
- matplotlib

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/fernandoriosgz/mxn-usd-forecast-arima-lstm.git
   cd mxn-usd-forecast-arima-lstm
