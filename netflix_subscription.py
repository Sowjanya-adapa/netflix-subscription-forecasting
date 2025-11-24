# netflix_subscription.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# ---------- Load data ----------
# Use the exact CSV filename or full path if needed
df = pd.read_csv('Netflix-Subscriptions.csv')

# Quick sanity prints
print(type(df))
print(df.info())

# ---------- Create Time Period index (from Year + Quarter) ----------
quarter_to_month = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}

df['Time Period'] = pd.to_datetime(
    df['Year'].astype(str) + '-' +
    df['Quarter'].map(quarter_to_month) + '-01'
)

df = df.sort_values('Time Period').set_index('Time Period')

# ---------- Normalize column names / create working 'Subscribers' series ----------
# Your CSV uses Subscribers_Millions; create a simple Subscribers column for the model
if 'Subscribers' not in df.columns and 'Subscribers_Millions' in df.columns:
    df['Subscribers'] = df['Subscribers_Millions']

# Ensure numeric columns are numeric (coerce if necessary)
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# ---------- Resampling (quarterly) on numeric columns only ----------
# Use 'QE' (quarter end) to avoid deprecation warnings and aggregate numeric columns only
numeric_cols = df.select_dtypes(include=['number']).columns
df_resampled = df[numeric_cols].resample('QE').mean().ffill()

# Keep a representative non-numeric label for the quarter if desired
if 'Quarter' in df.columns:
    df_resampled['Quarter'] = df['Quarter'].resample('QE').last()

# Also useful: Year column from index
df_resampled['Year'] = df_resampled.index.year

print("\nResampled dataframe (head):")
print(df_resampled.head())
print(df_resampled.info())

# ---------- Decomposition ----------
# Make sure the series exists and drop NaNs
series = df_resampled['Subscribers'].dropna()

# seasonal_decompose needs a period parameter for quarterly data => period=4
decomposition = seasonal_decompose(series, model='additive', period=4)

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(series, label='Original')
plt.legend(loc='upper left')
plt.title('Time Series - Original Data')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.title('Time Series - Trend Component')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.title('Time Series - Seasonal Component')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.title('Time Series - Residual Component')

plt.tight_layout()
plt.show()

# ---------- SARIMA Hyperparameter Search ----------
# Narrow search ranges so it runs in a reasonable time
p = d = q = range(0, 2)
pdq = list(product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 4) for x in pdq]  # 4 for quarterly seasonality

best_aic = float("inf")
best_pdq = None
best_seasonal_pdq = None

# Use try/except so a failing combination doesn't stop the search
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(series,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except Exception:
            continue

print("\nBest SARIMA (AIC):", best_aic, best_pdq, best_seasonal_pdq)

# ---------- Fit best model ----------
best_model = SARIMAX(series,
                    order=best_pdq,
                    seasonal_order=best_seasonal_pdq,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

best_results = best_model.fit(disp=False)
best_results.plot_diagnostics(figsize=(15, 12))
plt.show()

# ---------- Forecast ----------
steps = 8  # number of future quarters to forecast
forecast = best_results.get_forecast(steps=steps)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

plt.figure(figsize=(14, 7))
plt.plot(df_resampled.index, df_resampled['Subscribers'], label='Observed')
plt.plot(mean_forecast.index, mean_forecast, label='Forecast')
plt.fill_between(confidence_intervals.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], alpha=0.3)
plt.xlabel('Time Period')
plt.ylabel('Subscribers (Millions)')
plt.title('Netflix Subscription Forecast')
plt.legend()
plt.show()
