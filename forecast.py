import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the data with the correct delimiter and skip the first row
file_path = r'C:\Users\Administrator\Desktop\EAs\XAUUSD-Forecast\XAUUSD_D1.csv'  # Replace with your actual file path

# Define a custom date parser
def custom_date_parser(date_str):
    for fmt in ("%m.%d.%Y %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')  # Default fallback

# Read the CSV file with semicolon delimiter and skip the first row
data = pd.read_csv(file_path, delimiter=';', skiprows=1, parse_dates=['Date'], date_parser=custom_date_parser)

# Display basic information
print("Data Head:")
print(data.head())
print("\nData Description:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

# Preprocess the data
data = data.fillna(method='ffill')
data.set_index('Date', inplace=True)

# Use the 'Close' price for forecasting
y = data['Close']

# Split the data
train_size = int(len(y) * 0.8)
train_data = y[:train_size]
test_data = y[train_size:]

# Build and train the ARIMA model
warnings.filterwarnings("ignore")
model = ARIMA(train_data, order=(5, 1, 0))  # You can tune the order (p, d, q)
model_fit = model.fit()
print("\nModel Summary:")
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=len(test_data))
forecast_series = pd.Series(forecast, index=test_data.index)

# Remove NaN values
forecast_series = forecast_series.dropna()
test_data = test_data.loc[forecast_series.index]

# Evaluate the model
if len(test_data) > 0 and len(forecast_series) > 0:
    mse = mean_squared_error(test_data, forecast_series)
    print(f'\nTest MSE: {mse}')
else:
    print("test_data or forecast_series is empty.")

# Forecast the next day
model = ARIMA(y, order=(5, 1, 0))
model_fit = model.fit()
next_day_forecast = model_fit.forecast(steps=1)
print(f'\nNext day XAUUSD price prediction: {next_day_forecast.iloc[0]}')