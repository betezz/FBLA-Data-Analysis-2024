import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.var_model import VAR

# Load the PM2.5 data
df = pd.read_csv('data/grouped_data/Fine_particles__PM_2_5_.csv')

# Convert start_date to datetime
df['start_date'] = pd.to_datetime(df['start_date'])

# Let's focus on citywide measurements first
citywide_data = df[df['geo_type_name'] == 'Citywide']

# Sort by date
citywide_data = citywide_data.sort_values('start_date')

# Create a time series using the mean PM2.5 values
ts = citywide_data.set_index('start_date')['data_value']

# Resample to monthly frequency to handle any gaps
monthly_ts = ts.resample('ME').mean()

# Fill any missing values using forward fill
monthly_ts = monthly_ts.ffill()

# Plot the time series
plt.figure(figsize=(12,6))
plt.plot(monthly_ts)
plt.title('Monthly Average PM2.5 Levels')
plt.xlabel('Date')
plt.ylabel('PM2.5 (mcg/m3)')
#plt.show()

# Perform seasonal decomposition
decomposition = seasonal_decompose(monthly_ts, period=12)
decomposition.plot()
plt.tight_layout()
#plt.show()

# Fit SARIMA model
# Starting with order=(1,1,1) and seasonal_order=(1,1,1,12)
model = SARIMAX(monthly_ts,
                order=(1,1,1),
                seasonal_order=(1,1,1,12))

results = model.fit()

# Print model summary
print(results.summary())

# Plot diagnostics
results.plot_diagnostics(figsize=(12,8))
#plt.show()

# Make predictions
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(monthly_ts.index, monthly_ts, label='Observed')
plt.plot(forecast_mean.index, forecast_mean, color='r', label='Forecast')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:,0],
                 forecast_ci.iloc[:,1], color='r', alpha=.1)
plt.title('PM2.5 Levels: Actual vs Forecast')
plt.legend()
#plt.show()

# Load and prepare exogenous variables
truck_df = pd.read_csv('data/grouped_data/Annual_miles_traveled_trucks.csv')
traffic_df = pd.read_csv('data/grouped_data/Annual_vehicle_miles_traveled.csv')

# Filter for citywide measurements and convert dates
truck_citywide = truck_df[truck_df['geo_type_name'] == 'Citywide'].copy()
traffic_citywide = traffic_df.groupby('start_date')['data_value'].mean().reset_index()

truck_citywide['start_date'] = pd.to_datetime(truck_citywide['start_date'])
traffic_citywide['start_date'] = pd.to_datetime(traffic_citywide['start_date'])

# Create time series for exogenous variables
truck_ts = truck_citywide.set_index('start_date')['data_value']
traffic_ts = traffic_citywide.set_index('start_date')['data_value']

# Print raw data diagnostics
print("\nRaw Data Ranges:")
print(f"PM2.5: {monthly_ts.index.min()} to {monthly_ts.index.max()}")
print(f"Trucks: {truck_ts.index.min()} to {truck_ts.index.max()}")
print(f"Traffic: {traffic_ts.index.min()} to {traffic_ts.index.max()}")

# Resample to monthly frequency and interpolate
truck_monthly = truck_ts.resample('ME').mean().interpolate(method='linear')
traffic_monthly = traffic_ts.resample('ME').mean().interpolate(method='linear')

# Standardize the exogenous variables
truck_monthly = (truck_monthly - truck_monthly.mean()) / truck_monthly.std()
traffic_monthly = (traffic_monthly - traffic_monthly.mean()) / traffic_monthly.std()

# Align all series to the same date range
start_date = max(monthly_ts.index.min(), truck_monthly.index.min(), traffic_monthly.index.min())
end_date = min(monthly_ts.index.max(), truck_monthly.index.max(), traffic_monthly.index.max())

# Trim all series to the same date range
monthly_ts = monthly_ts[start_date:end_date]
truck_monthly = truck_monthly[start_date:end_date]
traffic_monthly = traffic_monthly[start_date:end_date]

# Create aligned dataset
aligned_data = pd.concat([monthly_ts, truck_monthly, traffic_monthly], axis=1)
aligned_data.columns = ['pm25', 'trucks', 'traffic']

# Print alignment diagnostics
print("\nAligned Data Info:")
print(aligned_data.describe())
print("\nMissing Values:")
print(aligned_data.isnull().sum())

# Remove any remaining NaN values
aligned_data = aligned_data.dropna()

# Check if we have enough data points
if len(aligned_data) < 24:  # minimum requirement for seasonal model
    print("\nWarning: Not enough data points for seasonal modeling")
    # Use simpler model parameters
    order = (1,0,0)
    seasonal_order = (0,0,0,0)
else:
    order = (1,1,1)
    seasonal_order = (1,1,1,12)

# Print date distributions and ranges
print("\n=== Date Range Analysis ===")
print("PM2.5 Data:")
print(f"Start: {monthly_ts.index.min()}")
print(f"End: {monthly_ts.index.max()}")
print(f"Number of observations: {len(monthly_ts)}")
print(f"Frequency of observations: {monthly_ts.index.freq}")

print("\nTraffic Data:")
print(f"Start: {traffic_ts.index.min()}")
print(f"End: {traffic_ts.index.max()}")
print(f"Number of observations: {len(traffic_ts)}")

# Check for gaps in time series
print("\n=== Missing Values Analysis ===")
pm25_gaps = monthly_ts.isna().sum()
traffic_gaps = traffic_ts.isna().sum()
print(f"PM2.5 missing values: {pm25_gaps}")
print(f"Traffic missing values: {traffic_gaps}")

# Resample traffic data to monthly frequency and interpolate
traffic_monthly = traffic_ts.resample('ME').mean().interpolate(method='linear')

# Normalize traffic data to PM2.5 scale
pm25_mean = monthly_ts.mean()
pm25_std = monthly_ts.std()
traffic_monthly_normalized = (traffic_monthly - traffic_monthly.mean()) / traffic_monthly.std() * pm25_std + pm25_mean

# Create aligned dataset with normalized traffic
aligned_data = pd.concat([monthly_ts, traffic_monthly_normalized], axis=1)
aligned_data.columns = ['pm25', 'traffic']

# Print correlation analysis
print("\n=== Correlation Analysis ===")
print("Correlation between PM2.5 and Traffic:")
print(aligned_data.corr())

# Print summary statistics
print("\n=== Summary Statistics ===")
print("PM2.5 Statistics:")
print(aligned_data['pm25'].describe())
print("\nNormalized Traffic Statistics:")
print(aligned_data['traffic'].describe())

# Visualize the alignment
plt.figure(figsize=(12,6))
plt.plot(aligned_data.index, aligned_data['pm25'], label='PM2.5')
plt.plot(aligned_data.index, aligned_data['traffic'], label='Normalized Traffic')
plt.title('PM2.5 vs Normalized Traffic Data')
plt.legend()
#plt.show()

# Continue with SARIMAX modeling using normalized traffic data
try:
    sarimax_model = SARIMAX(aligned_data['pm25'],
                           exog=aligned_data[['traffic']],
                           order=order,
                           seasonal_order=seasonal_order)

    sarimax_results = sarimax_model.fit()

    # Print model summary
    print("\nSARIMAX Model Summary:")
    print(sarimax_results.summary())

    # Only plot diagnostics if we have enough data
    if len(aligned_data) >= 24:
        sarimax_results.plot_diagnostics(figsize=(12,8))
        #plt.show()

    # Make predictions
    future_exog = pd.DataFrame({
        'traffic': [aligned_data['traffic'].iloc[-1]] * 12
    })

    sarimax_forecast = sarimax_results.get_forecast(steps=12, exog=future_exog)
    sarimax_mean = sarimax_forecast.predicted_mean
    sarimax_ci = sarimax_forecast.conf_int()

    # Plot SARIMAX forecast
    plt.figure(figsize=(12,6))
    plt.plot(aligned_data.index, aligned_data['pm25'], label='Observed')
    plt.plot(sarimax_mean.index, sarimax_mean, color='g', label='SARIMAX Forecast')
    plt.fill_between(sarimax_ci.index,
                     sarimax_ci.iloc[:,0],
                     sarimax_ci.iloc[:,1], color='g', alpha=.1)
    plt.title('PM2.5 Levels: SARIMAX Forecast with Traffic Data')
    plt.legend()
    plt.show()

    # Compare model performance
    print("\nModel Comparison:")
    print("SARIMA AIC:", results.aic)
    print("SARIMAX AIC:", sarimax_results.aic)

except ValueError as e:
    print(f"\nError fitting SARIMAX model: {e}")
    print("Consider using simpler model parameters or gathering more data.")

# Combine multiple time series
combined_data = pd.concat([
    monthly_ts,
    traffic_monthly
], axis=1)

model = VAR(combined_data)