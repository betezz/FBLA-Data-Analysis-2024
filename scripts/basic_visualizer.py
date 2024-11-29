import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import main_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load cleaned data

# features for each are:
PERSONAL_DATA_PATH = '/Users/morabp27/FBLA-Data-Analysis-2024/'

# Load Boiler Emissions Data
pm25_emissions = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Boiler_Emissions__Total_PM2_5_Emissions.csv')
nox_emissions = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Boiler_Emissions__Total_NOx_Emissions.csv')
so2_emissions = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Boiler_Emissions__Total_SO2_Emissions.csv')

# Load Pollutants Data
pm25 = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Fine_particles__PM_2_5_.csv')
ozone = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Ozone__O3_.csv')
no2 = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Nitrogen_dioxide__NO2_.csv')

# Load Asthma Data
asthma_ed_pm25 = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Asthma_emergency_department_visits_due_to_PM2_5.csv')
asthma_ed_ozone = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Asthma_emergency_departments_visits_due_to_Ozone.csv')
asthma_hosp_ozone = pd.read_csv(PERSONAL_DATA_PATH + 'data/grouped_data/Asthma_hospitalizations_due_to_Ozone.csv')

# Example: Cleaning PM2.5 Emissions Data
pm25_emissions = pm25_emissions.dropna(subset=['geo_place_name', 'year', 'data_value'])
pm25_emissions.rename(columns={'data_value': 'PM2.5_Emissions'}, inplace=True)

# Similarly, clean and rename columns for NOx and SO2 emissions
nox_emissions = nox_emissions.dropna(subset=['geo_place_name', 'year', 'data_value'])
nox_emissions.rename(columns={'data_value': 'NOx_Emissions'}, inplace=True)

so2_emissions = so2_emissions.dropna(subset=['geo_place_name', 'year', 'data_value'])
so2_emissions.rename(columns={'data_value': 'SO2_Emissions'}, inplace=True)

# Merge Emissions Data
emissions = pm25_emissions[['geo_place_name', 'year', 'PM2.5_Emissions']].merge(
    nox_emissions[['geo_place_name', 'year', 'NOx_Emissions']],
    on=['geo_place_name', 'year'],
    how='outer'
).merge(
    so2_emissions[['geo_place_name', 'year', 'SO2_Emissions']],
    on=['geo_place_name', 'year'],
    how='outer'
)

# Clean Pollutants Data
pm25_pollutants = pm25[['geo_place_name', 'year', 'data_value']].rename(columns={'data_value': 'PM2.5'})
ozone_pollutants = ozone[['geo_place_name', 'year', 'data_value']].rename(columns={'data_value': 'O3'})
no2_pollutants = no2[['geo_place_name', 'year', 'data_value']].rename(columns={'data_value': 'NO2'})

# Merge Pollutants Data
pollutants = pm25_pollutants.merge(
    ozone_pollutants,
    on=['geo_place_name', 'year'],
    how='outer'
).merge(
    no2_pollutants,
    on=['geo_place_name', 'year'],
    how='outer'
)

# Final Merge: Emissions + Pollutants
data = emissions.merge(
    pollutants,
    on=['geo_place_name', 'year'],
    how='inner'
)

# Handle Missing Values if any
data = data.dropna()

# Assuming 'season' is available in one of the datasets
# For demonstration, let's create a 'season' column based on 'time_period'

def determine_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

# Extract month from a date column if available
# For example, if 'start_date' is in the format YYYY-MM-DD

'''
# Now create the seasonal plots
seasonal_figs, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PM2.5 by season
sns.boxplot(x='season', y='PM2.5', data=pollutants, ax=ax1)
ax1.set_title('PM2.5 Levels by Season')
ax1.set_xlabel('Season')
ax1.set_ylabel('PM2.5 (mcg/m³)')

# NO2 by season
sns.boxplot(x='season', y='NO2', data=pollutants, ax=ax2)
ax2.set_title('NO2 Levels by Season')
ax2.set_xlabel('Season')
ax2.set_ylabel('NO2 (ppb)')

plt.tight_layout()
plt.show()

# Features and Target
features = ['PM2.5_Emissions', 'NOx_Emissions', 'SO2_Emissions']
target = 'PM2.5'

X = data[features]
y = data[target]

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Model
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Evaluate
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f'Random Forest R² Score: {r2_rf:.2f}')
print(f'Random Forest Mean Squared Error: {mse_rf:.2f}')

# Add correlation analysis between different boiler emissions
boiler_correlations = pd.DataFrame()
for file in ['SO2_Emissions', 'PM2_5_Emissions', 'NOx_Emissions']:
    data = pd.read_csv(PERSONAL_DATA_PATH + f'data/grouped_data/Boiler_Emissions__Total_{file}.csv')
    boiler_correlations[file] = data['data_value']

correlation_matrix = boiler_correlations.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Different Boiler Emissions')
plt.show()
'''

# Analyze trends over time
def analyze_emission_trends(emission_type):
    data = pd.read_csv(PERSONAL_DATA_PATH + f'data/grouped_data/Boiler_Emissions__Total_{emission_type}.csv')
    citywide_data = data[data['geo_place_name'] == 'New York City']
    
    plt.figure(figsize=(12, 6))
    plt.plot(citywide_data['year'], citywide_data['data_value'], marker='o')
    plt.title(f'Citywide {emission_type} Trends')
    plt.xlabel('Year')
    plt.ylabel('Emissions (number per km2)')
    plt.grid(True)
    plt.show()

# Create comprehensive statistical summary
def generate_emission_statistics():
    stats_dict = {}
    for emission in ['SO2_Emissions', 'PM2_5_Emissions', 'NOx_Emissions']:
        data = pd.read_csv(PERSONAL_DATA_PATH + f'data/grouped_data/Boiler_Emissions__Total_{emission}.csv')
        stats_dict[emission] = {
            'mean': data['data_value'].mean(),
            'median': data['data_value'].median(),
            'std': data['data_value'].std(),
            'max_location': data.loc[data['data_value'].idxmax(), 'geo_place_name'],
            'max_value': data['data_value'].max(),
            'min_location': data.loc[data['data_value'].idxmin(), 'geo_place_name'],
            'min_value': data['data_value'].min()
        }
    
    return pd.DataFrame(stats_dict).round(2)

'''

print("\n=== Analyzing Boiler Emissions Impact on Air Quality ===\n")

# 1. Generate and display statistical summary
print("Statistical Summary of Emissions:")
stats_df = generate_emission_statistics()
print(stats_df)
print("\n-----------------------------------\n")

# 2. Run correlation analysis
print("Analyzing correlations between different types of emissions...")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Different Boiler Emissions')
plt.tight_layout()
plt.show()
print("\n-----------------------------------\n")

# 3. Analyze temporal trends for each emission type
print("Analyzing emission trends over time...")
for emission in ['SO2_Emissions', 'PM2_5_Emissions', 'NOx_Emissions']:
    analyze_emission_trends(emission)
print("\n-----------------------------------\n")
print("\n-----------------------------------\n")

# 5. Random Forest Model Results
print("Random Forest Model Results:")
print(f'R² Score: {r2_rf:.2f}')
print(f'Mean Squared Error: {mse_rf:.2f}')

# 6. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Predicting PM2.5 Levels')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("\n=== Analysis Complete ===")
'''
# PM2.5 Emissions vs PM2.5 Pollutant
plt.figure(figsize=(6,4))
sns.scatterplot(x='PM2.5_Emissions', y='PM2.5', data=data)
sns.regplot(x='PM2.5_Emissions', y='PM2.5', data=data, scatter=False)
plt.title('PM2.5 Emissions vs PM2.5 Levels')
plt.xlabel('PM2.5 Emissions (Number per km²)')
plt.ylabel('PM2.5 (mcg/m³)')
plt.show()

# NOx Emissions vs NO2 Pollutant
plt.figure(figsize=(6,4))
sns.scatterplot(x='NOx_Emissions', y='NO2', data=data)
sns.regplot(x='NOx_Emissions', y='NO2', data=data, scatter=False)
plt.title('NOx Emissions vs NO2 Levels')
plt.xlabel('NOx Emissions (Number per km²)')
plt.ylabel('NO2 (ppb)')
plt.show()

# SO2 Emissions vs O3 Pollutant
plt.figure(figsize=(6,4))
sns.scatterplot(x='SO2_Emissions', y='O3', data=data)
sns.regplot(x='SO2_Emissions', y='O3', data=data, scatter=False)
plt.title('SO2 Emissions vs O3 Levels')
plt.xlabel('SO2 Emissions (Number per km²)')
plt.ylabel('O3 (ppb)')
plt.show()







