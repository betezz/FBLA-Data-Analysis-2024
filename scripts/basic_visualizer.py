import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import main_data

# Load cleaned data

# features for each are:
# 'unique_id,indicator_id,name,measure,geo_type_name,geo_join_id,geo_place_name,time_period,start_date,data_value,measure_unit,year,indicator_name,location,season'
fine_particulate_matter = main_data.fine_particulate_matter
nitrogen_dioxide = main_data.nitrogen_dioxide  
ozone = main_data.ozone
benzene = main_data.benzene
formaldehyde = main_data.formaldehyde

boiler_nox = main_data.boiler_nox
boiler_pm25 = main_data.boiler_pm25
boiler_so2 = main_data.boiler_so2

vehicle_miles = main_data.vehicle_miles
car_miles = main_data.car_miles
truck_miles = main_data.truck_miles

deaths_pm25 = main_data.deaths_pm25
cardiac_deaths_ozone = main_data.cardiac_deaths_ozone
cardio_hosp_pm25 = main_data.cardio_hosp_pm25
resp_hosp_pm25 = main_data.resp_hosp_pm25

asthma_ed_pm25 = main_data.asthma_ed_pm25
asthma_ed_ozone = main_data.asthma_ed_ozone
asthma_hosp_ozone = main_data.asthma_hosp_ozone

# Select relevant columns (including season)
fine_pm = fine_particulate_matter[['geo_place_name', 'year', 'data_value', 'season']].rename(columns={'data_value': 'fine_pm'})
ozone_data = ozone[['geo_place_name', 'year', 'data_value', 'season']].rename(columns={'data_value': 'ozone'})
deaths_data = deaths_pm25[['geo_place_name', 'year', 'data_value', 'season']].rename(columns={'data_value': 'deaths_pm25'})

# Merge on common keys like geo_place_name and year
merged_data = fine_pm.merge(ozone_data, on=['geo_place_name', 'year'], how='inner')
merged_data = merged_data.merge(deaths_data, on=['geo_place_name', 'year'], how='inner')

# Only select numeric columns for correlation
numeric_data = merged_data[['fine_pm', 'ozone', 'deaths_pm25']]
correlation_matrix = numeric_data.corr()

# Trend over time for pollutants
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_data, x='year', y='fine_pm', label='Fine Particulate Matter')
sns.lineplot(data=merged_data, x='year', y='ozone', label='Ozone')
plt.title('Air Pollutant Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Concentration')
plt.legend()
plt.show()

# Create a separate seasonal analysis for PM2.5 and NO2
seasonal_pm = fine_particulate_matter[['geo_place_name', 'year', 'data_value', 'season']].rename(columns={'data_value': 'PM2.5'})
seasonal_no2 = nitrogen_dioxide[['geo_place_name', 'year', 'data_value', 'season']].rename(columns={'data_value': 'NO2'})

# Merge these two specifically for seasonal analysis
seasonal_merged = seasonal_pm.merge(seasonal_no2, on=['geo_place_name', 'year', 'season'], how='inner')

# Calculate and plot seasonal averages
seasonal_avg = seasonal_merged.groupby('season')[['PM2.5', 'NO2']].mean()


plt.figure(figsize=(10, 6))
seasonal_avg.plot(kind='bar', figsize=(10, 6))
plt.title('Seasonal Variation in PM2.5 and NO2')
plt.xlabel('Season')
plt.ylabel('Concentration')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=merged_data, x='fine_pm', y='deaths_pm25')
plt.title('Relationship between PM2.5 and Mortality')
plt.xlabel('Fine Particulate Matter (PM2.5)')
plt.ylabel('Deaths attributed to PM2.5')
plt.show()


