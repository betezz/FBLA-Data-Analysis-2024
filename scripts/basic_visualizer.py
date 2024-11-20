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

# Select relevant columns (like data_value, geo_place_name, year, season)
fine_pm = fine_particulate_matter[['geo_place_name', 'year', 'data_value']].rename(columns={'data_value': 'fine_pm'})
ozone_data = ozone[['geo_place_name', 'year', 'data_value']].rename(columns={'data_value': 'ozone'})
deaths_data = deaths_pm25[['geo_place_name', 'year', 'data_value']].rename(columns={'data_value': 'deaths_pm25'})

# Merge on common keys like geo_place_name and year
merged_data = fine_pm.merge(ozone_data, on=['geo_place_name', 'year'], how='inner')
merged_data = merged_data.merge(deaths_data, on=['geo_place_name', 'year'], how='inner')

# Only select numeric columns for correlation
numeric_data = merged_data[['fine_pm', 'ozone', 'deaths_pm25']]
correlation_matrix = numeric_data.corr()

# Plot heatmap
region_averages = merged_data.groupby('geo_place_name').mean()

# Bar chart for deaths related to PM2.5
