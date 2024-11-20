import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

PERSONAL_DATA_PATH = '/Users/morabp27/FBLA-Data-Analysis-2024/data/grouped_data/' # path to grouped data, update if not Boden lmao
# features for each are:
# 'unique_id,indicator_id,name,measure,geo_type_name,geo_join_id,geo_place_name,time_period,start_date,data_value,measure_unit,year,indicator_name,location,season'

# Air Quality Measurements
fine_particulate_matter = pd.read_csv(PERSONAL_DATA_PATH + 'Fine_particles__PM_2_5_.csv')
nitrogen_dioxide = pd.read_csv(PERSONAL_DATA_PATH + 'Nitrogen_dioxide__NO2_.csv')
ozone = pd.read_csv(PERSONAL_DATA_PATH + 'Ozone__O3_.csv')
benzene = pd.read_csv(PERSONAL_DATA_PATH + 'Outdoor_Air_Toxics___Benzene.csv')
formaldehyde = pd.read_csv(PERSONAL_DATA_PATH + 'Outdoor_Air_Toxics___Formaldehyde.csv')

# Emissions Data
boiler_nox = pd.read_csv(PERSONAL_DATA_PATH + 'Boiler_Emissions__Total_NOx_Emissions.csv')
boiler_pm25 = pd.read_csv(PERSONAL_DATA_PATH + 'Boiler_Emissions__Total_PM2_5_Emissions.csv')
boiler_so2 = pd.read_csv(PERSONAL_DATA_PATH + 'Boiler_Emissions__Total_SO2_Emissions.csv')

# Traffic Data
vehicle_miles = pd.read_csv(PERSONAL_DATA_PATH + 'Annual_vehicle_miles_traveled.csv')
car_miles = pd.read_csv(PERSONAL_DATA_PATH + 'Annual_miles_traveled_cars.csv')
truck_miles = pd.read_csv(PERSONAL_DATA_PATH + 'Annual_miles_traveled_trucks.csv')

# Health Impact Data
deaths_pm25 = pd.read_csv(PERSONAL_DATA_PATH + 'Deaths_due_to_PM2_5.csv')
cardiac_deaths_ozone = pd.read_csv(PERSONAL_DATA_PATH + 'Cardiac_and_respiratory_deaths_due_to_Ozone.csv')
cardio_hosp_pm25 = pd.read_csv(PERSONAL_DATA_PATH + 'Cardiovascular_hospitalizations_due_to_PM2_5__age_40__.csv')
resp_hosp_pm25 = pd.read_csv(PERSONAL_DATA_PATH + 'Respiratory_hospitalizations_due_to_PM2_5__age_20__.csv')

# Asthma Data
asthma_ed_pm25 = pd.read_csv(PERSONAL_DATA_PATH + 'Asthma_emergency_department_visits_due_to_PM2_5.csv')
asthma_ed_ozone = pd.read_csv(PERSONAL_DATA_PATH + 'Asthma_emergency_departments_visits_due_to_Ozone.csv')
asthma_hosp_ozone = pd.read_csv(PERSONAL_DATA_PATH + 'Asthma_hospitalizations_due_to_Ozone.csv')

