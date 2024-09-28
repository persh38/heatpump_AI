
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
SPECIFIC_HEAT_CAPACITY_DRY_AIR = 1005  # J/kg·K for dry air
SPECIFIC_HEAT_WATER_VAPOR = 1860  # J/kg·K
LATENT_HEAT_VAPORIZATION = 2500 * 1e3  # Latent heat of condensation in J/kg
AIR_DENSITY = 1.2  # kg/m³, approximate density of air
AIRFLOW_RATE = 3000 / 3600  # Convert m³/h to m³/s (3000 m³/h is airflow rate)
BALANCE_TEMPERATURE = 15  # Balance temperature in °C, assumed
TOTAL_ENERGY_KWH = 20000  # Total energy required for heating season in kWh

# Function to calculate heat extracted per day (in kWh)
def calculate_heat_extracted_per_day(total_energy, temperature, balance_temp):
    total_temperature_diff = (balance_temp - df_meteo_cleaned['Temperature']).sum()
    daily_heat = total_energy * (balance_temp - temperature) / total_temperature_diff
    return daily_heat

# Function to calculate humidity ratio X
def calculate_humidity_ratio(temperature, rh):
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = (rh / 100) * p_sat  # Vapor pressure in kPa
    X = 0.622 * p_vapor / (101.3 - p_vapor)  # Humidity ratio (mass of water vapor per mass of dry air)
    return X

# Function to calculate specific heat of moist air
def calculate_specific_heat_moist_air(X):
    return SPECIFIC_HEAT_CAPACITY_DRY_AIR + X * SPECIFIC_HEAT_WATER_VAPOR

# Function to calculate temperature drop
def calculate_temperature_drop(heat_extracted_kwh, airflow_rate, rho, cp_moist):
    heat_extracted_joules = heat_extracted_kwh * 3.6e6  # Convert kWh to joules
    delta_t = heat_extracted_joules / (airflow_rate * rho * cp_moist)
    return delta_t

# Read CSV file containing temperature and RH data
df_meteo_cleaned = pd.read_csv('data/Meteo_MSR .csv')
df_meteo_cleaned = df_meteo_cleaned.rename(columns={
    'MONT-SUR-ROLLE - Humidité moy. (%)': 'RH',
    'MONT-SUR-ROLLE - Température moy. +2 m (°C)': 'Temperature'
})

# Calculate daily heat extracted
df_meteo_cleaned['Heat_Extracted_kWh'] = df_meteo_cleaned['Temperature'].apply(
    lambda temp: calculate_heat_extracted_per_day(TOTAL_ENERGY_KWH, temp, BALANCE_TEMPERATURE))

# Recalculating humidity ratio and specific heat of moist air
df_meteo_cleaned['Humidity_Ratio'] = df_meteo_cleaned.apply(
    lambda row: calculate_humidity_ratio(row['Temperature'], row['RH']), axis=1)
df_meteo_cleaned['Specific_Heat_Moist_Air'] = df_meteo_cleaned['Humidity_Ratio'].apply(calculate_specific_heat_moist_air)

# Calculate the correct temperature drop per day
df_meteo_cleaned['Temperature_Drop'] = df_meteo_cleaned.apply(
    lambda row: calculate_temperature_drop(row['Heat_Extracted_kWh'], AIRFLOW_RATE, AIR_DENSITY, row['Specific_Heat_Moist_Air']), axis=1)

# Plot the temperature drop over time
plt.figure(figsize=(10, 6))
plt.plot(df_meteo_cleaned['Date'], df_meteo_cleaned['Temperature_Drop'], label='Temperature Drop')

plt.xlabel('Date')
plt.ylabel('Temperature Drop (°C)')
plt.title('Daily Temperature Drop of Air Exhausted by Heat Pump')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
