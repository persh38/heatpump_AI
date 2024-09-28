import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
BALANCE_TEMPERATURE = 15  # Assumed balance point temperature in degrees Celsius
AIRFLOW_RATE = 3000 / 3600  # Convert m³/h to m³/s
FAN_DIAMETER = 0.75  # in meters
AIR_DENSITY = 1.2  # kg/m³, approximate density of air
SPECIFIC_HEAT_CAPACITY_DRY_AIR = 1005  # J/kg·K for dry air
LATENT_HEAT_VAPORIZATION = 2500 * 1e3  # Latent heat of condensation in J/kg
SPECIFIC_HEAT_WATER_VAPOR = 1860  # J/kg·K


def clean_column_names(dataframe):
    """
    Removes the string 'MONT-SUR-ROLLE -' from all column names in the DataFrame.

    Parameters:
    - dataframe: The DataFrame to clean column names for.

    Returns:
    - dataframe: The DataFrame with updated column names.
    """
    # dataframe.columns = dataframe.columns.str.replace('MONT-SUR-ROLLE -', '').str.strip()
    dataframe = dataframe.rename(columns={'MONT-SUR-ROLLE - Humidité moy. (%)': 'RH', 'MONT-SUR-ROLLE - Température moy. +2 m (°C)': 'Temperature'})
    return dataframe


# Function to calculate humidity ratio X
def calculate_humidity_ratio(temperature, rh):
    # Saturation vapor pressure at temperature (approximation using Tetens formula)
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = (rh / 100) * p_sat  # Vapor pressure in kPa
    X = 0.622 * p_vapor / (101.3 - p_vapor)  # Humidity ratio (mass of water vapor per mass of dry air)
    return X

# Function to calculate specific heat of moist air
def calculate_specific_heat_moist_air(X):
    return SPECIFIC_HEAT_CAPACITY_DRY_AIR + X * SPECIFIC_HEAT_WATER_VAPOR

# Function to calculate heat extracted per day (in kWh)
def calculate_heat_extracted_per_day(total_energy, temperature, balance_temp):
    total_temperature_diff = np.sum(balance_temp - temperature)
    return total_energy * (balance_temp - temperature) / total_temperature_diff

# Function to calculate temperature drop
def calculate_temperature_drop(heat_extracted_kwh, airflow_rate, rho, cp_moist):
    # Convert kWh to joules
    heat_extracted_joules = heat_extracted_kwh * 3.6 * 1e6
    delta_t = heat_extracted_joules / (airflow_rate * rho * cp_moist)
    return delta_t

# Function to calculate dew point (approximate)
def calculate_dew_point(temperature, rh):
    a = 17.27
    b = 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

# Read CSV file containing temperature and RH data
outside_site_data = pd.read_csv('data/Meteo_MSR .csv')  # Modify to your file path
df = clean_column_names(outside_site_data)
df['Humidity_Ratio'] = df.apply(lambda row: calculate_humidity_ratio(row['Temperature'], row['RH']), axis=1)
df['Specific_Heat_Moist_Air'] = df['Humidity_Ratio'].apply(calculate_specific_heat_moist_air)

# Assuming total heating energy for the season (in kWh)
total_energy_kwh = 20000  # Placeholder value, modify according to your data

# Calculate heat extracted per day
df['Heat_Extracted_kWh'] = df['Temperature'].apply(lambda temp: calculate_heat_extracted_per_day(total_energy_kwh, temp, BALANCE_TEMPERATURE))

# Calculate temperature drop per day
df['Temperature_Drop'] = df.apply(lambda row: calculate_temperature_drop(row['Heat_Extracted_kWh'], AIRFLOW_RATE, AIR_DENSITY, row['Specific_Heat_Moist_Air']), axis=1)

# Check if condensation occurs and adjust heat extracted for condensation
df['Dew_Point'] = df.apply(lambda row: calculate_dew_point(row['Temperature'], row['RH']), axis=1)
df['Condensation'] = df['Temperature'] - df['Temperature_Drop'] < df['Dew_Point']

# Adjust temperature drop if condensation occurs
df['Latent_Heat_Condensation'] = df.apply(lambda row: LATENT_HEAT_VAPORIZATION * row['Humidity_Ratio'] if row['Condensation'] else 0, axis=1)
df['Adjusted_Temperature_Drop'] = df.apply(lambda row: calculate_temperature_drop(row['Heat_Extracted_kWh'] + row['Latent_Heat_Condensation'], AIRFLOW_RATE, AIR_DENSITY, row['Specific_Heat_Moist_Air']), axis=1)

# Save to a CSV if needed
df.to_csv('heat_pump_impact_results.csv', index=False)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Temperature_Drop'], label='Temperature Drop')
plt.plot(df['Date'], df['Adjusted_Temperature_Drop'], label='Adjusted Temperature Drop (Condensation)', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Temperature Drop (°C)')
plt.title('Daily Temperature Drop of Air Exhausted by Heat Pump')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
