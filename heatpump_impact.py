import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Constants (same as before)
BALANCE_TEMPERATURE = 15  # degrees Celsius
AIRFLOW_RATE = 3000 / 3600  # m³/h converted to m³/s
AIR_DENSITY = 1.2  # kg/m³
SPECIFIC_HEAT_CAPACITY_DRY_AIR = 1005  # J/kg·K
LATENT_HEAT_VAPORIZATION = 2500 * 1e3  # J/kg
SPECIFIC_HEAT_WATER_VAPOR = 1860  # J/kg·K

# Functions (adjusted as per corrections)
def calculate_humidity_ratio(temperature, rh):
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = (rh / 100) * p_sat  # Vapor pressure in kPa
    X = 0.622 * p_vapor / (101.3 - p_vapor)  # Humidity ratio (kg/kg)
    return X

def calculate_specific_heat_moist_air(X):
    return SPECIFIC_HEAT_CAPACITY_DRY_AIR + X * SPECIFIC_HEAT_WATER_VAPOR

def calculate_temperature_drop(heat_extracted_kwh, airflow_rate, rho, cp_moist):
    heat_extracted_joules = heat_extracted_kwh * 3.6 * 1e6
    mass_per_day = airflow_rate * rho * 86400  # seconds in a day
    delta_t = heat_extracted_joules / (mass_per_day * cp_moist)
    return delta_t

def calculate_dew_point(temperature, rh):
    a = 17.27
    b = 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

# Read CSV file
df_meteo = pd.read_csv('data/Meteo_MSR.csv')
df = df_meteo.rename(columns={
    'MONT-SUR-ROLLE - Humidité moy. (%)': 'RH',
    'MONT-SUR-ROLLE - Température moy. +2 m (°C)': 'Temperature'
})

# Calculate humidity ratio and specific heat
df['Humidity_Ratio'] = df.apply(
    lambda row: calculate_humidity_ratio(row['Temperature'], row['RH']), axis=1
)
df['Specific_Heat_Moist_Air'] = df['Humidity_Ratio'].apply(calculate_specific_heat_moist_air)

# Assuming total heating energy for the season (in kWh)
total_energy_kwh = 19000  # Modify as per your data

# Correctly calculate heat extracted per day
df['Temperature_Difference'] = BALANCE_TEMPERATURE - df['Temperature']
df['Temperature_Difference'] = df['Temperature_Difference'].clip(lower=0)
total_temperature_diff = df['Temperature_Difference'].sum()
df['Heat_Extracted_kWh'] = total_energy_kwh * df['Temperature_Difference'] / total_temperature_diff

# Correctly calculate temperature drop per day
df['Temperature_Drop'] = df.apply(
    lambda row: calculate_temperature_drop(
        row['Heat_Extracted_kWh'], AIRFLOW_RATE, AIR_DENSITY, row['Specific_Heat_Moist_Air']
    ), axis=1
)

# Calculate dew point and check for condensation
df['Dew_Point'] = df.apply(lambda row: calculate_dew_point(row['Temperature'], row['RH']), axis=1)
df['Temperature_After_Drop'] = df['Temperature'] - df['Temperature_Drop']
df['Condensation'] = df['Temperature_After_Drop'] < df['Dew_Point']

# Save to a CSV if needed
df.to_csv('heat_pump_impact_results.csv', index=False)

def plot_results_with_rh(dates, exiting_temps, condensed_water, outside_temps, outside_rh):
    """
        Plots the exiting air temperature and condensed water over time with relative humidity for comparison.

        Parameters:
        - dates (pd.Series): Series of dates.
        - exiting_temps (list): List of exiting air temperatures.
        - condensed_water (list): List of condensed water amounts.
        - outside_temps (pd.Series): Series of outside temperatures.
        - outside_rh (pd.Series): Series of relative humidity percentages.

        Returns:
        - None
        """
    dates = pd.to_datetime(dates, format='%d.%m.%Y')

    plt.figure(figsize=(12, 6))

    # Plot Exiting Air Temperature and Outside Temperature
    plt.subplot(2, 1, 1)
    plt.plot(dates, exiting_temps, label="Temp Air PAC(°C)", color='r')
    plt.plot(dates, outside_temps, label="Temp d'Air(°C)", color='g', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature d'air PAC et Temperature d'air pour saison de chauffage 2023-2024")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.grid(True)
    plt.legend()

    # Plot Condensed Water with RH on Twin Axes
    ax1 = plt.subplot(2, 1, 2)
    ax1.plot(dates, condensed_water, label="Eau condensé  (litres)", color='b')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Eau condensé (litres/heure)")

    # Set x-axis to show month names
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(True)

    # Create a second y-axis for RH
    ax2 = ax1.twinx()
    ax2.plot(dates, outside_rh, label="Humidité Relative (%)", color='orange', linestyle='--')
    ax2.set_ylabel("Humidité Relative  (%)")

    # Combine legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Production d'eau condensé et Humidité Relative pour saison de chauffage 2023-2024")

    plt.tight_layout()
    plt.show()


plot_results_with_rh(df['Date'], df['Temperature_After_Drop'], df['Temperature_Difference'], df['Temperature']
                 ,df['RH'])