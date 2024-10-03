import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Constants
BALANCE_TEMPERATURE = 15  # degrees Celsius
AIRFLOW_RATE = 3000 / 3600  # m³/h converted to m³/s
AIR_DENSITY = 1.2  # kg/m³
SPECIFIC_HEAT_CAPACITY_DRY_AIR = 1005  # J/kg·K
LATENT_HEAT_VAPORIZATION = 2500 * 1e3  # J/kg
# LATENT_HEAT_VAPORIZATION = 0
SPECIFIC_HEAT_WATER_VAPOR = 1860  # J/kg·K

# Function to calculate humidity ratio X
def calculate_humidity_ratio(temperature, rh):
    # Saturation vapor pressure at temperature (approximation using Tetens formula)
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = (rh / 100) * p_sat  # Vapor pressure in kPa
    X = 0.622 * p_vapor / (101.3 - p_vapor)  # Humidity ratio (kg water/kg dry air)
    return X

# Function to calculate specific heat of moist air
def calculate_specific_heat_moist_air(X):
    return SPECIFIC_HEAT_CAPACITY_DRY_AIR + X * SPECIFIC_HEAT_WATER_VAPOR

# Function to calculate dew point (approximate)
def calculate_dew_point(temperature, rh):
    a = 17.27
    b = 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

# Function to calculate saturation humidity ratio at a given temperature
def calculate_saturation_humidity_ratio(temperature):
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # kPa
    p_vapor = p_sat  # at 100% RH
    X_saturated = 0.622 * p_vapor / (101.3 - p_vapor)  # kg water/kg dry air
    return X_saturated

# Function to calculate adjusted temperature drop iteratively
def calculate_adjusted_temperature_drop_iterative(heat_extracted_kwh, airflow_rate, rho, initial_cp_moist, initial_temp, initial_rh):
    # Convert kWh to joules
    total_heat_extracted_joules = heat_extracted_kwh * 3.6 * 1e6
    # Total mass of air per day
    mass_per_day = airflow_rate * rho * 86400  # seconds in a day

    # Initial guess for temperature drop
    delta_t = total_heat_extracted_joules / (mass_per_day * initial_cp_moist)

    # Initial values
    X_initial = calculate_humidity_ratio(initial_temp, initial_rh)
    cp_moist = calculate_specific_heat_moist_air(X_initial)

    # Iterative calculation
    for _ in range(10):  # Limit to 10 iterations to prevent infinite loops
        cooled_temp = initial_temp - delta_t
        dew_point = calculate_dew_point(initial_temp, initial_rh)

        # Check for condensation
        if cooled_temp < dew_point:
            # Condensation occurs
            X_saturated = calculate_saturation_humidity_ratio(cooled_temp)
            delta_X = X_initial - X_saturated
            if delta_X < 0:
                delta_X = 0  # Ensure non-negative

            # Calculate latent heat released
            mass_flow_rate_moist_air = airflow_rate * rho  # kg/s
            mass_flow_rate_dry_air = mass_flow_rate_moist_air / (1 + X_initial)  # kg/s
            total_mass_dry_air_per_day = mass_flow_rate_dry_air * 86400  # kg/day
            condensed_water_kg_per_day = total_mass_dry_air_per_day * delta_X
            latent_heat_joules = condensed_water_kg_per_day * LATENT_HEAT_VAPORIZATION

            # Subtract latent heat from total heat extracted to find sensible heat required
            sensible_heat_joules = total_heat_extracted_joules - latent_heat_joules
            if sensible_heat_joules < 0:
                # Cannot extract more latent heat than total heat extracted
                latent_heat_joules = total_heat_extracted_joules
                sensible_heat_joules = 0
                delta_t = 0
                break
        else:
            # No condensation
            latent_heat_joules = 0
            condensed_water_kg_per_day = 0
            sensible_heat_joules = total_heat_extracted_joules

        # Recalculate cp_moist if necessary
        cp_moist = calculate_specific_heat_moist_air(X_initial)

        # Update temperature drop
        new_delta_t = sensible_heat_joules / (mass_per_day * cp_moist)

        # Check for convergence
        if abs(new_delta_t - delta_t) < 0.01:  # Convergence criterion in degrees Celsius
            delta_t = new_delta_t
            break
        delta_t = new_delta_t  # Update for next iteration

    # Calculate final cooled temperature
    cooled_temp = initial_temp - delta_t

    return delta_t, latent_heat_joules, condensed_water_kg_per_day, cooled_temp

# Read CSV file containing temperature and RH data
df_meteo = pd.read_csv('data/Meteo_MSR.csv')
df = df_meteo.rename(columns={
    'MONT-SUR-ROLLE - Humidité moy. (%)': 'RH',
    'MONT-SUR-ROLLE - Température moy. +2 m (°C)': 'Temperature'
})

# Calculate humidity ratio and specific heat
df['Humidity_Ratio'] = df.apply(lambda row: calculate_humidity_ratio(row['Temperature'], row['RH']), axis=1)
df['Specific_Heat_Moist_Air'] = df['Humidity_Ratio'].apply(calculate_specific_heat_moist_air)

# Assuming total heating energy for the season (in kWh)
total_energy_kwh = 20000  # Placeholder value, modify according to your data

# Correctly calculate heat extracted per day
df['Temperature_Difference'] = BALANCE_TEMPERATURE - df['Temperature']
df['Temperature_Difference'] = df['Temperature_Difference'].clip(lower=0)
total_temperature_diff = df['Temperature_Difference'].sum()
df['Heat_Extracted_kWh'] = total_energy_kwh * df['Temperature_Difference'] / total_temperature_diff

# Initialize lists to store results
adjusted_temperature_drops = []
latent_heats = []
condensed_water_amounts = []
cooled_temperatures = []

# Loop over each row for iterative calculation
for index, row in df.iterrows():
    delta_t, latent_heat_joules, condensed_water_kg_per_day, cooled_temp = calculate_adjusted_temperature_drop_iterative(
        row['Heat_Extracted_kWh'],
        AIRFLOW_RATE,
        AIR_DENSITY,
        row['Specific_Heat_Moist_Air'],
        row['Temperature'],
        row['RH']
    )
    adjusted_temperature_drops.append(delta_t)
    latent_heats.append(latent_heat_joules)
    condensed_water_amounts.append(condensed_water_kg_per_day)
    cooled_temperatures.append(cooled_temp)

# Assign the new values to the dataframe
df['Adjusted_Temperature_Drop'] = adjusted_temperature_drops
df['Latent_Heat_Released_Joules'] = latent_heats
df['Condensed_Water_kg_per_day'] = condensed_water_amounts
df['Temperature_After_Adjusted_Drop'] = cooled_temperatures

# Update 'Condensation' based on new cooled temperatures
df['Dew_Point'] = df.apply(lambda row: calculate_dew_point(row['Temperature'], row['RH']), axis=1)
df['Condensation'] = df['Temperature_After_Adjusted_Drop'] < df['Dew_Point']

# Convert condensed water to liters per day (assuming 1 kg of water = 1 liter)
df['Condensed_Water_Liters_per_hour'] = df['Condensed_Water_kg_per_day']/24

# Output the results
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Heat Pump Impact Analysis with Condensation", dataframe=df)

# Save to a CSV if needed
df.to_csv('heat_pump_impact_with_condensation_results.csv', index=False)

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
    plt.title("Temperature d'air PAC et Temperature d'air pour saison de chauffage 2023-2024. Code géneré par chatGPT o1-preview")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.grid(True)
    plt.legend()

    # Plot Condensed Water with RH on Twin Axes
    ax1 = plt.subplot(2, 1, 2)
    ax1.plot(dates, condensed_water, label="Eau condensé  (litres/h)", color='b')
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

    plt.title("Production d'eau condensé et Humidité Relative pour saison de chauffage 2023-2024  Code géneré par chatGPT o1-preview")

    plt.tight_layout()
    plt.show()


plot_results_with_rh(df['Date'], df['Temperature_After_Adjusted_Drop'], df['Condensed_Water_Liters_per_hour'], df['Temperature']
                 ,df['RH'])