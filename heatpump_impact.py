import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Constants
DESIRED_INDOOR_TEMPERATURE = 21  # degrees Celsius
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

# Function to calculate temperature drop considering latent heat of condensation
def calculate_adjusted_temperature_drop(heat_extracted_kwh, latent_heat_joules, airflow_rate, rho, cp_moist):
    # Convert kWh to joules
    heat_extracted_joules = heat_extracted_kwh * 3.6 * 1e6
    # Subtract latent heat released by condensation from heat extracted
    net_heat_extracted_joules = heat_extracted_joules - latent_heat_joules
    # Ensure net heat is non-negative
    net_heat_extracted_joules = max(0, net_heat_extracted_joules)
    # Total mass of air per day
    mass_per_day = airflow_rate * rho * 86400  # seconds in a day
    # Calculate the temperature drop
    delta_t = net_heat_extracted_joules / (mass_per_day * cp_moist)
    return delta_t

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

# Calculate the temperature difference for each day (Indoor temperature - Outdoor temperature)
df['Temperature_Difference'] = DESIRED_INDOOR_TEMPERATURE - df['Temperature']

# Sum of all temperature differences over the season (for proportional energy distribution)
total_temperature_diff = df['Temperature_Difference'].sum()

# Calculate daily heat extracted proportional to the temperature difference
df['Heat_Extracted_kWh'] = total_energy_kwh * df['Temperature_Difference'] / total_temperature_diff

# Calculate temperature drop per day
df['Temperature_Drop'] = df.apply(
    lambda row: calculate_adjusted_temperature_drop(
        row['Heat_Extracted_kWh'], 0, AIRFLOW_RATE, AIR_DENSITY, row['Specific_Heat_Moist_Air']
    ), axis=1
)

# Calculate temperature after drop
df['Temperature_After_Drop'] = df['Temperature'] - df['Temperature_Drop']

# Calculate dew point and check for condensation
df['Dew_Point'] = df.apply(lambda row: calculate_dew_point(row['Temperature'], row['RH']), axis=1)
df['Condensation'] = df['Temperature_After_Drop'] < df['Dew_Point']

# Calculate saturated humidity ratio at cooled temperature for days with condensation
df['X_saturated'] = df.apply(
    lambda row: calculate_saturation_humidity_ratio(row['Temperature_After_Drop']) if row['Condensation'] else row['Humidity_Ratio'], axis=1
)

# Calculate the difference in humidity ratios (ΔX)
df['delta_X'] = df.apply(
    lambda row: row['Humidity_Ratio'] - row['X_saturated'] if row['Condensation'] else 0, axis=1
)

# Calculate mass flow rates
mass_flow_rate_moist_air = AIRFLOW_RATE * AIR_DENSITY  # kg/s

# Calculate mass flow rate of dry air for each day
df['Mass_Flow_Rate_Dry_Air'] = mass_flow_rate_moist_air / (1 + df['Humidity_Ratio'])  # kg/s

# Total mass of dry air per day
df['Total_Mass_Dry_Air_per_Day'] = df['Mass_Flow_Rate_Dry_Air'] * 86400  # kg/day

# Calculate mass of condensed water per day (only when condensation occurs)
df['Condensed_Water_kg_per_day'] = df.apply(
    lambda row: row['Total_Mass_Dry_Air_per_Day'] * row['delta_X'] if row['Condensation'] else 0, axis=1
)

# Convert to liters per day (assuming 1 kg of water = 1 liter)
df['Condensed_Water_Liters_per_day'] = df['Condensed_Water_kg_per_day']  # liters/day

# Calculate latent heat released by condensation (in joules)
df['Latent_Heat_Released_Joules'] = df['Condensed_Water_kg_per_day'] * LATENT_HEAT_VAPORIZATION

# Adjust temperature drop by accounting for latent heat released
df['Adjusted_Temperature_Drop'] = df.apply(
    lambda row: calculate_adjusted_temperature_drop(
        row['Heat_Extracted_kWh'],
        row['Latent_Heat_Released_Joules'],
        AIRFLOW_RATE,
        AIR_DENSITY,
        row['Specific_Heat_Moist_Air']
    ), axis=1
)

# Update temperature after drop with adjusted temperature drop
df['Temperature_After_Adjusted_Drop'] = df['Temperature'] - df['Adjusted_Temperature_Drop']

# Convert condensed water to liters per day (assuming 1 kg of water = 1 liter)
df['Condensed_Water_Liters_per_hour'] = df['Condensed_Water_kg_per_day']/24

# Output the results
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Heat Pump Impact Analysis with Condensation", dataframe=df)

# Save to a CSV if needed
df.to_csv('heat_pump_impact_with_condensation_results.csv', index=False)

OPERATING_HOURS_PER_DAY = 24
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


def plot_condensation_histogram(condensed_water_per_hour):
    """
    Plots a histogram showing the distribution of daily condensed water production in liters per hour.

    Parameters:
    - condensed_water_per_hour (list): List of condensed water production per hour for each day.

    Returns:
    - None
    """
    total_condensed_water = sum(condensed_water_per_hour)* OPERATING_HOURS_PER_DAY

    plt.figure(figsize=(8, 6))

    # Create a histogram with bin width of 1 liter
    plt.hist(condensed_water_per_hour, bins=range(0, int(max(condensed_water_per_hour)) + 2),
             edgecolor='black', align='left')

    # Add labels and title
    plt.xlabel('Eau Condensé (litres/heure)')
    plt.ylabel('Nombres de jours')
    plt.title("Nombres de jours par Production d'Eau condenseé (liters/hour)")

    # Add text to display total condensed water per year
    plt.text(0.95, 0.95, f"Eau condensê pour periode de chaufage : {total_condensed_water:.0f} liters",
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=10, color='blue')

    plt.grid(True)
    plt.show()

plot_results_with_rh(df['Date'], df['Temperature_After_Adjusted_Drop'], df['Condensed_Water_Liters_per_hour'], df['Temperature']
                 ,df['RH'])

condensed_water = df['Condensed_Water_Liters_per_hour']
plot_condensation_histogram(condensed_water)
print(
    f"Somme total d'eau condensê pour periode de chaufage {sum(condensed_water) * OPERATING_HOURS_PER_DAY:.0f} Litres")