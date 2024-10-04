import pandas as pd
import numpy as np

# Constants
DESIRED_INDOOR_TEMPERATURE = 21  # degrees Celsius
COP = 4.0  # Coefficient of Performance of the heat pump
AIRFLOW_RATE = 3000 / 3600  # m³/h converted to m³/s
AIR_DENSITY = 1.2  # kg/m³
SPECIFIC_HEAT_CAPACITY_DRY_AIR = 1005  # J/kg·K
LATENT_HEAT_VAPORIZATION = 2500 * 1e3  # J/kg
SPECIFIC_HEAT_WATER_VAPOR = 1860  # J/kg·K

# Function to calculate humidity ratio X
def calculate_humidity_ratio(temperature, rh):
    # Saturation vapor pressure at temperature (approximation using Tetens formula)
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = (rh / 100.0) * p_sat  # Vapor pressure in kPa
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
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = p_sat  # at 100% RH
    X_saturated = 0.622 * p_vapor / (101.3 - p_vapor)  # kg water/kg dry air
    return X_saturated

# Function to calculate adjusted temperature drop
def calculate_adjusted_temperature_drop(heat_extracted_kwh, latent_heat_joules, airflow_rate, rho, cp_moist):
    # Convert kWh to joules
    heat_extracted_joules = heat_extracted_kwh * 3.6e6  # 1 kWh = 3.6e6 J
    # Subtract latent heat released by condensation from heat extracted
    net_heat_extracted_joules = heat_extracted_joules - latent_heat_joules
    # Ensure net heat is non-negative
    net_heat_extracted_joules = max(0, net_heat_extracted_joules)
    # Total mass of air per day
    mass_per_day = airflow_rate * rho * 86400  # 86,400 seconds in a day
    # Calculate the temperature drop
    delta_t = net_heat_extracted_joules / (mass_per_day * cp_moist)
    return delta_t

# Read CSV file containing temperature and RH data
df = pd.read_csv('temperature_rh_data.csv')  # Modify to your file path

# Calculate humidity ratio and specific heat
df['Humidity_Ratio'] = df.apply(
    lambda row: calculate_humidity_ratio(row['Temperature'], row['RH']), axis=1
)
df['Specific_Heat_Moist_Air'] = df['Humidity_Ratio'].apply(calculate_specific_heat_moist_air)

# Assuming total heating energy for the season (in kWh)
total_heating_energy_kwh = 20000  # Total heating demand for the season

# Calculate the temperature difference for each day (Indoor temperature - Outdoor temperature)
df['Temperature_Difference'] = DESIRED_INDOOR_TEMPERATURE - df['Temperature']
df['Temperature_Difference'] = df['Temperature_Difference'].clip(lower=0)  # No heating if outdoor temp >= indoor temp

# Sum of all temperature differences over the season
total_temperature_diff = df['Temperature_Difference'].sum()

# Calculate daily heating energy demand proportional to the temperature difference
df['Heating_Energy_Demand_kWh'] = total_heating_energy_kwh * df['Temperature_Difference'] / total_temperature_diff

# Calculate the electrical energy input per day
df['Electrical_Energy_Input_kWh'] = df['Heating_Energy_Demand_kWh'] / COP

# Calculate the heat extracted from the air per day
df['Heat_Extracted_From_Air_kWh'] = df['Heating_Energy_Demand_kWh'] - df['Electrical_Energy_Input_kWh']

# Initially, we assume no latent heat released
df['Latent_Heat_Released_Joules'] = 0

# Calculate temperature drop per day (before accounting for latent heat)
df['Temperature_Drop'] = df.apply(
    lambda row: calculate_adjusted_temperature_drop(
        row['Heat_Extracted_From_Air_kWh'], 0, AIRFLOW_RATE, AIR_DENSITY, row['Specific_Heat_Moist_Air']
    ), axis=1
)

# Calculate temperature after drop
df['Temperature_After_Drop'] = df['Temperature'] - df['Temperature_Drop']

# Calculate dew point and check for condensation
df['Dew_Point'] = df.apply(
    lambda row: calculate_dew_point(row['Temperature'], row['RH']), axis=1
)
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

# Recalculate the adjusted temperature drop with latent heat included
df['Adjusted_Temperature_Drop'] = df.apply(
    lambda row: calculate_adjusted_temperature_drop(
        row['Heat_Extracted_From_Air_kWh'],
        row['Latent_Heat_Released_Joules'],
        AIRFLOW_RATE,
        AIR_DENSITY,
        row['Specific_Heat_Moist_Air']
    ), axis=1
)

# Update temperature after drop with adjusted temperature drop
df['Temperature_After_Adjusted_Drop'] = df['Temperature'] - df['Adjusted_Temperature_Drop']

# Output the results
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Heat Pump Impact Analysis with COP Included", dataframe=df)

# Save to a CSV if needed
df.to_csv('heat_pump_impact_with_cop_results.csv', index=False)
