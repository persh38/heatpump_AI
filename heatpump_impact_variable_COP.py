import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Constants
DESIRED_INDOOR_TEMPERATURE = 21  # degrees Celsius
AIRFLOW_RATE = 3000 / 3600        # m³/h converted to m³/s
AIR_DENSITY = 1.2                 # kg/m³
SPECIFIC_HEAT_CAPACITY_DRY_AIR = 1005    # J/kg·K
SPECIFIC_HEAT_WATER_VAPOR = 1860         # J/kg·K
LATENT_HEAT_VAPORIZATION = 2500e3        # J/kg
OPERATING_HOURS_PER_DAY = 24

# Function Definitions

def calculate_humidity_ratio(temperature, rh):
    """
    Calculate the humidity ratio (kg water/kg dry air) based on temperature and relative humidity.
    """
    # Saturation vapor pressure at temperature (Tetens formula)
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = (rh / 100.0) * p_sat  # Vapor pressure in kPa
    humidity_ratio = 0.622 * p_vapor / (101.3 - p_vapor)  # kg water/kg dry air
    return humidity_ratio

def calculate_specific_heat_moist_air(humidity_ratio):
    """
    Calculate the specific heat capacity of moist air (J/kg·K).
    """
    return SPECIFIC_HEAT_CAPACITY_DRY_AIR + humidity_ratio * SPECIFIC_HEAT_WATER_VAPOR

def calculate_dew_point(temperature, rh):
    """
    Calculate the dew point temperature (°C) based on temperature and relative humidity.
    """
    a = 17.27
    b = 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

def calculate_saturation_humidity_ratio(temperature):
    """
    Calculate the saturation humidity ratio at a given temperature (kg water/kg dry air).
    """
    # Saturation vapor pressure at temperature
    p_sat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # in kPa
    p_vapor = p_sat  # at 100% RH
    saturation_humidity_ratio = 0.622 * p_vapor / (101.3 - p_vapor)  # kg water/kg dry air
    return saturation_humidity_ratio

def calculate_adjusted_temperature_drop(heat_extracted_kwh, latent_heat_joules, airflow_rate, rho, cp_moist):
    """
    Calculate the adjusted temperature drop (°C) considering latent heat of condensation.
    """
    # Convert kWh to joules
    heat_extracted_joules = heat_extracted_kwh * 3.6e6  # 1 kWh = 3.6e6 J
    # Adjust heat extracted by subtracting the latent heat released
    net_heat_extracted_joules = heat_extracted_joules - latent_heat_joules
    # Ensure net heat extracted is non-negative
    net_heat_extracted_joules = max(0, net_heat_extracted_joules)
    # Total mass of air per day (kg/day)
    mass_per_day = airflow_rate * rho * 86400  # seconds in a day
    # Calculate the temperature drop (°C)
    delta_t = net_heat_extracted_joules / (mass_per_day * cp_moist)
    return delta_t

def plot_results_with_rh(dates, exiting_temps, condensed_water, outside_temps, outside_rh, electricity, cop):
    """
    Plot the exiting air temperature, condensed water, outside temperature, relative humidity,
    electricity consumption, and COP over time on separate figures.
    """
    dates = pd.to_datetime(dates, format='%d.%m.%Y')

    # First Plot: Exiting Air Temperature and Outside Temperature
    plt.figure(figsize=(10, 6))
    plt.plot(dates, exiting_temps, label="Temp Air PAC (°C)", color='r')
    plt.plot(dates, outside_temps, label="Temp d'Air (°C)", color='g', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Température de l'air exit PAC et Température de l'air pour la saison de chauffage")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Second Plot: Condensed Water with RH on Twin Axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(dates, condensed_water, label="Eau condensée (litres/h)", color='b')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Eau condensée (litres/heure)", color='b')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(True)

    # Create a second y-axis for RH
    ax2 = ax1.twinx()
    ax2.plot(dates, outside_rh, label="Humidité Relative (%)", color='orange', linestyle='--')
    ax2.set_ylabel("Humidité Relative (%)", color='orange')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title("Production d'eau condensée et Humidité Relative pour la saison de chauffage")
    plt.tight_layout()
    plt.show()

    # Third Plot: Electricity Consumption and temp Air
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(dates, electricity, label="Consommation Électrique (kWh)", color='b')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Consommation Électrique (kWh)", color='b')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(True)

    # Create a second y-axis for COP
    ax2 = ax1.twinx()
    ax2.plot(dates, outside_temps, label="Temp d'Air (°C)", color='y', linestyle='--')
    ax2.set_ylabel("Temp d'Air (°C)", color='g')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title("Consommation d'électricité et Temp d'Air pour la saison de chauffage")
    plt.tight_layout()
    plt.show()


def plot_condensation_histogram(condensed_water_per_hour):
    """
    Plot a histogram of the daily condensed water production in liters per hour.
    Excludes days with 0 liters of condensation and adds space between the bins.
    """
    # Exclude days with 0 condensed water
    condensed_water_nonzero = [x for x in condensed_water_per_hour if x > 0]
    total_condensed_water = sum(condensed_water_nonzero) * OPERATING_HOURS_PER_DAY
    total_days = len(condensed_water_nonzero)

    plt.figure(figsize=(8, 6))

    # Define the bin edges
    bins = np.arange(0.5, int(max(condensed_water_nonzero)) + 1.5, 1)

    # Count occurrences in each bin
    counts, bin_edges = np.histogram(condensed_water_nonzero, bins=bins)

    # Create bar chart with space between the bars
    bin_centers = bin_edges[:-1] + 0.5  # Centers of the bins
    plt.bar(bin_centers, counts, width=0.6, edgecolor='black')

    # Add labels and title
    plt.xlabel('Eau Condensée (litres/heure)')
    plt.ylabel('Nombre de jours')
    plt.title(f"Nombre de jours par production d'eau condensée (litres/heure) \nJournés avec Condensation. Total: {total_days}")

    # Add text to display total condensed water for the heating period
    plt.text(0.95, 0.95, f"Eau condensée pour la période de chauffage : {total_condensed_water:.0f} litres",
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=10, color='blue')

    plt.grid(True)
    plt.show()

def plot_scatter_vs_temperature(
    df,
    y_column,
    y_label,
    title,
    color,
    label,
    y2_column=None,
    y2_label=None,
    y2_color='red',
    y2_label_in_legend=None
):
    """
    Plots a scatter plot of the specified y_column against temperature.
    Optionally plots a second y_column on a secondary y-axis.

    Parameters:
    - df: DataFrame containing the data.
    - y_column: The column name in df for the y-axis data.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - color: Color of the scatter points.
    - label: Label for the data series (used in the legend).
    - y2_column: (Optional) The column name for the second y-axis data.
    - y2_label: (Optional) Label for the second y-axis.
    - y2_color: (Optional) Color for the second scatter points.
    - y2_label_in_legend: (Optional) Label for the second data series.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first scatter plot
    ax1.scatter(df['Temperature'], df[y_column], color=color, label=label)
    ax1.set_xlabel('Température Extérieure (°C)')
    ax1.set_ylabel(y_label, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    if y2_column is not None:
        # Create a secondary y-axis
        ax2 = ax1.twinx()
        ax2.scatter(df['Temperature'], df[y2_column], color=y2_color, label=y2_label_in_legend)
        ax2.set_ylabel(y2_label, color=y2_color)
        ax2.tick_params(axis='y', labelcolor=y2_color)

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    else:
        # Only one legend needed
        ax1.legend(loc='upper left')

    plt.title(title)
    plt.show()



def main():
    """
    Main function to perform heat pump impact analysis and plot results.
    """
    # Read the temperature and RH data
    df_meteo = pd.read_csv('data/Meteo_MSR.csv')
    df = df_meteo.rename(columns={
        'Date': 'Date',
        'MONT-SUR-ROLLE - Humidité moy. (%)': 'RH',
        'MONT-SUR-ROLLE - Température moy. +2 m (°C)': 'Temperature'
    })

    # Read the COP vs. temperature data
    cop_data = pd.read_csv('data/cop_vs_temperature.csv')

    # Interpolate COP values (Assuming 'COP_35C' column for 35°C water flow)
    cop_interp_func = interp1d(
        cop_data['Outdoor_Temperature'],
        cop_data['COP_35C'],
        kind='linear',
        fill_value="extrapolate"
    )

    # Calculate COP for each day based on outdoor temperature
    df['COP'] = df['Temperature'].apply(lambda temp: cop_interp_func(temp))

    # Calculate humidity ratio and specific heat
    df['Humidity_Ratio'] = df.apply(
        lambda row: calculate_humidity_ratio(row['Temperature'], row['RH']), axis=1
    )
    df['Specific_Heat_Moist_Air'] = df['Humidity_Ratio'].apply(calculate_specific_heat_moist_air)

    # Total heating energy for the season (in kWh)
    total_heating_energy_kwh = 19000  # Adjust based on your data

    # Calculate the temperature difference for each day
    df['Temperature_Difference'] = DESIRED_INDOOR_TEMPERATURE - df['Temperature']
    df['Temperature_Difference'] = df['Temperature_Difference'].clip(lower=0)

    # Sum of all temperature differences over the season
    total_temperature_diff = df['Temperature_Difference'].sum()

    # Calculate daily heating energy demand proportional to the temperature difference
    df['Heating_Energy_Demand_kWh'] = (
        total_heating_energy_kwh * df['Temperature_Difference'] / total_temperature_diff
    )

    # Calculate the electrical energy input per day using variable COP
    df['Electrical_Energy_Input_kWh'] = df['Heating_Energy_Demand_kWh'] / df['COP']

    # Calculate the heat extracted from the air per day
    df['Heat_Extracted_From_Air_kWh'] = (
        df['Heating_Energy_Demand_kWh'] - df['Electrical_Energy_Input_kWh'] * .9    # Assume 10% loss to non heating
    )

    # Calculate temperature drop per day (before accounting for latent heat)
    df['Temperature_Drop'] = df.apply(
        lambda row: calculate_adjusted_temperature_drop(
            row['Heat_Extracted_From_Air_kWh'],
            0,
            AIRFLOW_RATE,
            AIR_DENSITY,
            row['Specific_Heat_Moist_Air']
        ),
        axis=1
    )

    # Calculate temperature after drop
    df['Temperature_After_Drop'] = df['Temperature'] - df['Temperature_Drop']

    # Calculate dew point and check for condensation
    df['Dew_Point'] = df.apply(
        lambda row: calculate_dew_point(row['Temperature'], row['RH']),
        axis=1
    )
    df['Condensation'] = df['Temperature_After_Drop'] < df['Dew_Point']

    # Calculate saturated humidity ratio at cooled temperature for days with condensation
    df['X_saturated'] = df.apply(
        lambda row: calculate_saturation_humidity_ratio(row['Temperature_After_Drop'])
        if row['Condensation'] else row['Humidity_Ratio'],
        axis=1
    )

    # Calculate the difference in humidity ratios (ΔX)
    df['delta_X'] = df.apply(
        lambda row: row['Humidity_Ratio'] - row['X_saturated'] if row['Condensation'] else 0,
        axis=1
    )

    # Calculate mass flow rates
    mass_flow_rate_moist_air = AIRFLOW_RATE * AIR_DENSITY  # kg/s

    # Calculate mass flow rate of dry air for each day
    df['Mass_Flow_Rate_Dry_Air'] = (
        mass_flow_rate_moist_air / (1 + df['Humidity_Ratio'])
    )  # kg/s

    # Total mass of dry air per day
    df['Total_Mass_Dry_Air_per_Day'] = df['Mass_Flow_Rate_Dry_Air'] * 86400  # kg/day

    # Calculate mass of condensed water per day (only when condensation occurs)
    df['Condensed_Water_kg_per_day'] = df.apply(
        lambda row: row['Total_Mass_Dry_Air_per_Day'] * row['delta_X']
        if row['Condensation'] else 0,
        axis=1
    )

    # Convert to liters per day (assuming 1 kg of water = 1 liter)
    df['Condensed_Water_Liters_per_day'] = df['Condensed_Water_kg_per_day']

    # Calculate condensed water per hour
    df['Condensed_Water_Liters_per_hour'] = (
        df['Condensed_Water_Liters_per_day'] / OPERATING_HOURS_PER_DAY
    )

    # Calculate latent heat released by condensation (in joules)
    df['Latent_Heat_Released_Joules'] = (
        df['Condensed_Water_kg_per_day'] * LATENT_HEAT_VAPORIZATION
    )

    # Recalculate the adjusted temperature drop with latent heat included
    df['Adjusted_Temperature_Drop'] = df.apply(
        lambda row: calculate_adjusted_temperature_drop(
            row['Heat_Extracted_From_Air_kWh'],
            row['Latent_Heat_Released_Joules'],
            AIRFLOW_RATE,
            AIR_DENSITY,
            row['Specific_Heat_Moist_Air']
        ),
        axis=1
    )

    # Update temperature after drop with adjusted temperature drop
    df['Temperature_After_Adjusted_Drop'] = (
        df['Temperature'] - df['Adjusted_Temperature_Drop']
    )

    # Save to a CSV if needed
    df.to_csv('heat_pump_impact_with_variable_cop_results.csv', index=False)

    # Print total condensed water and electricity consumption
    total_condensed_water = df['Condensed_Water_Liters_per_hour'].sum() * OPERATING_HOURS_PER_DAY
    total_electricity_consumption = df['Electrical_Energy_Input_kWh'].sum()
    print(
        f"Somme totale d'eau condensée pour la période de chauffage: {total_condensed_water:.0f} litres"
    )
    print(
        f"Consommation totale d'électricité pour la période de chauffage: {total_electricity_consumption:.0f} kWh"
    )

    # Plot results
    plot_results_with_rh(
        df['Date'],
        df['Temperature_After_Adjusted_Drop'],
        df['Condensed_Water_Liters_per_hour'],
        df['Temperature'],
        df['RH'],
        df['Electrical_Energy_Input_kWh'],
        df['COP']
    )

    plot_condensation_histogram(df['Condensed_Water_Liters_per_hour'])

    plot_scatter_vs_temperature(
        df=df,
        y_column='COP',
        y_label='COP',
        title='COP en fonction de la Température Extérieure',
        color='blue',
        label='COP'
    )

    # Plot Heat Extracted From Air vs. Outdoor Temperature
    plot_scatter_vs_temperature(
        df=df,
        y_column='Electrical_Energy_Input_kWh',
        y_label='Electrical_Energy_Input_kWh',
        color='r',
        label='Electrical_Energy_Input_kWh',
        y2_column='Heat_Extracted_From_Air_kWh',
        y2_label='Chaleur Extraite de l\'Air (kWh)',
        title='Chaleur Extraite de l\'Air en fonction de la Température Extérieure',
        y2_color='green',
        y2_label_in_legend='Chaleur Extraite de l\'Air (kWh)',
        )



if __name__ == "__main__":
    main()
