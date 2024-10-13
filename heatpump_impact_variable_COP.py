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

def plot_time_series(
        df,
        date_column,
        y1_column,
        y1_label,
        y1_color='b',
        y1_linestyle='-',
        y1_axis_label=None,
        y2_column=None,
        y2_label=None,
        y2_color='r',
        y2_linestyle='--',
        y2_axis_label=None,
        title=None,
        legend_loc='upper left',
        date_format='%d.%m.%Y',
        secondary_y=True  # New parameter
):
    """
    Plots one or two time series data from a DataFrame against dates.

    Parameters:
    - df: pandas DataFrame containing the data.
    - date_column: Name of the column with date information.
    - y1_column: Name of the column for primary y-axis data.
    - y1_label: Label for y1 data series (used in legend).
    - y1_color: Color for y1 data series.
    - y1_linestyle: Linestyle for y1 data series.
    - y1_axis_label: Label for primary y-axis.
    - y2_column: (Optional) Name of the column for secondary y-axis data.
    - y2_label: (Optional) Label for y2 data series (used in legend).
    - y2_color: Color for y2 data series.
    - y2_linestyle: Linestyle for y2 data series.
    - y2_axis_label: (Optional) Label for secondary y-axis.
    - title: (Optional) Plot title.
    - legend_loc: (Optional) Location of legend.
    - date_format: (Optional) Date format for parsing dates.
    - secondary_y: (Optional) If True, uses a secondary y-axis for y2_column.
    """
    # Convert the date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first data series
    ax1.plot(df[date_column], df[y1_column], label=y1_label, color=y1_color, linestyle=y1_linestyle)
    if y1_axis_label:
        ax1.set_ylabel(y1_axis_label, color=y1_color)
        ax1.tick_params(axis='y', labelcolor=y1_color)
    ax1.set_xlabel('Date')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(True)

    if y2_column is not None:
        if secondary_y:
            # Plot the second data series on a secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(df[date_column], df[y2_column], label=y2_label, color=y2_color, linestyle=y2_linestyle)
            if y2_axis_label:
                ax2.set_ylabel(y2_axis_label, color=y2_color)
                ax2.tick_params(axis='y', labelcolor=y2_color)
            # Combine legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=legend_loc)
        else:
            # Plot the second data series on the primary y-axis
            ax1.plot(df[date_column], df[y2_column], label=y2_label, color=y2_color, linestyle=y2_linestyle)
            # Update the legend
            ax1.legend(loc=legend_loc)
    else:
        # Only one legend needed
        ax1.legend(loc=legend_loc)

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_condensation_histogram(condensed_water_per_hour):
    """
    Plot a histogram of the daily condensed water production in liters per hour.
    Excludes days with 0 liters of condensation and adds space between the bins.
    Clearly labels the bins to indicate the ranges they represent.
    """
    # Exclude days with 0 condensed water
    condensed_water_nonzero = [x for x in condensed_water_per_hour if x > 0]
    total_condensed_water = sum(condensed_water_nonzero) * OPERATING_HOURS_PER_DAY
    total_days = len(condensed_water_nonzero)

    plt.figure(figsize=(10, 6))

    # Define the bin edges starting from 0
    bins = np.arange(0, int(max(condensed_water_nonzero)) + 1.5, 1)

    # Count occurrences in each bin
    counts, bin_edges = np.histogram(condensed_water_nonzero, bins=bins)

    # Create bar chart with space between the bars
    bin_centers = bin_edges[:-1] + 0.5  # Centers of the bins
    plt.bar(bin_centers, counts, width=0.6, edgecolor='black', color='skyblue')

    # Create labels for the bins
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i+1]
        if lower_edge == 0:
            bin_label = "<1"
        else:
            bin_label = f"{int(lower_edge)}-{int(upper_edge)}"
        bin_labels.append(bin_label)

    # Set x-axis ticks and labels
    plt.xticks(bin_centers, bin_labels)

    # Add labels and title
    plt.xlabel('Eau Condensée (litres/heure)')
    plt.ylabel('Nombre de jours avec condensation')
    plt.title(f"Nombre de jours par production d'eau condensée (litres/heure) \nJournées avec Condensation. Total: {total_days}")

    # Add text to display total condensed water for the heating period
    plt.text(0.95, 0.95, f"Eau condensée pour la période de chauffage : {total_condensed_water:.0f} litres",
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=10, color='blue')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_scatter_vs_temperature(
    df,
    y_column,
    y_label,
    title,
    color,
    label,
    y2_column=None,
    y2_color='red',
    y2_label_in_legend=None,
    y3_column=None,
    y3_color='b',
    y3_label_in_legend=None
):
    """
    Plots a scatter plot of the specified y_column against temperature.
    Optionally plots a second y_column on the same y-axis.

    Parameters remain the same.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first scatter plot
    ax1.scatter(df['Temperature'], df[y_column], color=color, label=label)

    # Plot the second scatter plot on the same axis if provided
    if y2_column is not None:
        ax1.scatter(df['Temperature'], df[y2_column], color=y2_color, label=y2_label_in_legend)

    if y3_column is not None:
        ax1.scatter(df['Temperature'], df[y3_column], color=y3_color, label=y3_label_in_legend)

    ax1.set_xlabel('Température Extérieure (°C)')
    ax1.set_ylabel(y_label)
    ax1.legend()
    ax1.grid(True)

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
        df['Heating_Energy_Demand_kWh'] - df['Electrical_Energy_Input_kWh'] * 0.9  # Assume 10% loss to non-heating
    )

    # Initialize columns for iterative calculation
    # Initialize columns for iterative calculation
    df['Adjusted_Temperature_Drop'] = 0.0
    df['Temperature_After_Adjusted_Drop'] = df['Temperature']
    df['Condensed_Water_kg_per_day'] = 0.0
    df['Latent_Heat_Released_Joules'] = 0.0

    # Perform iterative calculation for each day
    for index, row in df.iterrows():
        # Initialize variables for iteration
        temperature = row['Temperature']
        rh = row['RH']
        humidity_ratio = row['Humidity_Ratio']
        specific_heat = row['Specific_Heat_Moist_Air']
        heat_extracted_kwh = row['Heat_Extracted_From_Air_kWh']
        mass_flow_rate_dry_air = AIRFLOW_RATE * AIR_DENSITY / (1 + humidity_ratio)
        total_mass_dry_air_per_day = mass_flow_rate_dry_air * 86400  # seconds in a day

        # Initialize guesses
        delta_t = 0.1  # Start with a small temperature drop
        latent_heat_released = 0.0
        condensed_water_kg_per_day = 0.0
        convergence_threshold = 0.0001  # Define a small threshold for convergence
        max_iterations = 100  # Set a maximum number of iterations to prevent infinite loops
        previous_condensed_water = None

        for iteration in range(max_iterations):
            # Calculate temperature after drop
            temperature_after_drop = temperature - delta_t

            # Calculate dew point
            dew_point = calculate_dew_point(temperature, rh)

            # Check for condensation
            condensation = temperature_after_drop < dew_point

            # Calculate saturated humidity ratio at cooled temperature
            if condensation:
                x_saturated = calculate_saturation_humidity_ratio(temperature_after_drop)
            else:
                x_saturated = humidity_ratio

            # Calculate delta_X
            delta_X = humidity_ratio - x_saturated if condensation else 0

            # Calculate condensed water mass
            new_condensed_water_kg_per_day = total_mass_dry_air_per_day * delta_X if condensation else 0

            # Calculate latent heat released
            new_latent_heat_released = new_condensed_water_kg_per_day * LATENT_HEAT_VAPORIZATION

            # Recalculate specific heat with updated humidity ratio
            new_humidity_ratio = humidity_ratio - delta_X if condensation else humidity_ratio
            specific_heat = calculate_specific_heat_moist_air(new_humidity_ratio)

            # Adjust temperature drop considering latent heat
            new_delta_t = calculate_adjusted_temperature_drop(
                heat_extracted_kwh,
                new_latent_heat_released,
                AIRFLOW_RATE,
                AIR_DENSITY,
                specific_heat
            )

            # Debug statements
            # print(f"Day {index}, Iteration {iteration}")
            # print(f"Temperature: {temperature}")
            # print(f"Delta T: {delta_t}")
            # print(f"Temperature After Drop: {temperature_after_drop}")
            # print(f"Dew Point: {dew_point}")
            # print(f"Condensation Occurs: {condensation}")
            # print(f"Condensed Water (kg/day): {new_condensed_water_kg_per_day}")
            # print(f"Latent Heat Released (J): {new_latent_heat_released}\n")

            # Check for convergence
            if previous_condensed_water is not None:
                if abs(new_condensed_water_kg_per_day - previous_condensed_water) < convergence_threshold:
                    break  # Converged

            previous_condensed_water = new_condensed_water_kg_per_day

            # Update variables for next iteration
            delta_t = new_delta_t
            latent_heat_released = new_latent_heat_released
            condensed_water_kg_per_day = new_condensed_water_kg_per_day

        # Update the DataFrame with converged values
        df.at[index, 'Adjusted_Temperature_Drop'] = delta_t
        df.at[index, 'Temperature_After_Adjusted_Drop'] = temperature_after_drop
        df.at[index, 'Condensed_Water_kg_per_day'] = condensed_water_kg_per_day
        df.at[index, 'Latent_Heat_Released_Joules'] = latent_heat_released

    # Convert condensed water to liters per day (assuming 1 kg of water = 1 liter)
    df['Condensed_Water_Liters_per_day'] = df['Condensed_Water_kg_per_day']

    # Calculate condensed water per hour
    df['Condensed_Water_Liters_per_hour'] = (
        df['Condensed_Water_Liters_per_day'] / OPERATING_HOURS_PER_DAY
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

    # Plotting functions
    plot_condensation_histogram(df['Condensed_Water_Liters_per_hour'])

    plot_time_series(
        df=df,
        date_column='Date',
        y1_column='Temperature_After_Adjusted_Drop',
        y1_label="Temp Air PAC (°C)",
        y1_color='r',
        y1_axis_label="Température (°C)",
        y2_column='Temperature',
        y2_label="Température (°C)",
        y2_color='g',
        y2_linestyle='--',
        title="Température de l'Air sortant du PAC et Température de l'Air pour la saison de chauffage",
        secondary_y=False
    )

    plot_time_series(
        df=df,
        date_column='Date',
        y1_column='Condensed_Water_Liters_per_hour',
        y1_label="Eau condensée (litres/h)",
        y1_color='b',
        y1_linestyle='-',
        y1_axis_label="Eau condensée (litres/heure)",
        y2_column='RH',
        y2_label="Humidité Relative (%)",
        y2_color='orange',
        y2_linestyle='--',
        y2_axis_label="Humidité Relative (%)",
        title="Production d'eau condensée et Humidité Relative pour la saison de chauffage"
    )
    df['Heating Energy (kW)'] = df ['Heating_Energy_Demand_kWh']/24

    plot_time_series(
        df=df,
        date_column='Date',
        y1_column='Heating Energy (kW)',
        y1_label="Energie Chauffage (kW)",
        y1_color='b',
        y1_linestyle='-',
        y1_axis_label="Demande de Chauffage (kWh)",
        y2_column='COP',
        y2_label="COP",
        y2_color='orange',
        y2_linestyle='--',
        y2_axis_label="COP",
        title="Heating Energie and COP"
    )

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
        y_label='Énergie (kWh)',
        color='r',
        label='Consommation Électrique (kWh)',
        y2_column='Heat_Extracted_From_Air_kWh',
        title="Contributions au Chauffage de l'Électricité et de l'Air en fonction de la Température Extérieure",
        y2_color='green',
        y2_label_in_legend='Chaleur Extraite de l\'Air (kWh)',
        y3_column='Heating_Energy_Demand_kWh',
        y3_label_in_legend='Demande de Chauffage (kWh)',
    )

if __name__ == "__main__":
    main()


