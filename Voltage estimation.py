import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

#List of temperature from txt file
temps = [26.8, 30.4, 34, 36, 37.8, 40.6, 45.7, 54.7, 65.5, 67.8, 69.0, 72.2, 82.2, 87.1, 100.2, 106.8, 111.0, 115.1, 116.8]
file_number_list = [1,2,3,4,5,6,7,8,10,11,12,14,16,17,19,20,21,22,23]
switch_indices = []
in_stable_indices = []
out_stable_indices = []




# Create the output plots directory for voltage and current
voltage_output_directory = 'voltage output plots'
if not os.path.exists(voltage_output_directory):
    os.makedirs(voltage_output_directory)
current_output_directory = 'current output plots'
if not os.path.exists(current_output_directory):
    os.makedirs(current_output_directory)


#Function compute slope
def compute_slopes(current, time_intervals):
    delta_current = np.diff(current)
    delta_time = np.diff(time_intervals)
    slopes = delta_current / delta_time
    return slopes

#Function find switch point, stable zone of current
def allthingscurrent(current,time_intervals):
    switch_index = None
    stable_begin_index = None
    stable_end_index = None

    found_switch = False
    found_stable_begin = False
    found_stable_end = False
    slopes = compute_slopes(current, time_intervals)
    from collections import Counter
    # Detect switch, stable begin, and stable end
    for i in range(len(slopes)):
        if slopes[i] > 8000 and not found_switch:
            switch_index = i 
            found_switch = True
            break

    if found_switch:
        for i in range(switch_index, len(slopes)):
            if abs(slopes[i]) < 100: streak += 1
            else:streak = 0
        
            if streak == 50 and not found_stable_begin and 0.73 > current[i + 1] > 0.729:
                stable_begin_index = i -50  
                found_stable_begin = True
                break

    if found_stable_begin:
        for i in range(stable_begin_index, len(slopes)):
            if slopes[i] < -1000 and not found_stable_end:
                stable_end_index = i  
                found_stable_end = True
                break
    return switch_index, stable_begin_index, stable_end_index

# save data function
def savedata(filenumber,t0_voltage):
    try:
        estimated_voltages = np.loadtxt("estimated_voltages.csv", delimiter=",")
    except FileNotFoundError:
        estimated_voltages = np.empty((0, 6))
    if filenumber == 1:
        estimated_voltages = np.empty((0, 6))
    for i in range(0,len(file_number_list)):
        if file_number_list[i]== filenumber:
            temperature =temps[i]
            switch_index = switch_indices[i]
            in_stable_index = in_stable_indices[i]
            out_stable_index = out_stable_indices[i]
    new_data = np.array([[filenumber, t0_voltage, temperature,switch_index,in_stable_index, out_stable_index]])
    estimated_voltages = np.vstack((estimated_voltages, new_data))
    np.savetxt("estimated_voltages.csv", estimated_voltages, delimiter=",",fmt="%f")
    


#Create readme file
def create_readme():
    readme_file = 'estimated_value.readme.txt'
    
    if not os.path.exists(readme_file):
        content = 'Label for estimated_value.csv: "File number, Estimated voltage, Temperature, Switch index, Begin of stable phase index, End of stable phase index"'
        
        with open(readme_file, 'w') as f:
            f.write(content)


# plot voltage function
def plotvoltage(filenumber):
    # Load CSV data
    csv_path = f'data_set/turn_on{filenumber}.csv'
    df_csv = pd.read_csv(csv_path, skiprows=8)
    voltage = df_csv.iloc[:, 1]
    current = df_csv.iloc[:, 14]
    time_intervals = df_csv.iloc[:, 23]
    
    # Get indices for current analysis
    switch_index, in_stable_index, out_stable_index = allthingscurrent(current, time_intervals)

    # Add the index to indices list
    switch_indices.append(switch_index)
    in_stable_indices.append(in_stable_index)
    out_stable_indices.append(out_stable_index)
    # Adjust time intervals relative to the switch index
    adjusted_time_intervals = time_intervals - time_intervals[switch_index]

    # Calculate the square root of time, handle negative values by taking absolute value before sqrt
    sqrt_time = np.sqrt(np.abs(adjusted_time_intervals)) * np.sign(adjusted_time_intervals)
    
    # Record values for the stable phase
    voltage_recorded = voltage[in_stable_index:out_stable_index]
    sqrt_time_recorded = sqrt_time[in_stable_index:out_stable_index]
    
    # Perform linear regression
    res = linregress(sqrt_time_recorded, voltage_recorded)
    extrapolated_voltage_at_t0 = res.slope * np.sqrt(0) + res.intercept
    # Calculate extrapolated voltage at t = 0
    t_0 = np.array([[0]])  
    extrapolated_voltage = res.intercept + res.slope * np.sqrt(t_0)

    #Extend fitted line
    t_extended = np.linspace(0, 0.01, 500)  
    sqrt_time_extended = np.sqrt(t_extended)
    extrapolated_voltage_extended = res.intercept + res.slope * sqrt_time_extended

    # Plot the entire voltage series against the square root of time
    plt.plot(sqrt_time, voltage, label='Entire Voltage Series')

    # Highlight specific points
    plt.scatter(sqrt_time[switch_index], voltage[switch_index], color='green', marker='o', label='Power Switch')
    plt.plot(sqrt_time_recorded, voltage_recorded, 'o', label='Stable Phase Data')
    plt.scatter(0, extrapolated_voltage, color='black', marker='x', label='Extrapolated t=0')

    # Plot the extended fitted line
    plt.plot(sqrt_time_extended, extrapolated_voltage_extended, 'r', label='Extended Fitted Line')

    # Labels and title
    plt.xlabel('Square Root of Adjusted Time')
    plt.ylabel('Voltage')
    plt.title('Voltage vs. Square Root of Adjusted Time')
    plt.legend()
    plt.grid(True)
    


# Save the plot in the 'output plots' directory
    plot_filename = os.path.join(voltage_output_directory, f'voltage_plot_{filenumber}.png')
    plt.savefig(plot_filename)
    plt.show()

    return filenumber, extrapolated_voltage_at_t0

    


def plotcurrent(filenumber):
    
    csv_path = f'data_set/turn_on{filenumber}.csv'
    df_csv = pd.read_csv(csv_path, skiprows=8)  
    current = df_csv.iloc[:, 14]
    time_intervals = df_csv.iloc[:, 23]
    switch_index, in_stable_index, out_stable_index = allthingscurrent(current,time_intervals)
    if switch_index is not None:
        current_recorded = current
        adjusted_time_intervals = time_intervals - time_intervals[switch_index]
    sqrt_time = np.sqrt(np.abs(adjusted_time_intervals)) * np.sign(adjusted_time_intervals)
    plt.plot(sqrt_time, current_recorded)

    if switch_index is not None:
        current_recorded = current
        adjusted_time_intervals = time_intervals - time_intervals[switch_index]
        sqrt_time = np.sqrt(np.abs(adjusted_time_intervals)) * np.sign(adjusted_time_intervals)
        plt.plot(sqrt_time, current_recorded)

        # Highlight specific points using adjusted times and square root of time
        plt.scatter(sqrt_time[switch_index], current[switch_index], color='green', marker='o', label='Power Switch')

        plt.scatter(sqrt_time[in_stable_index], current[in_stable_index], color='blue', marker='o', label='Start Stable')

        plt.scatter(sqrt_time[out_stable_index], current[out_stable_index], color='red', marker='o', label='End Stable')
        
    plt.xlabel('Square Root of Time')
    plt.ylabel('Current')
    plt.title('Current vs. Square Root of Time')
    plt.grid(True)
    plt.legend()
    
    plot_filename = os.path.join(current_output_directory, f'current_plot_{filenumber}.png')
    plt.savefig(plot_filename)

    plt.show()
    plt.show
    return filenumber




def main():
    
    for i in range (1,24):
        try:
            filenumber, t0_voltage = plotvoltage(i)
            savedata(filenumber,t0_voltage)
            plotcurrent(i)
        except FileNotFoundError:
            pass
    create_readme()

main()