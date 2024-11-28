import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


sf_output_directory = 'Sensitivity Function'
if not os.path.exists(sf_output_directory):
    os.makedirs(sf_output_directory)


# Load data from your CSV file 
df = pd.read_csv('estimated_voltages.csv')
temperature = df.iloc[:, 2]
voltage = df.iloc[:, 1]

# Perform quadratic regression
coefficients = np.polyfit(temperature, voltage, 2)  
a, b, c = coefficients        #  V(T) = a*T^2 + b*T + c

# Function return
def sensitivity_function(T):
    return a * T**2 + b * T + c

# Plot the data points
plt.scatter(temperature, voltage, label='Data')

# Plot the fitted curve
T_range = np.linspace(min(temperature), max(temperature), 100)
plt.plot(T_range, sensitivity_function(T_range), color='red', label='Fitted Curve (Quadratic)')
plt.xlabel('Temperature')
plt.ylabel('Estimated Voltage')
plt.title('Voltage vs. Temperature')
plt.legend()

# Show the plot
plt.show()

#Save the result to a txt file
sensitivity_function_text1 = f"Estimated sensitivity function: V(T) = {a:.4f} * T^2 + {b:.4f} * T + {c:.4f}"
sensitivity_function_text2 = f"Coefficients (a, b, c): {coefficients}"
with open(os.path.join(sf_output_directory, 'Sensitivity_function.txt'), 'w') as file:
    file.write(sensitivity_function_text1 + '\n')
    file.write(sensitivity_function_text2)
