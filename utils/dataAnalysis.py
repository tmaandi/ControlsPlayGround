import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def movingAverage(data, windowSize):
    """Compute a moving average with a sliding window.
    
    Args:
        data (list or np.ndarray): Input data sequence.
        windowSize (int): Size of the moving average window.
    
    Returns:
        list: Smoothed data sequence.
    """
    buffer = [0] * windowSize
    filtData = []
    runningSum = 0
    for i in range(len(data)):
        buffer_ind = i % windowSize
        runningSum -= buffer[buffer_ind]  # Remove oldest value
        buffer[buffer_ind] = data[i]      # Add new value
        runningSum += data[i]
        window = min(i + 1, windowSize)   # Adjust for initial steps
        filtData.append(runningSum / window if window > 0 else 0)
    return filtData

if __name__ == '__main__':
    try:
        # Get the absolute path to the data file
        data_path = os.path.join('..', 'Data', 'sample.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found at {data_path}. Please ensure the Data directory exists and contains sample.csv")
        
        # Read the CSV file
        testData = pd.read_csv(data_path)
        required_columns = ['time', 'voltage', 'current', 'temperature']
        if not all(col in testData.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        print("Raw data:")
        print(testData)
        
        # Apply moving average to current
        filtered_data = movingAverage(testData['current'].values, 3)
        print("\nFiltered data (smoothed current):")
        print(filtered_data)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(testData['time'], testData['voltage'], label='Voltage (V)', color='blue')
        plt.title('Voltage vs. Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(testData['time'], filtered_data, label='Smoothed Current (A)', color='red')
        plt.title('Smoothed Current vs. Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Calculate total energy discharged
        dt = testData['time'].iloc[1] - testData['time'].iloc[0]  # Assume constant time step
        power = testData['voltage'] * testData['current']
        totalEnergy = np.trapezoid(power, testData['time'])
        print(f"Total energy discharged: {totalEnergy:.2f} Joules")

    except FileNotFoundError as e:
        print(f"File error: {str(e)}")
    except ValueError as e:
        print(f"Data error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")