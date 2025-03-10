import numpy as np
import matplotlib.pyplot as plt
from controllers.pid import PIDController

def simulate_inverted_pendulum_pid():
    # Define system parameters
    pendulum_length = 1.0
    gravity = 9.81

    # Initialize PID controller
    pid_controller = PIDController()

    # Initialize state
    state = 0.0

    # Initialize time and angle arrays
    time = np.linspace(0, 10, 100)
    angles = []

    # Simulation loop
    for t in time:
        # Compute control input
        control_input = pid_controller.compute_control(state)
        
        # Update state (this is a placeholder, replace with actual state update logic)
        state += control_input * 0.1
        
        # Store the angle
        angles.append(state)

    # Plot results
    plt.plot(time, angles)
    plt.show()

if __name__ == "__main__":
    simulate_inverted_pendulum_pid()
