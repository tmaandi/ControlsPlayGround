import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from controllers.mpc import MPCController
from plants.mass_spring_damper import MassSpringDamper

# Plant Parameters
mass = 1.0
spring_constant = 1.0
damping_coefficient = 0.5
initial_position = 0.0
initial_velocity = 0.0

# Simulation Parameters
solver='continuous'
t_span = (0, 10)

# Controller Parameters
setpoint = 1.0  # desired position



def simulate_mass_spring_mpc():


    def closed_loop_dynamics(t, y):
        x, v = y
        u = mpc_control(x, v, setpoint)
        return mass_spring_dynamics(t, y, u)

    sol = solve_ivp(closed_loop_dynamics, t_span, y0, t_eval=np.linspace(0, 10, 100))

    # Initialize MPC controller
    mpc_controller = MPCController()

    # Simulation loop
    for t in range(100):
        state = [sol.y[0][t], sol.y[1][t]]  # Define the state as [position, velocity]
        control_input = mpc_controller.compute_control(state)

    plt.plot(sol.t, sol.y[0], label='Position')
    plt.plot(sol.t, sol.y[1], label='Velocity')
    plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.legend()
    plt.title('Mass-Spring System with MPC')
    plt.show()

if __name__ == "__main__":
    simulate_mass_spring_mpc()
