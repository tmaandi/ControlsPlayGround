import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpc import MPC  # Assuming an MPC class is defined in mpc.py

def mass_spring_dynamics(t, y, u):
    m = 1.0  # mass
    k = 1.0  # spring constant
    b = 0.2  # damping coefficient
    x, v = y
    dxdt = v
    dvdt = (u - b * v - k * x) / m
    return [dxdt, dvdt]

def mpc_control(x, v, setpoint):
    # Placeholder for a simple proportional controller
    kp = 10.0
    return kp * (setpoint - x)

def simulate_mass_spring_mpc():
    t_span = (0, 10)
    y0 = [0, 0]  # initial conditions: [position, velocity]
    setpoint = 1.0  # desired position

    def closed_loop_dynamics(t, y):
        x, v = y
        u = mpc_control(x, v, setpoint)
        return mass_spring_dynamics(t, y, u)

    sol = solve_ivp(closed_loop_dynamics, t_span, y0, t_eval=np.linspace(0, 10, 100))

    # Define system parameters
    mass = 1.0
    spring_constant = 1.0

    # Initialize MPC controller
    mpc_controller = MPC()

    # Simulation loop
    for t in range(100):
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
