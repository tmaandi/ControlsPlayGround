import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Model function with discharge
def battery_model(params, I, V_measured, dt, t, Rp):
    R, C = params
    Vc = np.zeros_like(t)
    V_pred = np.zeros_like(t)
    Vc[0] = 0
    for k in range(1, len(t)):
        Vc[k] = Vc[k-1] + dt * (I[k] / C - Vc[k-1] / (Rp * C))
        V_pred[k] = E0_true - I[k] * R - Vc[k]
    return V_pred - V_measured

if __name__ == '__main__':
    # Simulated true parameters
    E0_true = 3.7  # Open-circuit voltage (V)
    R_true = 0.1   # Resistance (Ohm)
    C_true = 1000  # Capacitance (F)
    I = 2.0        # Constant current (A)
    dt = 0.1       # Time step (s)
    t = np.arange(0, 10, dt)  # Time vector (0 to 10 s)

    # Simulate true voltage with initial condition Vc[0] = 0
    Rp = 10.0  # Parallel resistance (Ohm)
    # Simulate true voltage
    Vc = np.zeros_like(t)
    V_measured = np.zeros_like(t)
    Vc[0] = 0
    for k in range(1, len(t)):
        Vc[k] = Vc[k-1] + dt * (I / C_true - Vc[k-1] / (Rp * C_true))
        V_measured[k] = E0_true - I * R_true - Vc[k]

    # Add noise
    V_measured += np.random.normal(0, 0.01, len(t))

    # Optimization
    params0 = [0.05, 500]
    result = least_squares(battery_model, params0, args=(I * np.ones_like(t), V_measured, dt, t, Rp))
    R_est, C_est = result.x

    print(f"Estimated R: {R_est:.3f} Ohm (True: {R_true} Ohm)")
    print(f"Estimated C: {C_est:.0f} F (True: {C_true} F)")