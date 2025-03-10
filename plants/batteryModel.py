import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Simulated true parameters
E0_true = 3.7  # Open-circuit voltage (V)
R_true = 0.1   # Resistance (Ohm)
C_true = 1000  # Capacitance (F)
I = 2.0        # Constant current (A)
dt = 0.1       # Time step (s)
t = np.arange(0, 10, dt)  # Time vector (0 to 10 s)

# Simulate true voltage with initial condition Vc[0] = 0
Vc = np.zeros_like(t)
V_measured = np.zeros_like(t)
Vc[0] = 0
for k in range(1, len(t)):
    Vc[k] = Vc[k-1] + (dt / C_true) * I
    V_measured[k] = E0_true - I * R_true - Vc[k]

# Add some noise
V_measured += np.random.normal(0, 0.01, len(t))  # Noise (0.01 V std)

# Model and optimization
def battery_model(params, I, V_measured, dt, t):
    R, C = params
    Vc = np.zeros_like(t)
    V_pred = np.zeros_like(t)
    Vc[0] = 0  # Initial condition
    for k in range(1, len(t)):
        Vc[k] = Vc[k-1] + (dt / C) * I[k]
        V_pred[k] = E0_true - I[k] * R - Vc[k]
    return V_pred - V_measured  # Residuals

if __name__ == "__main__":
    # Initial guess
    params0 = [0.05, 500]  # Initial R, C
    result = least_squares(battery_model, params0, args=(I * np.ones_like(t), V_measured, dt, t))
    R_est, C_est = result.x

    print(f"Estimated R: {R_est:.3f} Ohm (True: {R_true} Ohm)")
    print(f"Estimated C: {C_est:.0f} F (True: {C_true} F)")

    # Validate and plot
    Vc_est = np.zeros_like(t)
    V_pred = np.zeros_like(t)
    Vc_est[0] = 0
    for k in range(1, len(t)):
        Vc_est[k] = Vc_est[k-1] + (dt / C_est) * I
        V_pred[k] = E0_true - I * R_est - Vc_est[k]

    rmse = np.sqrt(np.mean((V_pred - V_measured) ** 2))
    print(f"RMSE: {rmse:.3f} V")

    plt.figure(figsize=(10, 6))
    plt.plot(t, V_measured, label='Measured Voltage', marker='o')
    plt.plot(t, V_pred, label='Predicted Voltage', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Battery Discharge: Measured vs. Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()