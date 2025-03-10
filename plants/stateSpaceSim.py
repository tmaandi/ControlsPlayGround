import numpy as np
import matplotlib.pyplot as plt

class StateSpaceSim:
    def __init__(self, A, B, C, D, x_0, u, dt):
        """
        Initialize a state-space model for simulation.
        
        Args:
            A (np.ndarray): State transition matrix (n x n)
            B (np.ndarray): Input matrix (n x m)
            C (np.ndarray): Output matrix (p x n)
            D (np.ndarray): Feedforward matrix (p x m)
            x_0 (np.ndarray): Initial state vector (n x 1)
            u (np.ndarray): Input sequence (time_steps x m)
            dt (float): Time step for discrete simulation
        """
        # Validate input dimensions
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")
        if A.shape[0] != x_0.shape[0]:
            raise ValueError(f"A rows ({A.shape[0]}) must match x_0 rows ({x_0.shape[0]})")
        if B.shape[0] != A.shape[0]:
            raise ValueError(f"B rows ({B.shape[0]}) must match A rows ({A.shape[0]})")
        if B.shape[1] != u.shape[1]:
            raise ValueError(f"B columns ({B.shape[1]}) must match u columns ({u.shape[1]})")
        if C.shape[1] != A.shape[0]:
            raise ValueError(f"C columns ({C.shape[1]}) must match A rows ({A.shape[0]})")
        if D.shape[0] != C.shape[0]:
            raise ValueError(f"D rows ({D.shape[0]}) must match C rows ({C.shape[0]})")
        if D.shape[1] != u.shape[1]:
            raise ValueError(f"D columns ({D.shape[1]}) must match u columns ({u.shape[1]})")

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n = A.shape[0]  # Number of states
        self.p = C.shape[0]  # Number of outputs
        self.time_steps = u.shape[0]
        self.x = np.zeros((self.time_steps, self.n))  # Pre-allocate state array
        self.x[0] = x_0.flatten()  # Set initial state
        self.u = u  # Input sequence
        self.dt = dt  # Time step
        self.y = np.zeros((self.time_steps, self.p))  # Pre-allocate output array
        self.y[0] = (self.C @ x_0 + self.D @ u[0]).flatten()  # Initial output

    def simulate(self):
        """
        Simulate the state-space model using Euler integration.
        Updates state and output over time based on input sequence.
        """
        for k in range(1, len(self.u)):
            # Euler method: x[k+1] = x[k] + dt * (A * x[k] + B * u[k])
            x_k = self.x[k-1].reshape(-1, 1)  # Ensure column vector
            term1 = self.A @ x_k
            term2 = self.B @ self.u[k].reshape(-1, 1)  # Ensure u[k] is a column vector
            if term1.shape != (self.n, 1):
                raise ValueError(f"Expected term1 shape ({self.n}, 1), got {term1.shape}")
            if term2.shape != (self.n, 1):
                raise ValueError(f"Expected term2 shape ({self.n}, 1), got {term2.shape}")
            x_next = x_k + self.dt * (term1 + term2)
            if x_next.shape != (self.n, 1):
                raise ValueError(f"Expected x_next shape ({self.n}, 1), got {x_next.shape}")
            self.x[k] = x_next.flatten()  # Store as flat array
            # Output: y[k] = C * x[k] + D * u[k]
            y_next = self.C @ x_next + self.D @ self.u[k].reshape(-1, 1)
            self.y[k] = y_next.flatten()  # Store as flat array
        
        return self.x, self.y

# Example usage
if __name__ == "__main__":
    # Example 2nd-order system (e.g., a damped oscillator with input)
    A = np.array([[0, 1], [-1, -0.1]])  # State matrix (2 x 2)
    B = np.array([[0], [1]])            # Input matrix (2 x 1)
    C = np.array([[1, 0]])              # Output matrix (1 x 2)
    D = np.array([[0]])                 # Feedforward matrix (1 x 1)
    x_0 = np.array([[1.0], [0.0]])     # Initial state (2 x 1)
    t = np.arange(0, 10, 0.1)           # Time vector
    u = np.sin(t).reshape(-1, 1)        # Sinusoidal input (100 x 1)
    dt = 0.1                            # Time step

    sim = StateSpaceSim(A, B, C, D, x_0, u, dt)
    states, outputs = sim.simulate()

    # Plot results
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, states[:, 0], label='State x1')
    plt.plot(t, states[:, 1], label='State x2')
    plt.title("State Trajectories")
    plt.xlabel("Time (s)")
    plt.ylabel("States")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, outputs, label='Output')
    plt.title("Output Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()
    plt.show()