import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
from plants.base_plant import BasePlant

class MassSpringDamper(BasePlant):
    def __init__(self, mass=1.0, spring_constant=1.0, damping_coefficient=0.5, initial_position=0.0, initial_velocity=0.0, solver='continuous'):
        self.mass = mass
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.state = np.array([initial_position, initial_velocity])  # State: [position, velocity]
        self.solver = solver  # Solver type: 'continuous', 'euler', 'zoh', 'tustin', etc.

    def _continuous_dynamics_matrices(self):
        """Return the continuous-time system matrices A_cont and B_cont."""
        A_cont = np.array([[0, 1],
                           [-self.spring_constant/self.mass, -self.damping_coefficient/self.mass]])
        B_cont = np.array([[0],
                           [1/self.mass]])
        return A_cont, B_cont

    def _dynamics(self, t, state, input_force):
        """Continuous-time dynamics of the system (internal use)."""
        A_cont, B_cont = self._continuous_dynamics_matrices()
        x_dot = A_cont @ state + B_cont @ np.array([[0], [input_force]])
        return x_dot

    def _discretize(self, dt, method='euler'):
        """Discretize the continuous-time dynamics (internal use)."""
        A_cont, B_cont = self._continuous_dynamics_matrices()
        A, B, _, _, _ = cont2discrete((A_cont, B_cont, np.eye(2), np.zeros((2,1))), dt, method=method)
        return A, B
    
    def update_state(self, input_force, dt):
        """Update the state using the specified solver."""
        if self.solver == 'continuous':
            sol = solve_ivp(self._dynamics, [0, dt], self.state, args=(input_force,), dense_output=True)
            self.state = sol.y[:, -1]  # Get the state at the final time step
        else:
            A, B = self._discretize(dt, self.solver)
            self.state = A @ self.state + B @ np.array([[0], [input_force]])
        return self.state

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state