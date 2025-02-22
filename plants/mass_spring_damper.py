import numpy as np
from scipy.integrate import solve_ivp
from plants.base_plant import BasePlant

class MassSpringDamper(BasePlant):
    def __init__(self, mass=1.0, spring_constant=1.0, damping_coefficient=0.5, initial_position=0.0, initial_velocity=0.0):
        self.mass = mass
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.state = np.array([initial_position, initial_velocity])  # State: [position, velocity]

    def _dynamics(self, t, state, input_force):
        """Continuous-time dynamics of the system."""
        x, x_dot = state
        x_ddot = (input_force - self.damping_coefficient * x_dot - self.spring_constant * x) / self.mass
        return [x_dot, x_ddot]

    def update_state(self, input_force, dt):
        """Update the state using numerical integration."""
        sol = solve_ivp(self._dynamics, [0, dt], self.state, args=(input_force,), dense_output=True)
        self.state = sol.y[:, -1]  # Get the state at the final time step
        return self.state

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def update(self, force, dt):
        position, velocity = self.state
        acceleration = (force - self.damping_coefficient * velocity - self.spring_constant * position) / self.mass
        velocity += acceleration * dt
        position += velocity * dt
        self.state = [position, velocity]

    def discretize(self, dt, method='euler'):
        """Discretize the continuous-time dynamics."""
        if method == 'euler':
            A = np.array([[1, dt],
                          [-dt*self.spring_constant/self.mass, 1 - dt*self.damping_coefficient/self.mass]])
            B = np.array([[0],
                          [dt/self.mass]])
        elif method == 'zoh':
            from scipy.signal import cont2discrete
            A_cont = np.array([[0, 1],
                               [-self.spring_constant/self.mass, -self.damping_coefficient/self.mass]])
            B_cont = np.array([[0],
                               [1/self.mass]])
            A, B, _, _, _ = cont2discrete((A_cont, B_cont, np.eye(2), np.zeros((2,1))), dt, method='zoh')
        else:
            raise ValueError("Invalid discretization method. Choose 'euler' or 'zoh'.")
        return A, B

    def __str__(self):
        return f"Mass-Spring-Damper (m={self.mass}, k={self.spring_constant}, c={self.damping_coefficient})"
