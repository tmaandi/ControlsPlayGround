import numpy as np
import cvxpy as cp
from .base_controller.py import BaseController

class MPCController(BaseController):
    def __init__(self, 
                 dt=0.1,                # sampling time
                 horizon=10,            # prediction horizon
                 Q=None,               # state cost matrix
                 R=None,               # input cost matrix
                 mass=1.0,             # mass of the system
                 spring_k=1.0,         # spring constant
                 damping_b=0.2):       # damping coefficient
        
        # System dimensions
        self.nx = 2  # number of states [position, velocity]
        self.nu = 1  # number of inputs [force]
        self.N = horizon
        
        # System parameters
        self.mass = mass
        self.spring_k = spring_k
        self.damping_b = damping_b
        self.dt = dt
        
        # Cost matrices
        self.Q = Q if Q is not None else np.eye(self.nx)
        self.R = R if R is not None else np.eye(self.nu)
        
        # State and input constraints
        self.x_min = np.array([-5.0, -5.0])  # min position and velocity
        self.x_max = np.array([5.0, 5.0])    # max position and velocity
        self.u_min = np.array([-10.0])       # min force
        self.u_max = np.array([10.0])        # max force
        
        # Create discrete state-space model
        self._discretize_system()
        self._setup_optimization_problem()
        
    def _discretize_system(self):
        """Convert continuous system to discrete state-space"""
        # Continuous state-space matrices
        Ac = np.array([[0, 1],
                      [-self.spring_k/self.mass, -self.damping_b/self.mass]])
        Bc = np.array([[0],
                      [1/self.mass]])
        
        # Discretize using forward Euler (for simplicity)
        self.A = np.eye(2) + self.dt * Ac
        self.B = self.dt * Bc
        
    def _setup_optimization_problem(self):
        """Setup the MPC optimization problem"""
        # Variables
        self.x = cp.Variable((self.nx, self.N + 1))
        self.u = cp.Variable((self.nu, self.N))
        
        # Parameters
        self.x0 = cp.Parameter(self.nx)
        self.xr = cp.Parameter(self.nx)  # reference state
        
        # Initialize objective and constraints
        objective = 0
        constraints = [self.x[:, 0] == self.x0]
        
        # Add stage costs and constraints
        for k in range(self.N):
            # Objective function
            objective += cp.quad_form(self.x[:, k] - self.xr, self.Q) + \
                        cp.quad_form(self.u[:, k], self.R)
            
            # System dynamics
            constraints += [self.x[:, k+1] == self.A @ self.x[:, k] + self.B @ self.u[:, k]]
            
            # State and input constraints
            constraints += [self.x_min <= self.x[:, k], self.x[:, k] <= self.x_max]
            constraints += [self.u_min <= self.u[:, k], self.u[:, k] <= self.u_max]
        
        # Terminal constraint
        constraints += [self.x_min <= self.x[:, -1], self.x[:, -1] <= self.x_max]
        
        # Create and store the optimization problem
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        
    def compute_control(self, state, reference=None):
        """Compute the control input for the current state"""
        if reference is None:
            reference = np.zeros(self.nx)
            
        # Update parameters
        self.x0.value = state
        self.xr.value = reference
        
        try:
            # Solve the optimization problem
            self.problem.solve(solver=cp.OSQP)
            
            if self.problem.status == cp.OPTIMAL:
                # Return the first control input
                return self.u.value[:, 0]
            else:
                print(f"Warning: Problem status {self.problem.status}")
                return np.zeros(self.nu)
                
        except cp.error.SolverError:
            print("Error: Solver failed")
            return np.zeros(self.nu)
            
    def get_prediction(self):
        """Return the predicted trajectory"""
        if self.x.value is None:
            return None
        return self.x.value
