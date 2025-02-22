from .base_plant import BasePlant
import math

class InvertedPendulum(BasePlant):
    def __init__(self, length, mass, damping_coefficient):
        self.length = length
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.state = [0, 0]  # [angle, angular_velocity]

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def update(self, torque, dt):
        angle, angular_velocity = self.state
        gravity = 9.81
        angular_acceleration = (torque - self.damping_coefficient * angular_velocity - self.mass * gravity * self.length * math.sin(angle)) / (self.mass * self.length ** 2)
        angular_velocity += angular_acceleration * dt
        angle += angular_velocity * dt
        self.state = [angle, angular_velocity]
