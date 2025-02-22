from .base_plant import BasePlant

class DCMotor(BasePlant):
    def __init__(self, resistance, inductance, back_emf_constant, torque_constant, inertia, damping_coefficient):
        self.resistance = resistance
        self.inductance = inductance
        self.back_emf_constant = back_emf_constant
        self.torque_constant = torque_constant
        self.inertia = inertia
        self.damping_coefficient = damping_coefficient
        self.state = [0, 0]  # [current, angular_velocity]

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def update(self, voltage, dt):
        current, angular_velocity = self.state
        back_emf = self.back_emf_constant * angular_velocity
        current_dot = (voltage - back_emf - self.resistance * current) / self.inductance
        torque = self.torque_constant * current
        angular_acceleration = (torque - self.damping_coefficient * angular_velocity) / self.inertia
        current += current_dot * dt
        angular_velocity += angular_acceleration * dt
        self.state = [current, angular_velocity]
