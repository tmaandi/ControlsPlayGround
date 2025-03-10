from base_controller import BaseController

class PIDController(BaseController):

    def __init__(self, Kp = 0, Ki = 0, Kd = 0, DT = 1, INT_LIMITS = [-100, 100]):
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        if DT > 0:
            self.DT = DT
        else:
            self.DT = 1

        self.error = 0
        self.integral = 0
        self.integralLimits = INT_LIMITS


    def update(self, setpoint, processVariable):

        error = setpoint - processVariable

        derivative = (error - self.error)/self.DT

        self.error = error

        self.integral += (self.error * self.DT)

        # Clamp Integral Term
        if (self.integral < self.integralLimits[0]):
            self.integral = self.integralLimits[0]
        elif (self.integral > self.integralLimits[1]):
            self.integral = self.integralLimits[1]
        else:
            pass

        controlVariable = (self.Kp * self.error) + (self.Ki * self.integral) + (self.Kd * derivative)

        return controlVariable
    
    def reset(self):
        
        self.error = 0
        self.integral = 0

       
