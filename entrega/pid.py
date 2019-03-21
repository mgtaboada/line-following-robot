class PID:
    def __init__(self,Kp,Ki,Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error = 0
        self.integral = 0
    def compute(self,error):
        """Devuelve el nuevo valor de output y actualiza las constantes"""

        p = error
        i = self.integral + error
        d = error - self.error

        self.integral = i
        self.error = error

        return self.Kp*p + self.Ki*i + self.Kd*d
