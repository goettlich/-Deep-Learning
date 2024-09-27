from abc import ABC, abstractmethod
import numpy as np
import torch

class Integrator(ABC):
    
    @abstractmethod
    def solve(self, f, x0, t):
        "The Solver method"
    
class RK4(Integrator):
    
    def __init__(self, dt=0.01):

        super().__init__()
        self.dt = dt

    def rk4_step(self, f, x, t, dt):

        k1 = f(x, t)
        k2 = f(x + dt*k1/2, t + dt/2)
        k3 = f(x + dt*k2/2, t + dt/2)
        k4 = f(x + dt*k3, t + dt)
        
        return x + dt/6* (k1 + 2*k2 + 2*k3 + k4)

    def solve(self, f, x0, t):

        idx=1
        t_now = t[0]
        # x = np.zeros(shape=(len(t),len(x0)))
        x = torch.zeros(size=(len(t),*x0.shape))
        x[0], x_now = x0, x0
        
        while idx < len(t):
            
            dt_step = min(self.dt, t[idx]-t_now)
            x_now = self.rk4_step(f, x_now, t_now, dt_step)
            t_now += dt_step

            if abs(t_now - t[idx]) < 1e-8:
                x[idx] = x_now
                idx += 1
                
        return x



    