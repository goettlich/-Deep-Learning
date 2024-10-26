from abc import ABC, abstractmethod
import numpy as np
import torch

class Integrator(ABC):
    
    @abstractmethod
    def solve(self, f, x0, t):
        "The Solver method"
    
class RK4(Integrator):
    
    def __init__(self, dt_solver=0.01):

        super().__init__()
        self.dt = dt_solver

    def rk4_step(self, f, x, t, dt):

        k1 = f(x, t)
        k2 = f(x + dt*k1/2, t + dt/2)
        k3 = f(x + dt*k2/2, t + dt/2)
        k4 = f(x + dt*k3, t + dt)
        
        return x + dt/6* (k1 + 2*k2 + 2*k3 + k4)

    def solve(self, f, x0, t, last_only=False):
        
        t = torch.as_tensor(t)
        dt_positive = (t[1]-t[0] > 0).item()
        if dt_positive==False:
            pass
        idx=1
        t_now = t[0].clone()
        x = torch.zeros(size=(len(t),*x0.shape))
        x[0], x_now = x0, x0
        
        while idx < len(t):
            
            dt_step = min(self.dt, t[idx]-t_now) if dt_positive else -min(self.dt, t_now-t[idx])
            x_now = self.rk4_step(f, x_now, t_now, dt_step)
            t_now += dt_step

            if abs(t_now - t[idx]) < 1e-8:
                x[idx] = x_now
                idx += 1
                
        return x[-1] if last_only else x
    

class IntegratorFactory:

    @staticmethod
    def get_integrator(name: str, **kwargs) -> Integrator:
        if name == 'RK4':
            return RK4(**kwargs)
        
        else:
            raise ValueError(f"Unknown Integrator: {name}")



    