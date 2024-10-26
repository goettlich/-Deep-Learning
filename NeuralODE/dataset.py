import torch
import numpy as np
from systems import pendulum_time_invariant, lorenz
from integrators import RK4

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Provides a standard configuration dataset for different physics problems + solvers 
    """
    # TODO: 
    # - pass parameters to integrator (dt, ...)
    # - pass parameters to ode (physics parameters, ...)
    def __init__(self, ode, num_timesteps_out=100, dt_out=0.05, dt_solver=0.01, samples_per_epoch=100):
        
        self.ode = ode
        self.integrator = RK4(dt_solver=dt_solver)

        self.num_timesteps_out = num_timesteps_out
        self.samples_per_epoch = samples_per_epoch
        self.dt_out = dt_out
        self.dt_solver = dt_solver

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        
        torch.manual_seed(idx)
        
        if self.ode is pendulum_time_invariant:
            start = torch.rand(2)-0.5
            # dt = 0.07
        
        elif self.ode is lorenz:
            perturbation = 1e-2
            start = torch.tensor([0,1.,1.]) + (torch.rand(3)-0.5)*perturbation
            # dt = 0.01
        else:
            raise(NotImplementedError)
        
        t = np.arange(start=0, stop=self.num_timesteps_out*self.dt_out, step=self.dt_out)
        # Solver solves with dt_solver, but outputs dt_out steps 
        trajectory = self.integrator.solve(f=self.ode, x0=start, t=t)
        return (torch.as_tensor(trajectory).float(), torch.as_tensor(t).float())