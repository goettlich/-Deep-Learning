import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from integrators import RK4
import numpy as np
import math
from models import RNN, MLP

# class ODEF(nn.Module):
    
#     def forward_with_grad(self, zi, ti, dLdz):
        
#         loss = self.forward(zi, ti)
        
#         adfdz, adfdt, *adfdp = torch.autograd.grad(
#             outputs=(loss,), 
#             inputs = (zi,ti)+tuple(self.parameters()), 
#             grad_outputs=(dLdz),
#             allow_unused=True,
#             retain_graph=True
#         )

#         # expand back into required shapes
#         pass


# TODO s current: 
# 1. Are func.parameters() and t needed for augmented_dynamics?
# 2. backward(ctx, dLdz), but returns (adj_z.view(bs, *z_shape), adj_t, adj_p, None) --> how can it be repeatedly called? IT CANNOT --> Backward is only called ONCE in the backprop of ONE sample/Batch
# 3. But then, integrator.solve(func,z0,t) is called on augmented_dynamics --> how? And how is it called repeatedly + what does it return?
# 4. How does the gradient update work


class AdjointODE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z0: torch.Tensor, t: torch.Tensor, func: nn.Module, integrator: RK4):

        with torch.no_grad():
            z = integrator.solve(func, z0, t)
        
        # params = torch.cat([p.flatten() for p in func.parameters()])
        # ctx.save_for_backward(t, z.clone(), params)
        
        # ctx values don't need requires grad as only the tensors are used as grad_output
        ctx.func = func
        ctx.integrator = integrator
        ctx.save_for_backward(t, z.clone())
        return z

    @staticmethod
    def backward(ctx, dLdz):

        with torch.no_grad():

            # t, z, params = ctx.saved_tensors()
            t, z = ctx.saved_tensors
            func = ctx.func
            integrator = ctx.integrator

            T,B,*z_shape = z.size()
            n_dim = np.prod(z_shape)
            aug_state = [z[-1], dLdz[-1], torch.zeros_like(param) for param in func.parameters()]

            def aug_dynamics(aug_state_i, ti):
                # aug_state = [z(t), az(t)=dL/dz, at(t)=dL/dt, ap(t)=dL/dp]
                # dim(a(t)) = dim(z(t))

                zi = aug_state_i[0]
                adj_zi = aug_state_i[1]
                
                with torch.enable_grad():
                        
                    loss = func(zi, ti)

                    # chain rule: dL/d[z(i-1),t(i-1),p(i-1)] = -previous_sensitivity * dL/d[zi,ti,pi]
                    # general : df(g(x))/dx = df/dg * dg/dx --> df/dg =: grad_outputs (i.e. previous_sensitivity)
                    adfdz, adfdt, *adfdp = torch.autograd.grad(
                        outputs=(loss,), 
                        inputs = (zi,ti)+tuple(func.parameters()), 
                        grad_outputs=(-adj_zi),
                        allow_unused=True,
                        retain_graph=True
                    )

                adfdz = torch.zeros_like(zi) if adfdz is None else adfdz
                adfdt = torch.zeros_like(t) if adfdt is None else adfdt
                adfdp = [torch.zeros_like(param) if adfdp is None else adfdp_element 
                        for param, adfdp_element in zip(func.parameters(), adfdp)]

                return (loss, adfdz, adfdt, *adfdp) # [z(t), dL/dz(t), dL/dt, dL/dp]
            
            for i in range(len(t)-1, 0, -1):
                
                aug_state = integrator.solve()


class NeuralODE(nn.Module):

    def __init__(self, func: nn.Module, integrator: RK4):

        super().__init__()

        self.func = func
        self.integrator = integrator

    def forward(self, z0, t):
        # requires_grad can only be changed outside of torch.autograd.Function, so pass with grad
        z=AdjointODE.apply(z0.requires_grad_(True), t, self.func, self.integrator)
        return z

# Questions:
# - can i also build a neural ode without adjoint (why is the odeint() import dependent on the flag "adjoint" --> and if so, why adjoint at all?)
from dataset import TrajectoryDataset
from systems import pendulum_time_invariant

n_dim = 2
num_timesteps = 4
dt_out = 0.05
dt_solver = 0.01 
ode = pendulum_time_invariant

# Define Models
mlp = MLP(n_input_states=n_dim, n_output_states=n_dim)
state_rnn = RNN(input_size=n_dim, hidden_size=32, num_layers=2, output_size=n_dim)
rk4 = RK4(dt=dt_solver) # Not passed to TrajectoryGenerator anymore, maybe change
node = NeuralODE(func=mlp, integrator=rk4)

# Prepare Data
dataset = TrajectoryDataset(ode=ode, samples_per_epoch=10, num_timesteps=4)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3)

sample = next(iter(dataloader)) # shape=(3,4,2)=(B,T,dim)
B,T,dim = sample.shape
sample = sample.reshape(T,B,dim)
first_frame = sample[0,:,:] # Dynamic MLP simply predicts the dynamics matrix based on a single sample

# Model predictions
pred_state_mlp = mlp(first_frame,torch.tensor(0)) # Frame predictor
# pred_state_rnn_from_frame,_ = state_rnn(frame,infer_n_steps=3) # Sequence predictor
# pred_state_rnn_from_sample,_ = state_rnn(sample[0])
# TODO: Streamline Solver and metadata definitions and make them consistent across NODE, TrajGenerator, ...
t=torch.arange(0., num_timesteps * dt_out, dt_out, dtype=torch.float32)


# pred_state_node = node(first_frame.requires_grad_(True),t) # Sequence or frame predictor, based on passed t vector
pred_state_node = node(first_frame,t)
loss = torch.sum((sample - pred_state_node)**2)
loss.backward()

pass