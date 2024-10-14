import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from integrators import RK4
import numpy as np
from models import MLP
from dataset import TrajectoryDataset
from systems import pendulum_time_invariant
import matplotlib.pyplot as plt
import random
import os
import json

# NOTE:
# - What made the difference: Batched samples (only use random-indexed part of the time sequence) --> This way the model gets to see what is happening farer inside of the spiral --> otherwise we have some vanishing gradient there --> Find out why that exactly is the case theoretically (shouldnt vanishing gradient apply to first steps due to backwards adjoint?)

# TODO
# - Try again with tanh --> check if it works now
# - correct time problem of plots and only plot same "test"sample
# - train RNN the same way!
# - Why does a Layernorm in the MLP destroy a neural ODE?
# - add noise to samples --> Then compare with RNN!

class AdjointODE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func: nn.Module, z0: torch.Tensor, t: torch.Tensor, integrator: RK4):

        with torch.no_grad():
            z_states = integrator.solve(func, z0, t)

        ctx.save_for_backward(z_states.clone(), t)
        ctx.func = func
        ctx.integrator = integrator

        return z_states

    @staticmethod
    def backward(ctx, dLdz):
        # input to backward(): Gradients of loss w.r.t. outputs of forward()

        with torch.no_grad():

            (z_states, t) = ctx.saved_tensors
            func = ctx.func
            integrator = ctx.integrator

            T,B,*z_shape = z_states.size()
            n_dim = np.prod(z_shape)

            aug_state  = [z_states[-1], dLdz[-1]]
            aug_state += [torch.zeros_like(param) for param in func.parameters()]
            aug_state = torch.cat([a.flatten() for a in aug_state])

            def aug_dynamics(aug_state_i, ti):
                # aug_state = [z(t), a_z(t)=dL/dz, a_t(t)=dL/dt, a_p(t)=dL/dp]
                # dim(a_z(t)) = dim(z(t))
                
                zi = aug_state_i[0:B*n_dim]
                adj_zi = aug_state_i[B*n_dim:2*B*n_dim]
                adj_p = aug_state[2*B*n_dim:] # don't need, fill via 

                with torch.enable_grad():

                    zi = zi.requires_grad_(True)
                    adj_zi = adj_zi.requires_grad_(True)

                    # TODO: Why need for prediction? isn't zi already available in context?
                    f_zi = func(zi.view(B,n_dim), ti).flatten()

                    # chain rule: dL/d[z(i-1),t(i-1),p(i-1)] = -previous_sensitivity * dL/d[zi,ti,pi]
                    # general : df(g(x))/dx = df/dg * dg/dx --> df/dg := grad_outputs (i.e. previous_sensitivity)
                    adfdz, *adfdp = torch.autograd.grad(
                        outputs=(f_zi,),
                        inputs = (zi,)+tuple(func.parameters()),
                        grad_outputs=(-adj_zi),
                        allow_unused=True,
                        retain_graph=True )

                adfdz = torch.zeros_like(B*n_dim) if adfdz is None else adfdz
                # adfdt = torch.zeros_like(t) if adfdt is None else adfdt
                adfdp = torch.zeros_like(adj_p) if adfdp is None else torch.cat(
                    [ap.flatten() for ap, p in zip(adfdp, func.parameters())])

                return torch.cat([f_zi, adfdz, adfdp]) # z(t), a_z(t), a_theta(t)

            for i in range(len(t)-1, 0, -1):

                t_inv_step = [t[i], t[i-1]]
                aug_state = integrator.solve(
                    aug_dynamics, aug_state, t_inv_step, last_only=True )

            # hidden states not required ( z_states = aug_state[:B*n_dim].view(B, n_dim) )
            adj_z = aug_state[B*n_dim:B*n_dim*2].view(B, n_dim) # reconstruct from aug_shape

            # Filling up the grad attribute manually --> No passing required in backward pass
            offset = B*n_dim*2 # Here the parameters of 
            for p in func.parameters():
                grad = aug_state[offset : offset + p.shape.numel()].view(p.shape)
                p.grad = grad if p.grad is None else p.grad + grad
                offset += p.shape.numel()
            # Alternative: additionally pass the flattened parameters and return these!
            
            # returned: One for each input of the forward pass in that exact order!
            # grad_func (None, updated above & not needed upstream), grad_z or adj_z (adjoint, required upstream),
            # grad_t (None for equal steps), grad_integrator (Not of interest)
            return None, adj_z, None, None


class NeuralODE(nn.Module):

    def __init__(self, func: nn.Module, integrator: RK4):

        super().__init__()

        self.func = func
        self.integrator = integrator

    def forward(self, z0, t):
        # requires_grad can only be changed outside of torch.autograd.Function, so pass with grad
        z=AdjointODE.apply(self.func, z0.requires_grad_(True), t, self.integrator)
        return z


def train_one_epoch(epoch_nr, model, train_dataloader, optimizer, loss_fn, run_dir, training_points=32, log_every=50):

    last_loss = 1e6
    running_loss = 0.0
    model.train()

    for i, (states,t) in enumerate(train_dataloader):

        states = states.detach(); states.requires_grad_(True)
        t = t.detach(); t.requires_grad_(True)

        optimizer.zero_grad()

        B,T,dim = states.shape
        states = states.permute((1,0,2))
        t = t.permute(1,0)

        start_indices = torch.randint(0, T-training_points, (B,))
        indices = torch.arange(training_points).unsqueeze(1) + start_indices.unsqueeze(0)
        states_short = states[indices, torch.arange(B)]
        # t_short = t[indices, torch.arange(B)]
        first_state = states_short[0]

        # ERROR HERE, NEED TO MAKE NEURAL ODE FIT FOR BATCHED T
        # Neural Network ignores t at the moment, BUT Neural ODE doesn't, it uses t in solver
        # --> Since problem is time-invariant (doesnt matter where we start) we should JUST use t starting at 0 right?
        pred = model(first_state, t[:training_points,0])

        loss = loss_fn(pred,states_short)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i%log_every == log_every-1:
            last_loss = running_loss/log_every

            print(f"Epoch {epoch_nr+1}, Iteration {i+1}, Training loss {last_loss}")
            plt.figure(figsize=(5,5))
            plt.plot(*(np.array(model(first_state, t[:,0])[:,0].view(1,T,dim).detach()).T),label="NeuralODE")
            plt.plot(*(np.array(states[:,0].view(1,T,dim).detach()).T),label="GT")
            plt.legend(loc="upper left")

            filename = f"epoch_{str(epoch_nr+1).zfill(3)}_iter_{str(i+1).zfill(4)}.png"
            file_path = os.path.join(run_dir, filename)
            plt.savefig(file_path)
            plt.close()

            running_loss=0.0

    return model, last_loss



seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

n_dim = 2

end_time_seconds      = 5
num_discrete_steps    = 250
solver_steps_per_step = 1
num_training_points   = 32

batch_size = 4
learning_rate = 0.01 # Too high and it might explode. Too low, it might not learn

model_class = 'MLP'
hidden_layers = [16,]
total_iterations = int(1e6)

ode = pendulum_time_invariant

dt_out = end_time_seconds / num_discrete_steps
dt_solver = end_time_seconds / (num_discrete_steps * solver_steps_per_step)

# Define Models
# NOTE: If the number of neurons (per layer?) is too small, the neuralODE's 
#       prediction is too stiff and makes straight lines.
if model_class == 'MLP':
    model = MLP(n_input_states=n_dim, n_output_states=n_dim, hidden_layers=hidden_layers)

# state_rnn = RNN(input_size=n_dim, hidden_size=32, num_layers=2, output_size=n_dim)
rk4 = RK4(dt=dt_solver) # Not passed to TrajectoryGenerator anymore, maybe change
neural_ode = NeuralODE(func=model, integrator=rk4)

# Prepare Data
dataset = TrajectoryDataset(
    ode=ode, 
    samples_per_epoch=total_iterations, 
    num_timesteps_out=num_discrete_steps, 
    dt_out=dt_out, 
    dt_solver=dt_solver
    )
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

optimizer = torch.optim.Adam(neural_ode.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

eval_dir="eval/node_pendulum"
run_nr = 0
while os.path.exists(os.path.join(eval_dir, f"run_{str(run_nr).zfill(2)}")):
    run_nr += 1
run_dir = os.path.join(eval_dir, f"run_{str(run_nr).zfill(2)}")
os.makedirs(run_dir, exist_ok=True)

settings = {
    'n_dim' : n_dim,
    'end_time_seconds' : end_time_seconds,
    'num_discrete_steps' : num_discrete_steps,
    'solver_steps_per_step' : solver_steps_per_step,
    'batch_size' : batch_size,
    'learning_rate' : learning_rate,
    'model_class': model_class,
    'hidden_layers' : hidden_layers,
    'total_iterations' : total_iterations,
    'run_dir' : run_dir
}

with open(os.path.join(run_dir, 'settings.json'), 'w') as fp:
    json.dump(settings, fp, indent=4)

neural_ode, last_loss = train_one_epoch(
    epoch_nr=0,
    model=neural_ode,
    train_dataloader=dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    run_dir=run_dir,
    log_every=20
    )

torch.save(neural_ode.func.state_dict(), os.path.join(run_dir, 'model.pt'))
print(f'Saved model and settings to {run_dir}')

# Loading the saved model:

model_infer = MLP(n_input_states=n_dim, n_output_states=n_dim, hidden_layers=hidden_layers)
model_infer.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt')))
dt_out = end_time_seconds / num_discrete_steps
dt_solver = end_time_seconds / (num_discrete_steps * solver_steps_per_step)
rk4 = RK4(dt=dt_solver) # Not passed to TrajectoryGenerator anymore, maybe change
neural_ode_infer = NeuralODE(func=model_infer, integrator=rk4)

with torch.no_grad():
    states,t = next(iter(dataloader))
    B,T,dim = states.shape
    first_state_first_sample = states[0][0].unsqueeze(0)
    pred = neural_ode_infer(first_state_first_sample,t[0])