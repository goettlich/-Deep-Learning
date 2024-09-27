from models import RNN
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from solvers import ODESolve
from systems import pendulum_time_invariant, lorenz
import torch
import torch.nn as nn


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Provides a standard configuration dataset for different physics problems + solvers 
    """
    # TODO: 
    # - pass parameters to integrator (dt, ...)
    # - pass parameters to ode (physics parameters, ...)
    def __init__(self, ode, integrator, num_samples=100, num_timesteps=5000):
        self.num_timesteps = num_timesteps
        self.num_samples = num_samples
        self.ode = ode
        self.integrator = integrator

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        torch.manual_seed(idx)
        
        if self.ode is pendulum_time_invariant:
            start = torch.rand(2)-0.5
            dt = 0.07
        
        elif self.ode is lorenz:
            perturbation = 1e-2
            start = torch.tensor([0,1.,1.]) + (torch.rand(3)-0.5)*perturbation
            dt = 0.01
        else:
            raise(NotImplementedError)
        
        t = np.arange(start=0, stop=self.num_timesteps*dt, step=dt)
        trajectory = ODESolve(f=self.ode, y0=start, t=t, dt=dt)
        return torch.tensor(trajectory).float()

def train_one_epoch(epoch_nr, model, train_dataloader, optimizer, loss_fn, run_dir, log_every=100):

    running_loss = 0.0
    model.train()

    for i, states in enumerate(train_dataloader):

        states.requires_grad_(True)
        optimizer.zero_grad()
      
        pred, _ = model(states)

        loss = loss_fn(pred,states)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i%log_every == log_every-1:
            last_loss = running_loss/log_every

            print(f"Epoch {epoch_nr+1}, Iteration {i+1}, Training loss {last_loss}")
            plt.figure(figsize=(5,5))
            plt.plot(*(np.array(pred[0].detach()).T),label="RNN")
            plt.plot(*(np.array(states[0].detach()).T),label="GT")
            plt.legend(loc="upper left")

            filename = f"epoch_{str(epoch_nr+1).zfill(3)}_iter_{str(i+1).zfill(4)}.png"
            file_path = os.path.join(run_dir, filename)
            plt.savefig(file_path)
            plt.close()

            running_loss=0.0
        
    return last_loss

learning_rate = 0.01 # Too high and it might explode. Too low, it might not learn
n_hidden = 64
n_states = 2
n_layers = 1

model = RNN(input_size=n_states, hidden_size=n_hidden, num_layers=n_layers, output_size=n_states)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.MSELoss()

dataset_pendulum = TrajectoryDataset(integrator='rk4', ode=pendulum_time_invariant, num_timesteps=100, num_samples=2000)
dataloader_pendulum = torch.utils.data.DataLoader(dataset=dataset_pendulum, batch_size=1, shuffle=False)

eval_dir="eval/rnn_pendulum"
run_nr = 0
while os.path.exists(os.path.join(eval_dir, f"run_{str(run_nr).zfill(2)}")):
    run_nr += 1
run_dir = os.path.join(eval_dir, f"run_{str(run_nr).zfill(2)}")
os.makedirs(run_dir, exist_ok=True)

train_one_epoch(
    epoch_nr=0, 
    model=model, 
    train_dataloader=dataloader_pendulum, 
    optimizer=optimizer, 
    loss_fn=loss_fn, 
    run_dir=run_dir,
    log_every=10
    )