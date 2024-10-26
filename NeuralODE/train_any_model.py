import torch
import torch.nn as nn
from integrators import IntegratorFactory
import numpy as np
from models import MLP, NeuralODE, RNN
from dataset import TrajectoryDataset
from systems import pendulum_time_invariant
import matplotlib.pyplot as plt
import random
import os
from omegaconf import OmegaConf
from fire import Fire
from tqdm import tqdm
from generate_gif import generate_gif

# NOTE:
# - What made the difference: Batched samples (only use random-indexed part of the time sequence) 
# --> This way the model gets to see what is happening farer inside of the spiral --> otherwise we have some vanishing gradient there 
# --> Find out why that exactly is the case theoretically (shouldnt vanishing gradient apply to first steps due to backwards adjoint?)

# TODO
# - Get it to work for RNN TOO
# - add noise to samples

class ModelFactory:
    """
    Factory class to instantiate different models based on their names.
    """
    @staticmethod
    def get_model(model_name: str, d_in: int, d_out: int, model_weights_fn: str = None, train: bool = True, **kwargs) -> nn.Module:
        
        if model_name == "NODE-MLP":
            integrator = IntegratorFactory().get_integrator(
                kwargs['integrator_name'], dt_solver=kwargs['dt_solver'])
            internal = MLP(d_in=d_in, d_out=d_out, hidden_layers=kwargs['hidden_layers'])
            if model_weights_fn is not None:
                internal.load_state_dict(torch.load(model_weights_fn))
            if train:
                internal.train()
            else:
                internal.eval()
            model = NeuralODE(func=internal, integrator=integrator)
        
        elif model_name == "RNN":
            model = RNN(d_in=d_in, d_out=d_out, d_hidden=kwargs['hidden_size'], n_layers=kwargs['num_layers'])
            if model_weights_fn is not None:
                model.load_state_dict(torch.load(model_weights_fn))
            if train:
                model.train()
            else:
                model.eval()
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return model
    
    @staticmethod
    def get_model_statedict(model_name: str, model):
        if model_name == "NODE-MLP":
            return model.func.state_dict()
        elif model_name == "RNN":
            return model.state_dict()
    
    @staticmethod
    def get_model_input(model_name: str, x,t):
        inputs = {'NODE-MLP': (x[:,0], t[0]), 'RNN': (x,)}
        return inputs[model_name]
    
    @staticmethod
    def get_model_output(model_name: str, x):
        outputs = {'NODE-MLP': x.permute((1,0,2)), 'RNN': x}
        return outputs[model_name]
    

def seed_all(seed):

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(exp_dir, system, system_cfg, model_names, model_cfg, train_cfg, model_name):

    seed_all(train_cfg.seed) # Make sure different Model initializations still use the same seed (for display data)

    T,B = system_cfg[system].n_sample_steps, train_cfg.batch_size

    if system == 'pendulum':
        n_dim = 2
        ode = pendulum_time_invariant
    elif system == 'lorenz':
        n_dim = 3
        ode = ...

    factory = ModelFactory()
    model = factory.get_model(model_name=model_name, d_in=n_dim, d_out=n_dim, **model_cfg[model_name])

    dataset = TrajectoryDataset(
        ode=ode, 
        samples_per_epoch=train_cfg.training_iters, 
        num_timesteps_out=T, 
        dt_solver=system_cfg[system].dt_solver, 
        dt_out=system_cfg[system].dt_out
        )
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=train_cfg.batch_size)

    if model_cfg[model_name]['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg[model_name]['learning_rate'])
    elif model_cfg[model_name]['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=model_cfg[model_name]['learning_rate'])
    loss_fn = nn.MSELoss()

    last_loss = 1e6
    running_loss = 0.0

    result_gt, time_gt = next(iter(dataloader)) # (B,T,n_dim), (B,T)
    result_model = []

    with tqdm(total=len(dataloader)) as pbar:

        for i, (states,t) in enumerate(dataloader):

            optimizer.zero_grad()
            pbar.set_description(f'Training model {model_name}, iter: {i} / {len(dataloader)}, last loss: {last_loss:.3f}.')

            states = states.detach(); states.requires_grad_(True)
            t = t.detach(); t.requires_grad_(True)

            def get_subset(x, t, T_subset):
                B, T, D = x.shape
                start_indices = torch.randint(0, T_subset, (B,))
                indices = torch.arange(T_subset).unsqueeze(0) + start_indices.unsqueeze(1)
                return states[torch.arange(B).unsqueeze(1), indices], t[torch.arange(B).unsqueeze(1), indices]
            
            states_subset, t_subset = get_subset(states, t, train_cfg.points_per_sample)

            pred = model(*factory.get_model_input(model_name, states_subset, t_subset))
            loss = loss_fn( factory.get_model_output(model_name,pred), states_subset)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i%train_cfg.log_every == train_cfg.log_every-1:

                last_loss = running_loss/train_cfg.log_every
                with torch.no_grad():

                    pred = model(*factory.get_model_input(model_name, states, t))
                    result_model.append(factory.get_model_output(model_name, pred))

                running_loss=0.0

            pbar.update()
    
    model_weights_fn = os.path.join(exp_dir, 'model.pt')
    torch.save(factory.get_model_statedict(model_name, model), model_weights_fn)
    print(f'Saved model and settings to {exp_dir}')

    return result_gt, result_model


def inference(exp_dir, model_name, model_weights_fn):

    config = OmegaConf.load(os.path.join(exp_dir, 'config.yml'))
    dim = 2 if config.system == 'pendulum' else 3
    
    model = ModelFactory().get_model(
        model_name=model_name, 
        d_in=dim, 
        d_out=dim, 
        model_weights_fn=model_weights_fn, 
        train=False, 
        **config.model_cfg[model_name]
        )
    
    return model


def main(config=None, **kwargs):
    
    config = OmegaConf.load(config) if (config is not None) else {}
    subfolder = 'NeuralODE'
    assert config, (f'No config provided, use python {subfolder}/train.py --config={subfolder}/config.yml,' 
                    'or other vscode launch configuration')
    run = 0
    basename = os.path.join(subfolder, 'eval', f'run_')
    while os.path.exists(basename + str(run).zfill(2)):
        run +=1
    config.exp_dir = basename + str(run).zfill(2)
    os.makedirs(config.exp_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.exp_dir, 'config.yml'))

    print('\nConfiguration:         \n--------------------\n'
         f'{OmegaConf.to_yaml(config)}--------------------\n')

    results = {}
    for mname in config.model_names:
        results_gt, results_model = train(**config, model_name=mname)
        results['ground_truth'] = results_gt
        results[mname] = results_model

    generate_gif(results, config.model_names, config.train_cfg.log_every, exp_dir=config.exp_dir)


if __name__ == "__main__":
    Fire(main)



