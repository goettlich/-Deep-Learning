import torch
import torch.nn as nn
import math
from integrators import RK4
import numpy as np


class RNN(nn.Module):

    # based on Elman RNN (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
    # but added output layer to be able to output values larger outside of [-1,1]

    def __init__(self, d_in, d_out, d_hidden, n_layers):
        super(RNN, self).__init__()

        self.num_layers  = n_layers
        self.hidden_size = d_hidden

        self.w_ih = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.LayerNorm(d_hidden)
            )
            for _ in range(n_layers)])
        
        self.w_hh = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.LayerNorm(d_hidden)
                )
                for _ in range(n_layers)])
        
        for l_ih, l_hh in zip(self.w_ih, self.w_hh):
            nn.init.uniform_(l_ih[0].weight, -math.sqrt(1 / d_hidden), math.sqrt(1 / d_hidden))
            nn.init.uniform_(l_hh[0].weight, -math.sqrt(1 / d_hidden), math.sqrt(1 / d_hidden))
        
        self.w_ho = nn.Linear(d_hidden, d_out)
        nn.init.uniform_(self.w_ho.weight, -math.sqrt(1 / d_out), math.sqrt(1 / d_out))

    def forward(self, x, infer_n_steps=None):

        if infer_n_steps is None:
            x = x.permute((1,0,2)) # (B,T,D)
            batch_size, seq_len, D = x.shape
        else:
            seq_len = infer_n_steps
            batch_size, D = x.shape 
        
        # h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size) 
        # --> breaks computational graph due to in-place operation on tensors, so take list 
        h_t_minus_1 = [torch.zeros(batch_size, self.hidden_size).to(x.device) 
                       for _ in range(self.num_layers)]
        h_t = h_t_minus_1

        output = []

        for t in range(seq_len):

            if infer_n_steps is None:
                x_in = x[:,t] if t==0 else output[-1]
            else:
                x_in = x if t==0 else output[-1]
            
            for layer in range(self.num_layers):
                h_t[layer] = torch.tanh(
                    self.w_ih[layer](x_in) +
                    self.w_hh[layer](h_t_minus_1[layer])
                )

            # output.append(self.w_ho(h_t[-1]))
            output.append(self.w_ho(h_t[-1]) + x_in) # Residual-block
            h_t_minus_1 = h_t

        output = torch.stack(output)
        output = output.transpose(1,0)

        output = output.permute((1,0,2)) # (T,B,n_dim)
        
        return output #, h_t
    

class MLP(nn.Module):

        def __init__(self, d_in, d_out, hidden_layers=[32]):

            super().__init__()
        
            layer_dims = [d_in] + hidden_layers
            
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(layer_dims[i], layer_dims[i+1]),
                    nn.ELU()#,
                    # nn.LayerNorm(layer_dims[i-1])
                    )
                for i in range(len(layer_dims[:-1]))])
            
            for i,h in enumerate(self.layers):
                nn.init.uniform_(h[0].weight, -math.sqrt(1 / layer_dims[i+1]), math.sqrt(1 / layer_dims[i+1]))

            self.out_layer = nn.Sequential(
                nn.Linear(layer_dims[-1], d_out)
            )
            nn.init.uniform_(self.out_layer[0].weight, -math.sqrt(1 / d_out), math.sqrt(1 / d_out))

        def forward(self, x, t):
        
            for layer in self.layers:
                x = layer(x)
            x = self.out_layer(x)

            return x
        



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