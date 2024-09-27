import torch
import torch.nn as nn
import math
from integrators import RK4
import numpy as np

class RNN(nn.Module):

    # based on Elman RNN (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
    # but added output layer to be able to output values larger outside of [-1,1]

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        self.w_ih = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(num_layers)])
        
        self.w_hh = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
                )
                for _ in range(num_layers)])
        
        for l_ih, l_hh in zip(self.w_ih, self.w_hh):
            nn.init.uniform_(l_ih[0].weight, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
            nn.init.uniform_(l_hh[0].weight, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
        
        self.w_ho = nn.Linear(hidden_size, output_size)
        nn.init.uniform_(self.w_ho.weight, -math.sqrt(1 / output_size), math.sqrt(1 / output_size))

    def forward(self, x, infer_n_steps=None):

        if infer_n_steps is None:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len = infer_n_steps
            batch_size = 1
        
        # h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size) 
        # --> breaks computational graph due to in-place operation on tensors, so take list 
        h_t_minus_1 = [torch.zeros(batch_size, self.hidden_size).to(x.device) for i in range(self.num_layers)]
        h_t = h_t_minus_1

        output = []

        for t in range(seq_len):
            
            if t==0 or infer_n_steps is None:
                x_in = x[:,t]
            else:
                x_in = output[-1]
            
            for layer in range(self.num_layers):
                h_t[layer] = torch.tanh( # tanh
                    # self.w_ih[layer](x[:,t]) +
                    self.w_ih[layer](x_in) +
                    self.w_hh[layer](h_t_minus_1[layer])
                )

            # output.append(self.w_ho(h_t[-1]))
            output.append(self.w_ho(h_t[-1]) + x_in)
            h_t_minus_1 = h_t

        output = torch.stack(output)
        output = output.transpose(1,0)
        
        return output, h_t
    


class MLP(nn.Module):

        def __init__(self, n_input_states, n_output_states, hidden_layers=[32]):

            super().__init__()
        
            layer_dims = [n_input_states] + hidden_layers
            
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(layer_dims[i-1], layer_dims[i]),
                    nn.Tanh()
                    # nn.LayerNorm(n_hidden[i-1])
                    )
                for i in range(len(layer_dims[1:]))])
            
            for i,h in self.layers[1:]:
                nn.init.uniform_(h.weight[0], -math.sqrt(1 / layer_dims[i+1]), math.sqrt(1 / layer_dims[i+1]))

            self.out_layer = nn.Sequential(
                nn.Linear(layer_dims[-1], n_output_states)
            )
            nn.init.uniform_(self.out_layer[0].weight, -math.sqrt(1 / n_output_states), math.sqrt(1 / n_output_states))

        def forward(self, x0):
        
            x = self.layers(x0)
            x = self.out_layer(x)


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

class AdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z0: torch.Tensor, t: torch.Tensor, func: nn.Module, integrator: RK4):
        # assert isinstance(func, ODEF)
        
        with torch.no_grad:
            z = torch.Tensor(integrator.solve(func, z0, t))
        
        # params = torch.cat([p.flatten() for p in func.parameters()])
        # ctx.save_for_backward(t, z.clone(), params)
        
        ctx.func = func
        ctx.integrator = integrator
        ctx.save_for_backward(t, z.clone())
        return z

    @staticmethod
    def backward(ctx, dLdz):

        with torch.no_grad():

            # t, z, params = ctx.saved_tensors()
            t, z = ctx.saved_tensors()
            func = ctx.func
            integrator = ctx.integrator

            T,B,*z_shape = z.size()
            n_dim = np.prod(z_shape)
            aug_state = torch.cat([z[-1].flatten(), dLdz[-1].flatten()])

            def aug_dynamics(aug_state_i, ti):
                # aug_state = [z(t), az(t)=dL/dz, at(t)=dL/dt, ap(t)=dL/dp]
                # dim(a(t)) = dim(z(t))

                zi = aug_state_i[:n_dim]
                adj_zi = aug_state_i[n_dim:]
                
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
                





            





        

# Questions:
# - can i also build a neural ode without adjoint (why is the odeint() import dependent on the flag "adjoint" --> and if so, why adjoint at all?)


mlp = MLP(2,2)
pass