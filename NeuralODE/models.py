import torch
import torch.nn as nn
import math

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