from models import MLP
import torch 

n_dim = 2
mlp = MLP(n_input_states=n_dim, n_output_states=n_dim, hidden_layers=[32,32])
z = torch.rand(3,2)
zgt = torch.rand(3,2)
pred = mlp(z,0.01)

loss = ((pred-zgt)**2).sum()
loss.backward()

print(tuple(m.grad for m in mlp.parameters()))