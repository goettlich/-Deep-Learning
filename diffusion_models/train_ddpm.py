import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def plot_noisy_pair(x0, xt, title, idx):
    x0 = torch.tensor(x0)
    xt = torch.tensor(xt)
    
    grid_x0 = make_grid(x0, nrow=4)
    grid_xt = make_grid(xt, nrow=4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(grid_x0.permute(1, 2, 0))
    axes[0].set_title('x0')
    axes[0].axis('off')

    axes[1].imshow(grid_xt.permute(1, 2, 0))
    axes[1].set_title('xt')
    axes[1].axis('off')

    plt.savefig(f"eval/{idx}.png")

    return

transform = torchvision.transforms.transforms.Compose([
    transforms.Resize((16,16)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

T = 1000
beta_schedule = np.linspace(start=1e-4, stop=0.02, num=T)
alpha_schedule = 1 - beta_schedule
alpha_cumprod = np.cumprod(alpha_schedule) 

for i, (batch, label) in enumerate(dataloader):

    x0 = batch.numpy()
    t = np.random.randint(0,T, size=(x0.shape[0]))
    noise = np.random.normal(size=x0.shape)
    alpha = alpha_cumprod[t]
    xt = x0.copy()
    for j in range(xt.shape[0]):
        xt[j] = np.sqrt(alpha[j]) * x0[j] + (1-alpha[j]) * noise[j] # in terms of x0
    
    plot_noisy_pair(x0, xt, f"alpha: {alpha}, beta: {1- alpha}", i)
    if i==10:
        break

    # predict noise, given (xt[B,1,W,H],t[B])
    

    


    
