import torch
from abc import ABC, abstractmethod
from einops import rearrange, repeat, reduce
import matplotlib.pyplot as plt

class VarianceSchedule(ABC):

    def __init__(self, T=1000):

        if type(self) is VarianceSchedule:
            raise TypeError("Base class 'VarianceSchedule' cannot be instantiated")
        self.T = T
        self.betas = None
        self.alphas = None
        self.alphas_cumprod = None
        self.sqrt_alphas_cumprod = None
        self.sqrt_one_minus_alphas_cumprod = None

    def add_noise(self, x0, noise, t):
        
        B,C,H,W = x0.shape 
        sqrt_alphas_cumprod = repeat(
            self.sqrt_alphas_cumprod[t], 'B -> B C H W', B=B, C=C, H=H, W=W)
        sqrt_one_minus_alphas_cumprod = repeat(
            self.sqrt_one_minus_alphas_cumprod[t], 'B -> B C H W', B=B, C=C, H=H, W=W)
        
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
    
    def sample_previous_timestep(self, xt, noise_prediction, t):

        assert torch.all(t == t[0]) # sampling only occurs on batch of same t
        # network-estimated x0 --> might require rearrangement
        x0_est = (xt -(self.sqrt_one_minus_alphas_cumprod[t]) * noise_prediction) / self.sqrt_alphas_cumprod[t]
        x0_est.clamp_(-1.,1.) # inplace

        mean = xt - ((self.betas[t] * noise_prediction) / self.sqrt_one_minus_alphas_cumprod[t])
        mean/= torch.sqrt(self.alphas[t])

        if t.all() == 0:
            return mean, x0_est
        else:
            variance = (1-self.alphas_cumprod[t-1]) / (1. - self.alphas_cumprod[t]) * self.betas[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma*z, x0_est

    def plot_schedule(self, savefig=None):
        plt.figure(figsize=(4,4))
        t = torch.arange(start=1, end=self.T+1, step=1)
        plt.plot(t, self.betas, label=r'$\beta_t$')
        plt.plot(t, self.alphas, label=r'$\alpha_t$')
        plt.plot(t, self.alphas_cumprod, label=r'$\bar{\alpha}_t$')
        plt.legend()
        if savefig is None:
            plt.show()
        else:
            plt.savefig(savefig)

    
class CosineVarianceSchedule(VarianceSchedule):
    """Cosine schedule from annotated transformers."""
    
    def __init__(self, T=1000, s=0.008):
        super().__init__(T=T)

        x = torch.linspace(start=0, end=T, steps=T+1)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.alphas_cumprod = alphas_cumprod[:-1] # simplify, keep same dims, hope thats ok
        # The cumulative product of alphas (alphas_cumprod) is used to determine the noise schedule over time, and betas represent the incremental noise added at each time step. Since betas are differences between consecutive alphas_cumprod values, they naturally have one fewer element

        self.betas = (1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])).clip(0.0001, 0.9999)
        self.alphas = 1-self.betas

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)
        
        
class LinearVarianceSchedule(VarianceSchedule):

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__(T=T)
        self.betas = torch.linspace(start=beta_start, end=beta_end, steps=T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)
        pass


# lin = LinearVarianceSchedule()
# lin.plot_schedule("ddpm/linear_schedule.png")

# cos = CosineVarianceSchedule()
# cos.plot_schedule("ddpm/cosine_schedule.png")
