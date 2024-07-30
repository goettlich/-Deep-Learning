import torch
import torch.nn as nn
from einops import repeat, rearrange

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, t_emb_dim=128):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.time_mlp = nn.Sequential(
                nn.Linear(t_emb_dim, mid_channels),
                nn.SiLU(),
                nn.Linear(mid_channels, mid_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t_emb):
        x1 = self.conv1(x)
        x1 = x1 + rearrange(self.time_mlp(t_emb), 'B midfeatures -> B midfeatures 1 1')
        return self.conv2(x1) + self.residual_conv(x) # make it residual 

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, t_emb_dim)

    def forward(self, x,t_emb):
        x = self.maxpool_conv(x)
        x = self.double_conv(x,t_emb)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Upsample + conv2d --> less params + better performance
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        # )
        self.double_conv = DoubleConv(in_channels, out_channels, t_emb_dim)


    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x, t_emb)


class Unet(nn.Module):
    
    def __init__(self, n_channels, t_emb_dim, img_size, initial_layer=16):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.t_emb_dim = t_emb_dim
        
        # Guarantee that smallest image size doesnt surpass min_latent_size --> better deep and small min_latent_size
        min_latent_size = 4 # minimal latent size of (2,2) 
        multiplications=[1,2,4,8] # Cannot be set from outside currently, needs to be multiplied by 2 all the time
        self.nfeatures = [1]+[initial_layer*m for i,m in enumerate(multiplications) if img_size//2**i >= min_latent_size] 

        self.first_layer = DoubleConv(n_channels, self.nfeatures[1], mid_channels=None, t_emb_dim=t_emb_dim)
        
        self.down = nn.ModuleList([
            Down( self.nfeatures[i], self.nfeatures[i+1], t_emb_dim)
            for i in range(1,len(self.nfeatures)-1)
            ])

        self.up = nn.ModuleList()
        for i in reversed(range(1, len(self.nfeatures)-1)):
            self.up.append(Up(self.nfeatures[i+1], self.nfeatures[i], t_emb_dim))

        self.out = nn.Conv2d(self.nfeatures[1], self.n_channels, kernel_size=1)
        
    def forward(self, x, t):
        
        xi = []
        t_emb = get_time_embedding(time_steps=t, temb_dim=self.t_emb_dim)
        xi.append(self.first_layer(x,t_emb))

        for down_i in self.down:
            xi.append(down_i(xi[-1], t_emb))

        x = xi[-1]
        for i,up_i in enumerate(self.up):    
            x = up_i(x, xi[-2-i], t_emb)
            
        out = self.out(x)
        return out

if __name__=="__main__":
    bs = 16
    x=torch.randn(bs,1,32,32)
    t=torch.randint(0,1000,(bs,))
    model=Unet(n_channels=1, t_emb_dim=128, img_size=32)
    y=model(x,t)
    print(y.shape)