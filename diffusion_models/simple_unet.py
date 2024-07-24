import torch.nn as nn
import torch.nn.functional as F
import torch

# Modified from https://github.com/obravo7/satellite-segmentation-pytorch/blob/master/models/unet.py

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layer(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)


class SegmentationUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SegmentationUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.first_layer = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.first_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out(x)
        return out
    


class SmallSegmentationUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SmallSegmentationUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.first_layer = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16) # maxpool -> doubleconv
        self.down2 = Down(16, 32) # maxpool -> doubleconv

        factor = 2 if bilinear else 1
        self.down3 = Down(32, 64 // factor)
        
        self.up1 = Up(64, 32 // factor, bilinear) # upsample -> doubleconv
        self.up2 = Up(32, 16 // factor, bilinear)  # upsample -> doubleconv
        self.up3 = Up(16,8, bilinear)

        self.out = OutConv(8, self.n_classes)

    def forward(self, x):

        x1 = self.first_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = self.out(x)
        return out
    



class SmallUNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(SmallUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.first_layer = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16) # maxpool -> doubleconv
        self.down2 = Down(16, 32) # maxpool -> doubleconv

        factor = 2 if bilinear else 1
        self.down3 = Down(32, 64 // factor)
        
        self.up1 = Up(64, 32 // factor, bilinear) # upsample -> doubleconv
        self.up2 = Up(32, 16 // factor, bilinear)  # upsample -> doubleconv
        self.up3 = Up(16,8, bilinear)

        self.out = OutConv(8, self.n_channels)

    def forward(self, x):

        x1 = self.first_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = self.out(x)
        return out



class SmallDiffstepConditionedUNet(nn.Module):
    
    # model conditioned on noise in first layer
    # using embedding is not elegant because 
    # 1) we would need to pre-define the embedding size
    # (e.g. nn.Linear(1,x.width*x.height)) --> so the model could only work for one size
    # 2) a senseless learnable layer
    # instead, we just broadcast t into the respective shape in the forward pass 
    # (and we scale it to a range between [0,1] to keep things nice for the network)
    
    def __init__(self, n_channels, n_diffsteps, bilinear=True):
        super(SmallDiffstepConditionedUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_diffsteps = n_diffsteps

        self.first_layer = DoubleConv(n_channels + 1, 8)
        self.down1 = Down(8, 16) # maxpool -> doubleconv
        self.down2 = Down(16, 32) # maxpool -> doubleconv

        factor = 2 if bilinear else 1
        self.down3 = Down(32, 64 // factor)
        
        self.up1 = Up(64, 32 // factor, bilinear) # upsample -> doubleconv
        self.up2 = Up(32, 16 // factor, bilinear)  # upsample -> doubleconv
        self.up3 = Up(16,8, bilinear)

        self.out = OutConv(8, self.n_channels)

    def forward(self, x, t):
        
        condition = torch.ones_like(x)*t.view(t.shape[0],1,1,1)/self.n_diffsteps
        x = torch.concat([x,condition], dim=1)

        x1 = self.first_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = self.out(x)
        return out
    


class SmallMultiDiffstepConditionedUNet(nn.Module):
    
    # model conditioned on noise in first layer
    # using embedding is not elegant because 
    # 1) we would need to pre-define the embedding size
    # (e.g. nn.Linear(1,x.width*x.height)) --> so the model could only work for one size
    # 2) a senseless learnable layer
    # instead, we just broadcast t into the respective shape in the forward pass 
    # (and we scale it to a range between [0,1] to keep things nice for the network)
    
    def __init__(self, n_channels, n_diffsteps, initial_layer=32, multiplications=[1,2,4,8], bilinear=True):
        super(SmallMultiDiffstepConditionedUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_diffsteps = n_diffsteps
        features = [initial_layer*m for m in multiplications] 
        
        self.first_layer = DoubleConv(n_channels + 1, features[0])
        self.down1 = Down(features[0] + 1, features[1]) # maxpool -> doubleconv
        self.down2 = Down(features[1] + 1, features[2]) # maxpool -> doubleconv

        factor = 2 if bilinear else 1
        self.down3 = Down(features[2] + 1, features[3] // factor)
        
        self.up1 = Up(features[3]+1, features[2] // factor, bilinear) # upsample -> doubleconv
        self.up2 = Up(features[2]+1, features[1] // factor, bilinear)  # upsample -> doubleconv
        self.up3 = Up(features[1]+1, features[0], bilinear)

        self.out = OutConv(features[0], self.n_channels)

    def forward(self, x, t):
        
        B,C,H,W = x.shape
        t = (t/self.n_diffsteps).clone()
        t0 = t.view(B,1,1,1).expand(B,1,H,W)
        x = torch.cat([x,t0], dim=1)
        x1 = self.first_layer(x)
        
        B,C,H,W = x1.shape
        t1 = t.view(B,1,1,1).expand(B,1,H,W)
        x1 = torch.cat([x1,t1], dim=1)
        x2 = self.down1(x1)

        B,C,H,W = x2.shape
        t2 = t.view(B,1,1,1).expand(B,1,H,W)
        x2 = torch.cat([x2,t2], dim=1)
        x3 = self.down2(x2)

        B,C,H,W = x3.shape
        t3 = t.view(B,1,1,1).expand(B,1,H,W)
        x3 = torch.cat([x3,t3], dim=1)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = self.out(x)
        return out
    







