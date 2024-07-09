# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:00:32 2022

@author: c
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import scipy.io as scio
import time

class DC1(nn.Module):
    def __init__(self, inc, interc, outc, k_size, CNN_act):
        super(DC1, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = inc, out_channels = interc, kernel_size = k_size, stride = 1, padding = (k_size-1)//2, dilation = 1)
        self.cnn2 = nn.Conv2d(in_channels = interc, out_channels = outc, kernel_size = k_size, stride = 1, padding = (k_size-1)//2, dilation = 1)
        self.act = CNN_act
        if self.act == 'tanh':
            self.activation = nn.Tanh()
        elif self.act == 'gelu':
            self.activation = F.gelu
        elif self.act == 'relu':
            self.activation = nn.ReLU()
        else:
            print('wrong activation')
    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        # x = self.activation(x)
        return x
    
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, FNO_act, CNN_act, DC_in, cin):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.cin = cin
        self.DC_in = DC_in
        self.padding = 9 # pad the domain if input is non-periodic
        
        self.FNO_act = FNO_act
        self.CNN_act = CNN_act
        if self.FNO_act == 'tanh':
            self.activation = nn.Tanh()
        elif self.FNO_act == 'gelu':
            self.activation = F.gelu
        elif self.FNO_act == 'relu':
            self.activation = nn.ReLU()
        else:
            print('wrong activation')
        
        self.fc0 = nn.Linear(self.cin + 2, self.width) # input is (cin, x, y)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)
        
        if self.DC_in:
            self.DC = DC1(self.cin, 8, self.cin, 3, CNN_act = self.CNN_act)

    def forward(self, x):
        if self.DC_in:
            x = self.DC(x)
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def get_grid(self, shape, device):
        batchsize, channel, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        x = np.linspace(0, 1, size_x)
        y = np.linspace(0, 1, size_y)
        X, Y = np.meshgrid(x, y)
        X = torch.tensor(X, dtype=torch.float).to(device)
        Y = torch.tensor(Y, dtype=torch.float).to(device)
        gridx = X.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        gridy = Y.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        return torch.cat((gridx, gridy), dim=1)
    
class Neumann_FNO(nn.Module):
    def __init__(self, N, k, modes1, modes2, width, FNO_act, CNN_act, DC_in):
        super(Neumann_FNO, self).__init__()
        
        self.k = k
        self.N = N
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.FNO_act = FNO_act
        self.CNN_act = CNN_act
        self.DC_in = DC_in
        
        self.fno1 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, FNO_act = self.FNO_act, 
                          CNN_act = self.CNN_act, DC_in = self.DC_in, cin = 1)
        self.fno2 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, FNO_act = self.FNO_act, 
                          CNN_act = self.CNN_act, DC_in = self.DC_in, cin = 2)
        self.fno3 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, FNO_act = self.FNO_act, 
                          CNN_act = self.CNN_act, DC_in = self.DC_in, cin = 2)

    def forward(self, x):
        q = x[:,0,:,:]
        f = x[:,1,:,:]
        f = f.unsqueeze(1)
        u0 = self.fno1(f)
        
        q = q.unsqueeze(1)
        u1 = self.fno2(-self.k**2*q*u0)

        u2 = self.fno3(-self.k**2*q*u1)
        
        u = u0 + u1 + u2
        return u

class Neumann_FNO_mnist(nn.Module):
    def __init__(self, N, k, modes1, modes2, width, FNO_act, CNN_act, DC_in):
        super(Neumann_FNO_mnist, self).__init__()
        
        self.k = k
        self.N = N
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.FNO_act = FNO_act
        self.CNN_act = CNN_act
        self.DC_in = DC_in
        
        self.fno1 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, FNO_act = self.FNO_act, 
                          CNN_act = self.CNN_act, DC_in = self.DC_in, cin = 2)
        self.fno2 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, FNO_act = self.FNO_act, 
                          CNN_act = self.CNN_act, DC_in = self.DC_in, cin = 2)
        self.fno3 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, FNO_act = self.FNO_act, 
                          CNN_act = self.CNN_act, DC_in = self.DC_in, cin = 2)

    def forward(self, x):
        q = x[:,0,:,:]
        f = x[:,1:3,:,:]
        # f = f.unsqueeze(1)
        u0 = self.fno1(f)
        
        q = q.unsqueeze(1)
        u1 = self.fno2(-self.k**2*q*u0)

        u2 = self.fno3(-self.k**2*q*u1)
        
        u = u0 + u1 + u2
        return u

class FNO_iteration(nn.Module):
    def __init__(self, modes1, modes2, width, FNO_act, num_layers = 4):
        super(FNO_iteration, self).__init__()

        layers = [FNOBlock(modes1, modes2, width, FNO_act, if_activation = True) for _ in range(num_layers - 1)]
        layers.append(FNOBlock(modes1, modes2, width, FNO_act, if_activation = False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class FNOBlock(nn.Module):
    def __init__(self, modes1, modes2, width, FNO_act, if_activation):
        super(FNOBlock, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.FNO_act = FNO_act
        self.if_activation = if_activation
        
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w = nn.Conv2d(self.width, self.width, 1)
        
        if self.FNO_act == 'tanh':
            self.activation = nn.Tanh()
        elif self.FNO_act == 'gelu':
            self.activation = F.gelu
        elif self.FNO_act == 'relu':
            self.activation = nn.ReLU()
        else:
            print('wrong activation')

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if self.if_activation:
            x = self.activation(x)
        
        return x

class EB_block(nn.Module):
    def __init__(self, cin, width, CNN_act, DC_in):
        super(EB_block, self).__init__()
        
        self.cin = cin
        self.CNN_act = CNN_act
        self.DC_in = DC_in
        
        if self.DC_in:
            self.DC = DC1(inc = self.cin, interc = 8, outc = self.cin, k_size = 3, CNN_act = self.CNN_act)
        
        self.width = width
        
        self.up = nn.Sequential(
            nn.Conv2d(self.cin+2, self.width//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.width//2, self.width//2, kernel_size=1, stride=1),
            nn.Conv2d(self.width//2, self.width//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.width//2, self.width, kernel_size=1, stride=1)
        )

    def forward(self, x):
        if self.DC_in:
            x = self.DC(x)
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        
        x = self.up(x)
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, channel, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        x = np.linspace(0, 1, size_x)
        y = np.linspace(0, 1, size_y)
        X, Y = np.meshgrid(x, y)
        X = torch.tensor(X, dtype=torch.float).to(device)
        Y = torch.tensor(Y, dtype=torch.float).to(device)
        gridx = X.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        gridy = Y.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        return torch.cat((gridx, gridy), dim=1)

class Upblock(nn.Module):
    def __init__(self, upc1, upc2, convc1, convc2):
        super(Upblock, self).__init__()
        
        self.upc1 = upc1
        self.upc2 = upc2
        self.convc1 = convc1
        self.convc2 = convc2
        
        self.up = nn.ConvTranspose2d(self.upc1, self.upc2, kernel_size = 3, stride = 2, padding = 1)
        self.conv1 = nn.Conv2d(self.convc1, self.convc2, kernel_size=1, stride=1)

    def forward(self, z1, z2):
        
        z2 = self.up(z2)
        z = self.conv1(torch.cat([z1, z2], dim=1))
        
        return z
    
    def get_grid(self, shape, device):
        batchsize, channel, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        x = np.linspace(0, 1, size_x)
        y = np.linspace(0, 1, size_y)
        X, Y = np.meshgrid(x, y)
        X = torch.tensor(X, dtype=torch.float).to(device)
        Y = torch.tensor(Y, dtype=torch.float).to(device)
        gridx = X.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        gridy = Y.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        return torch.cat((gridx, gridy), dim=1)    
    
class MyUNO(nn.Module):
    def __init__(self, N, k, modes1, modes2, width, FNO_act, CNN_act, DC_in, num_layer, cin):
        super(MyUNO, self).__init__()
        
        self.k = k
        self.N = N
        self.cin = cin
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.FNO_act = FNO_act
        self.CNN_act = CNN_act
        self.DC_in = DC_in
        self.num_layer = num_layer
        
        self.EB1 = EB_block(cin = self.cin, width = self.width//4, CNN_act = self.CNN_act, DC_in = self.DC_in)
        self.EB2 = EB_block(cin = self.cin, width = self.width//2, CNN_act = self.CNN_act, DC_in = self.DC_in)
        self.EB3 = EB_block(cin = self.cin, width = self.width, CNN_act = self.CNN_act, DC_in = self.DC_in)
        
        self.Down1 = nn.Conv2d(self.width//4, self.width//2, kernel_size = 3, stride = 2, padding = 1)
        self.Down2 = nn.Conv2d(self.width//2, self.width, kernel_size = 3, stride = 2, padding = 1)
        
        self.conv1 = nn.Conv2d(self.width, self.width//2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(self.width*2, self.width, kernel_size=1, stride=1)
        
        self.fno1 = FNO_iteration(modes1 = self.modes1, modes2 = self.modes2, width = self.width//4,
                                  FNO_act = self.FNO_act, num_layers = self.num_layer)
        self.fno2 = FNO_iteration(modes1 = self.modes1, modes2 = self.modes2, width = self.width//2,
                                  FNO_act = self.FNO_act, num_layers = self.num_layer)
        self.fno3 = FNO_iteration(modes1 = self.modes1, modes2 = self.modes2, width = self.width,
                                  FNO_act = self.FNO_act, num_layers = self.num_layer)
        
        self.up1 = Upblock(self.width, self.width//2, self.width, self.width//2)
        self.up2 = Upblock(self.width//2, self.width//4, self.width//2, self.width//4)

        self.out = nn.Conv2d(self.width//4, 2, kernel_size = 3, stride = 1, padding = 1)

        # self.out = DC1(self.width//4, self.width//4, 2, k_size = 3, CNN_act = self.CNN_act)

    def forward(self, x):
        
        m = nn.AvgPool2d(kernel_size = 2, stride=(2, 2))
        x_2 = F.pad(x, (1, 0, 1, 0), "replicate")
        x_2 = m(x_2)
        x_4 = F.pad(x_2, (1, 0, 1, 0), "replicate")
        x_4 = m(x_4)

        # x_2 = x[:, :, ::2, ::2]
        # x_4 = x[:, :, ::4, ::4]
        
        x = self.EB1(x) # 257*257*w
        
        x2 = torch.cat([self.EB2(x_2), self.Down1(x)], dim = 1) # 129*129*4w
        x2 = self.conv1(x2) # 129*129*2w
        
        x4 = torch.cat([self.EB3(x_4), self.Down2(x2)], dim = 1) # 65*65*8w
        x4 = self.conv2(x4) # 65*65*4w
        
        z = self.fno1(x) # 257*257*w
        z2 = self.fno2(x2) # 129*129*2w
        z4 = self.fno3(x4) # 65*65*4w
        
        z2 = self.up1(z2, z4) # 129*129*2w
        z = self.up2(z, z2) # 257*257*w

        z = self.out(z) # 257*257*2
        
        return z

class Neumann_MyUNO(nn.Module):
    def __init__(self, N, k, modes1, modes2, width, FNO_act, CNN_act, DC_in, num_layer):
        super(Neumann_MyUNO, self).__init__()
        
        self.N = N
        self.k = k
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.FNO_act = FNO_act
        self.CNN_act = CNN_act
        self.DC_in = DC_in
        self.num_layer = num_layer

        self.myuno1 = MyUNO(N, k, modes1 = self.modes1, modes2 = self.modes2, width = self.width,
                             FNO_act = self.FNO_act, CNN_act = CNN_act, DC_in = self.DC_in,
                             num_layer = self.num_layer, cin = 1)
        
        self.myuno2 = MyUNO(N, k, modes1 = self.modes1, modes2 = self.modes2, width = self.width,
                             FNO_act = self.FNO_act, CNN_act = CNN_act, DC_in = self.DC_in,
                             num_layer = self.num_layer, cin = 2)
        
        self.myuno3 = MyUNO(N, k, modes1 = self.modes1, modes2 = self.modes2, width = self.width,
                             FNO_act = self.FNO_act, CNN_act = CNN_act, DC_in = self.DC_in,
                             num_layer = self.num_layer, cin = 2)

    def forward(self, x):
        q = x[:,0,:,:]
        f = x[:,1,:,:]
        f = f.unsqueeze(1)
        u0 = self.myuno1(f)
        
        q = q.unsqueeze(1)
        u1 = self.myuno2(-self.k**2*q*u0)
        
        u2 = self.myuno3(-self.k**2*q*u1)
        
        u = u0 + u1 + u2
        return u
    
class Neumann_MyUNO_mnist(nn.Module):
    def __init__(self, N, k, modes1, modes2, width, FNO_act, CNN_act, DC_in, num_layer):
        super(Neumann_MyUNO_mnist, self).__init__()
        
        self.N = N
        self.k = k
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.FNO_act = FNO_act
        self.CNN_act = CNN_act
        self.DC_in = DC_in
        self.num_layer = num_layer

        self.myuno1 = MyUNO(N, k, modes1 = self.modes1, modes2 = self.modes2, width = self.width,
                             FNO_act = self.FNO_act, CNN_act = CNN_act, DC_in = self.DC_in,
                             num_layer = self.num_layer, cin = 2)
        
        self.myuno2 = MyUNO(N, k, modes1 = self.modes1, modes2 = self.modes2, width = self.width,
                             FNO_act = self.FNO_act, CNN_act = CNN_act, DC_in = self.DC_in,
                             num_layer = self.num_layer, cin = 2)
        
        self.myuno3 = MyUNO(N, k, modes1 = self.modes1, modes2 = self.modes2, width = self.width,
                             FNO_act = self.FNO_act, CNN_act = CNN_act, DC_in = self.DC_in,
                             num_layer = self.num_layer, cin = 2)

    def forward(self, x):
        q = x[:,0,:,:]
        f = x[:,1:3,:,:]
        u0 = self.myuno1(f)
        
        q = q.unsqueeze(1)
        u1 = self.myuno2(-self.k**2*q*u0)
        
        u2 = self.myuno3(-self.k**2*q*u1)
        
        u = u0 + u1 + u2
        return u