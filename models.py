import os


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import math

from hyper import *
from util import *

class Resblock(nn.Module):
    def __init__(self, channel):
        super(Resblock, self).__init__()
        self.channel = channel
        self.left = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(channel)
        )
        
    def forward(self, x):
        out = self.left(x)
        out += x
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = Resblock(192)
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=(1,1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.resblock2 = Resblock(192)
        self.conv3 = nn.Conv2d(192, c, kernel_size=3, stride=2, padding=(1,1), bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.conv2(out)
        out = self.resblock2(out)
        out = self.conv3(out)
        # out: [batchsz, c, 64, 64]
        return out
    
class Importance(nn.Module):
    def __init__(self):
        super(Importance, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = Resblock(192)
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=(1,1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.resblock2 = Resblock(192)
        self.conv3 = nn.Conv2d(192, 1, kernel_size=3, stride=2, padding=(1,1), bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.conv2(out)
        out = self.resblock2(out)
        out = self.conv3(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 192, kernel_size=3, stride=1, padding=(1,1), bias=True),
            nn.PixelShuffle(2), 
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.resblock1 = Resblock(48)
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 192, kernel_size=3, stride=1, padding=(1,1), bias=True),
            nn.PixelShuffle(4), # note sure
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        self.resblock2 = Resblock(12)
        self.conv3 = nn.Conv2d(12, 3, kernel_size=3, stride=2, padding=(1,1),bias=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.conv2(out)
        out = self.resblock2(out)
        out = self.conv3(out)
        return out
   
# this module is from "vector quantized VAE"
# from vector_quantize_pytorch import VectorQuantize

# vq = VectorQuantize(
#     dim = 1,
#     n_embed = math.log(c, 2),   # size of the dictionary
#     decay = 0.8,                # the exponential moving average decay, lower means the dictionary will change faster
#     commitment = 1.             # the weight on the commitment loss
# )

class L2C(nn.Module):
    # for summary only
    def __init__(self, device):
        super(L2C, self).__init__()
        
        self.encoder = Encoder()
        self.importance = Importance()
        self.decoder = Decoder()
        
        self.device = device
        self._round_straightthrough = RoundStraightThrough().apply
    
    def forward(self, x):
        # encoder
        y = self.encoder(x)
        
        # importance
        tau = normalize(self.importance(x)) 
        # tau, _, _ = quantized, indices, commit_loss = vq(tau)
        tau, _, _ = self._round_straightthrough(tau, c)
        tau = tau * c
        tau = tau.expand(tau.shape[0], c, tau.shape[2], tau.shape[3])
        
        cond = torch.linspace(0, c-1, c).to(self.device)
        cond = cond.reshape(-1,1,1).repeat(1,tau.shape[2],tau.shape[3])
        m = torch.where(tau < cond, torch.tensor(1.).to(self.device), torch.tensor(0.).to(self.device))
        
        # y_telta
        y_telta = torch.mul(y, m)
        z, _s, _z = self._round_straightthrough(y_telta, b)
        y_hat = dequentize(z, _s, _z) 
        
        # decoder
        x_hat = self.decoder(y_hat)
        
        return y, m, z, x_hat
        
class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu2_2, h_relu4_3
 