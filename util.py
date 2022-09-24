import torch
# from util import *
from pytorch_msssim import ms_ssim
from torch import nn
from torchsummary import summary
from hyper import *
import numpy as np

def normalize(x):
    # normalize a vector s.t each element is in [0,1]
    min_x = torch.min(x)
    max_x = torch.max(x)
    range_x = max_x - min_x
    
    return (x - min_x) / range_x


# https://arxiv.org/pdf/2004.09602.pdf
# should change to https://arxiv.org/pdf/1703.00395.pdf
# https://github.com/alexandru-dinu/cae/blob/master/models/cae_16x16x16_zero_pad_bin.py
# def quantize(x, n_bits):
#     max_x = torch.max(x)
#     min_x = torch.min(x)
    
#     _s = (2**n_bits -1) / (max_x-min_x)
#     _z = -torch.round(min_x*_s)-2**(n_bits-1)
#     # f(x) = sx + z
#     return torch.clamp(_s*x+_z, -2**(n_bits-1), 2**(n_bits-1)-1), _s, _z

class RoundStraightThrough(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input, n_bits):
        # rounded = torch.round(input, out=None)
        max_x = torch.max(input)
        min_x = torch.min(input)
        
        _s = (2**n_bits -1) / (max_x-min_x)
        _z = -torch.round(min_x*_s)-2**(n_bits-1)
        # f(x) = sx + z
        return torch.clamp(_s*input+_z, -2**(n_bits-1), 2**(n_bits-1)-1), _s, _z
        # return rounded

    @staticmethod
    def backward(ctx, grad_output, _s, _z):
        
        grad_input = grad_output.clone()
        
        # grad_s     = grad_s.clone()
        # grad_z     = grad_z.clone()
        # return grad_input , grad_s , grad_z
        return grad_input, None



def dequentize(xq, s, z):
    return (xq-z)/s

def calculate_loss_M(m, tau, device, batchsz):
    # nonzero_ratio = torch.nonzero(m).shape[0] / torch.numel(m)
    # print(nonzero_ratio)
    # loss = torch.sum(torch.tensor([torch.abs(torch.mean(tau[batchidx])-nonzero_ratio) for batchidx in range(num_tasks)])).to(device)
    # loss.requires_grad = True
    return torch.sum(torch.tensor([torch.abs(torch.mean(tau[batchidx])-nonzero_ratio) for batchidx in range(batchsz)])).to(device)
    
    #return loss
        
def calculate_m(x_spt, importance, round_straightthrough, device):

    tau = normalize(importance(x_spt))
    # tau: [batchsz, 1, 64, 64]
    tau_ = tau
    
    #tau_, _, _ = round_straightthrough(tau_, c)
    tau_ = tau_ * c
    # tau_: [batchsz, 1, 64, 64]
    tau_ = tau_.expand(tau_.shape[0], c, tau_.shape[2], tau_.shape[3])
    # tau_: [batchsz, c, 64, 64]
    cond = torch.linspace(0, c-1, c).to(device)
    cond = cond.reshape(-1,1,1).repeat(tau_.shape[0], 1, tau_.shape[2], tau_.shape[3])

    
    # cond: [batchsz, c, 64, 64]
    m = torch.where(cond < tau_, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
    
    #m.requires_grad = True
   
    
    
    return m, tau
    
# https://github.com/VainF/pytorch-msssim
def calculate_msssim_loss(x, y):
    ms_ssim_loss = (1 - ms_ssim(x, y, data_range=1., size_average=True))
    return ms_ssim_loss

def calculate_perceptual_loss(x, x_hat, L1_loss):
    x_2_2, x_4_3 = x
    
    x_hat_2_2, x_hat_4_3 = x_hat
    return L1_loss(x_2_2, x_hat_2_2) + L1_loss(x_4_3, x_hat_4_3)
    
    

def print_summary(model, device):
    model = model(device).to(device)
    summary(model, (3, 256, 256))
    
def calculate_PSNR(x_spt, x_hat):
    x_spt = x_spt.cpu().numpy()
    x_spt = x_spt * np.array([[[0.229]], [[0.224]], [[0.225]]])
    x_spt = x_spt + np.array([[[0.485]], [[0.456]], [[0.406]]])
    x_hat = x_hat.cpu().numpy()
    x_hat = x_hat * np.array([[[0.229]], [[0.224]], [[0.225]]])
    x_hat = x_hat + np.array([[[0.485]], [[0.456]], [[0.406]]])
    MSE = np.sum((x_spt-x_hat) ** 2) / 256 / 256 / 3
    return 10*np.log10(1/MSE), torch.Tensor(x_spt).unsqueeze(0).to(device), torch.Tensor(x_hat).unsqueeze(0).to(device)
                              