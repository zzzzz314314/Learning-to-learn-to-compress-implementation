#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:44:51 2021

@author: user
"""
import os

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from hyper import *



class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding = (1,1), bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding = (1,1), bias=False),
            nn.BatchNorm2d(3)
        )
    def forward(self, x):
        out = self.left(x)
        out += x
        out = F.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, Q):
        super(CNN, self).__init__()
        self.Reslayer = nn.ModuleList([Resblock() for _ in range(5)])
        self.conv1 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding = (1,1), bias=False)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding = (1,1), bias=False)
        self.convp = nn.Conv2d(3, Q, kernel_size=3, stride=1, padding = (1,1), bias=False) 
        self.convq = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding = (1,1), bias=False)
        
    def forward(self, q, z_i, m_i):
 
        x = torch.cat((z_i, m_i), 1)
        x = self.conv1(x)
        
        x = x + q
        orginal_x = x
        for resblock in self.Reslayer:
            x = resblock(x)
        x = x + orginal_x
        x = self.conv2(x)
        p = self.convp(x)
        q = self.convq(x)
        return p, q


# class ProgressiveStatisticalModel(nn.Module):
#     def __init__(self, H, W):
#         super(ProgressiveStatisticalModel, self).__init__()
#         self.CNNlist = nn.ModuleList([CNN(100) for _ in range(4)])
#         # self.upsampler = F.upsample_nearest(scale_factor = 2)
#         self.m1 = torch.zeros(( num_tasks, 1, H, W)).to(device)
#         self.m1[:, :, 1:H:2, 1:W:2] = 1
#         self.m2 = self.m1
#         self.m2[:, :, 0:H:2, 0:W:2] = 1
#         self.m3 = self.m2
#         self.m3[:, :, 0:H:2, 1:W:2] = 1
#         self.m4 = self.m3
#         self.m4[:, :, 1:H:2, 0:W:2] = 1

#     def forward(self, q, z_i, z_i_previous):
#         z_hat = F.upsample_nearest(z_i, scale_factor = 2)

#         q = F.upsample_nearest(q, scale_factor = 2)
#         p1, q1 = self.CNNlist[0](q, z_hat, self.m1)
#         loss = discretized_mix_logistic_loss(z_i_previous, p1)
#         z_hat = self.mixer(z_hat, z_i_previous, self.m1)
#         p2, q2 = self.CNNlist[1](q1, z_hat, self.m2)
#         loss += discretized_mix_logistic_loss(z_i_previous, p2)
#         z_hat = self.mixer(z_hat, z_i_previous, self.m2)
#         p3, q3 = self.CNNlist[2](q2, z_hat, self.m3)
#         loss += discretized_mix_logistic_loss(z_i_previous, p3)
#         z_hat = self.mixer(z_hat, z_i_previous, self.m3)
#         p4, q4 = self.CNNlist[3](q3, z_hat, self.m4)
#         loss += discretized_mix_logistic_loss(z_i_previous, p4)
#         #return (p1, p2, p3, p4), q4, loss
#         return q4, loss
    
class ProgressiveStatisticalModel(nn.Module):
    def __init__(self, H, W):
        super(ProgressiveStatisticalModel, self).__init__()
        self.CNNlist = nn.ModuleList([CNN(100) for _ in range(4)])
        self.H = H
        self.W = W
 

    def forward(self, q, z_i, z_i_previous):
        z_hat = F.upsample_nearest(z_i, scale_factor = 2)
        q = F.upsample_nearest(q, scale_factor = 2)
        
        m1 = torch.zeros((z_hat.shape[0], 1, self.H, self.W)).to(device)
        m1[:, :, 1:self.H:2, 1:self.W:2] = 1
        m2 = m1
        m2[:, :, 0:self.H:2, 0:self.W:2] = 1
        m3 = m2
        m3[:, :, 0:self.H:2, 1:self.W:2] = 1
        m4 = m3
        m4[:, :, 1:self.H:2, 0:self.W:2] = 1
        
        p1, q1 = self.CNNlist[0](q, z_hat, m1)
        loss = discretized_mix_logistic_loss(z_i_previous, p1)
        z_hat = self.mixer(z_hat, z_i_previous, m1)
        p2, q2 = self.CNNlist[1](q1, z_hat, m2)
        loss += discretized_mix_logistic_loss(z_i_previous, p2)
        z_hat = self.mixer(z_hat, z_i_previous, m2)
        p3, q3 = self.CNNlist[2](q2, z_hat, m3)
        loss += discretized_mix_logistic_loss(z_i_previous, p3)
        z_hat = self.mixer(z_hat, z_i_previous, m3)
        p4, q4 = self.CNNlist[3](q3, z_hat, m4)
        loss += discretized_mix_logistic_loss(z_i_previous, p4)
        #return (p1, p2, p3, p4) q4, loss
        return q4, loss
    
        
    def mixer(self, z_hat, z_i_previous, m_i):
        z_hat = z_hat * (1-m_i) + z_i_previous * m_i
        return z_hat
        


class ProbabilityModel(nn.Module):
    def __init__(self):
        super(ProbabilityModel, self).__init__()
        # self.downsampler = F.interpolate(scale_factor = 1/2, mode = 'nearest')
        self.progressive1 = ProgressiveStatisticalModel(64, 64)
        self.progressive2 = ProgressiveStatisticalModel(32, 32)
        
    def forward(self, z_0):
        z_1 = F.interpolate(z_0, scale_factor = 1/2, mode = 'nearest')
        z_2 = F.interpolate(z_1, scale_factor = 1/2, mode = 'nearest')

        # print('z_1:', z_1.shape)
        # print('z_2:', z_2.shape)
        q_3 = torch.zeros(z_2.size()).to(device)
        #p_2, q_2, loss2 = self.progressive2(q_3, z_2, z_1)
        q_2, loss2 = self.progressive2(q_3, z_2, z_1)
        #p_1, q_1, loss1 = self.progressive1(q_2, z_1, z_0)
        q_1, loss1 = self.progressive1(q_2, z_1, z_0)
        #return (p_1, p_2), loss1 + loss2
        return  loss1 + loss2



# Pixel-CNN++ pytorch implement by 
# https://github.com/pclucas14/pixel-cnn-pp

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    # here and below: unpacking the params of the mixture of logistics
    #nr_mix = 10
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    
    means = l[:, :, :, :, :nr_mix]
    
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    #m4 = (means[:, :, :, 3, :] + coeffs[:, :, :, 3, :] * x[:, :, :, 0, :] +
    #           coeffs[:, :, :, 4, :] * x[:, :, :, 1, :] + 
    #            coeffs[:, :, :, 5, :] * x[:, :, :, 2, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    #print(x.shape, means.shape)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs))        

def print_summary(model, device):
    model = model.to(device)
    summary(model, (3, 64, 64))
    
if __name__ == "__main__":
    from torchsummary import summary
    model = ProbabilityModel()
    print_summary(model, device)