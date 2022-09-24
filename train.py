import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import copy

from models import *
from prob import ProbabilityModel
from dataloader import *
from hyper import *
from util import *


def train_init(jpeg_ai, encoder, importance, decoder, probabilityModel, 
               writer, init_opt, epoch, device, L1_loss, MSE_loss,
               vgg16, round_straightthrough):
    
    encoder.train(), importance.train(), decoder.train(), probabilityModel.train()
    
    # fetch num_tasks image each time
    db = DataLoader(jpeg_ai, num_tasks, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    
    len_db = len(db)
    step = 0
    

    it = tqdm(db)
    # for step, x_spt in enumerate(db):
    for x_spt in it:
        it.set_description('epoch %d' %epoch)
        # x_spt: [num_tasks, 3, 256, 256]
        init_opt.zero_grad()
        
        x_spt = x_spt.to(device)
        
        # forward the whole network
        y = encoder(x_spt)
        
        m, tau = calculate_m(x_spt, importance, round_straightthrough, device)
        y_telta = torch.mul(y, m)
        # y_telta = y
        z, _s, _z = round_straightthrough(y_telta, b)
        y_hat = dequentize(z, _s, _z)
        x_hat = decoder(y_hat)
        # x_hat = decoder(y_telta)
        
        
        # losses
     
        loss_M = calculate_loss_M(m, tau, device, num_tasks)                     # impoortant map loss
        loss_d1 = calculate_msssim_loss(x_spt, x_hat)                              # MS-SSIM loss
        loss_d2 = MSE_loss(x_spt, x_hat)/num_tasks                                         # MSE loss
        loss_d3 = calculate_perceptual_loss(vgg16(x_spt), vgg16(x_hat), L1_loss)/num_tasks   # perceptual loss
        loss_r = probabilityModel(z)                                              # probability model loss
        
        
        loss =  loss_d2 + lamb_msssim * loss_d1 + loss_d3* lamb_percep + loss_M*lamb_M + loss_r
        
        writer.add_scalar('init_loss/loss_M', loss_M, len_db*epoch+step)
        writer.add_scalar('init_loss/loss_d1', loss_d1, len_db*epoch+step)
        writer.add_scalar('init_loss/loss_d2', loss_d2, len_db*epoch+step)
        writer.add_scalar('init_loss/loss_d3', loss_d3, len_db*epoch+step)
        writer.add_scalar('init_loss/loss_r', loss_r, len_db*epoch+step)
        writer.add_scalar('init_loss/loss', loss, len_db*epoch+step)
        
        loss.backward()
        init_opt.step()
        
        step += 1
        
def train_meta(jpeg_ai, encoder, importance, decoder, probabilityModel, 
               writer, meta_opt, epoch, device, L1_loss, MSE_loss,
               vgg16, round_straightthrough):
    
    encoder.train(), importance.train(), decoder.train(), probabilityModel.train()
    # fetch num_tasks image each time
    db = DataLoader(jpeg_ai, num_tasks, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    
    len_db = len(db)
    step = 0
    
    it = tqdm(db)
    
    for x_spt in it:
        it.set_description('epoch %d' %epoch)
        meta_opt.zero_grad()
        x_spt = x_spt.to(device)
        # update meta learner once
        y = encoder(x_spt)
       
        m, tau = calculate_m(x_spt, importance, round_straightthrough, device)
        y_telta = torch.mul(y, m)
        # y_telta = y
        
        # inner loop
        for i in range(inner_update):
            # forward the whole network
            y_telta.retain_grad()
            z, _s, _z = round_straightthrough(y_telta, b)
            y_hat = dequentize(z, _s, _z)
            x_hat = decoder(y_hat)
            # x_hat = decoder(y_telta)
            
            # losses
            
            loss_M = calculate_loss_M(m, tau, device, num_tasks)                     # impoortant map loss
            loss_d1 = calculate_msssim_loss(x_spt, x_hat)                              # MS-SSIM loss
            loss_d2 = MSE_loss(x_spt, x_hat)/num_tasks                                           # MSE loss
            loss_d3 = calculate_perceptual_loss(vgg16(x_spt), vgg16(x_hat), L1_loss)/num_tasks   # perceptual loss
            loss_r = probabilityModel(z)                                              # probability model loss
        
            loss = loss_d2 + lamb_msssim * loss_d1 + loss_d3* lamb_percep +loss_M*lamb_M + loss_r
            
            loss.backward(retain_graph=True)
            y_telta = y_telta - meta_inner_lr * y_telta.grad.data
            
        # outer loop
        # forward the  whole network
        
        #z, _s, _z = round_straightthrough(y_telta, b)
        #y_hat = dequentize(z, _s, _z)
        #x_hat = decoder(y_hat)
        x_hat = decoder(y_telta)
        
        # losses
        
        loss_M = calculate_loss_M(m, tau, device, num_tasks)                     # impoortant map loss
        loss_d1 = calculate_msssim_loss(x_spt, x_hat)                              # MS-SSIM loss
        loss_d2 = MSE_loss(x_spt, x_hat)/num_tasks                                           # MSE loss
        loss_d3 = calculate_perceptual_loss(vgg16(x_spt), vgg16(x_hat), L1_loss)/num_tasks   # perceptual loss
        loss_r = probabilityModel(z)                                              # probability model loss
       
        loss =  loss_d2 + lamb_msssim * loss_d1 + loss_d3* lamb_percep+loss_M*lamb_M + loss_r
        
        writer.add_scalar('meta_loss/loss_M', loss_M, len_db*epoch+step)
        writer.add_scalar('meta_loss/loss_d1', loss_d1, len_db*epoch+step)
        writer.add_scalar('meta_loss/loss_d2', loss_d2, len_db*epoch+step)
        writer.add_scalar('meta_loss/loss_d3', loss_d3, len_db*epoch+step)
        writer.add_scalar('meta_loss/loss_r', loss_r, len_db*epoch+step)
        writer.add_scalar('meta_loss/loss', loss, len_db*epoch+step)
        
        loss.backward()
        meta_opt.step()
        
        step += 1


def overfit_dec(jpeg_ai, encoder, importance, decoder, probabilityModel, 
                   writer, device, L1_loss, MSE_loss,
                   vgg16, round_straightthrough):
    
    encoder.eval(), importance.eval(), decoder.eval(), probabilityModel.eval()
    
    params_all = list(encoder.parameters()) + list(importance.parameters()) + list(decoder.parameters()) + list(probabilityModel.parameters())
   
    for p in params_all:
        p.requires_grad = False
    
    # fetch 1 image each time
    db = DataLoader(jpeg_ai, 1, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    len_db = len(db)
    
    step = 0
    
    it = tqdm(db)
    
    dec_bias = []
    
    for x_spt in it:
        
        it.set_description('overfitting decoder for img %d' %step)
        decoder_clone = copy.deepcopy(decoder)
        for p in decoder_clone.parameters():
            p.requires_grad = True
        x_spt = x_spt.to(device)
        
        
        params_dec = [# bias only
                    {"params": (decoder_clone.conv1[0].bias)},
                    #{"params": (decoder_clone.resblock1.left[0].bias)},
                    #{"params": (decoder_clone.resblock1.left[3].bias)},
                    {"params": (decoder_clone.conv2[0].bias)},
                    #{"params": (decoder_clone.resblock2.left[0].bias)},
                    #{"params": (decoder_clone.resblock2.left[3].bias)},
                    {"params": (decoder_clone.conv3.bias)}] 
      
    
        
        dec_opt = optim.Adam(params_dec, lr=overfit_dec_lr)
        
        for i in range(dec_update):
            
            dec_opt.zero_grad()
            
            # forward the whole network
            y = encoder(x_spt)
            
            m, tau = calculate_m(x_spt, importance, round_straightthrough, device)
            y_telta = torch.mul(y, m)
            # y_telta = y
            z, _s, _z = round_straightthrough(y_telta, b)
            y_hat = dequentize(z, _s, _z)
            x_hat = decoder_clone(y_hat)
            # x_hat = decoder_clone(y_telta)
            
            
            
            # losses
           
            # loss_M = calculate_loss_M(m, tau, device, 1)                             # impoortant map loss
            loss_d1 = calculate_msssim_loss(x_spt, x_hat)                            # MS-SSIM loss
            loss_d2 = MSE_loss(x_spt, x_hat)                            # MSE loss
            loss_d3 = calculate_perceptual_loss(vgg16(x_spt), vgg16(x_hat), L1_loss) # perceptual loss
            #loss_r = probabilityModel(z)                                             # probability model loss
           
            loss = lamb_msssim * loss_d1 + loss_d2 + loss_d3* lamb_percep 
            
            loss.backward()
            dec_opt.step()
            
        step += 1
        
        dec_bias.append(list(decoder_clone.conv1[0].bias.detach().cpu().numpy()) +
                        list(decoder_clone.conv2[0].bias.detach().cpu().numpy()) +
                        list(decoder_clone.conv3.bias.detach().cpu().numpy()))
        
    
    return np.array(dec_bias), len(list(decoder_clone.conv1[0].bias.detach().cpu().numpy())), \
                                len(list(decoder_clone.conv2[0].bias.detach().cpu().numpy())), \
                                len(list(decoder_clone.conv3.bias.detach().cpu().numpy()))