
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import sys
import copy
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

from models import *
from prob import ProbabilityModel
from dataloader import *
from hyper import *
from util import *
#from train2 import *

def finetune(jpeg_ai, encoder, importance, decoder, probabilityModel, 
               device, L1_loss, MSE_loss,
               #vgg16, round_straightthrough):
               vgg16, round_straightthrough, dec_bias, d1, d2, d3):
    
    mse_total = 0
    msssim_total = 0
    perceptual_total = 0
    bpp_total = 0
    PSNR_total = 0
   
    params_all = list(encoder.parameters()) + list(importance.parameters()) + list(decoder.parameters()) + list(probabilityModel.parameters())
   
    for p in params_all:
        p.requires_grad = True
    
    encoder.eval(), importance.eval(), decoder.eval(), probabilityModel.eval()
    # fetch num_tasks image each time
    db = DataLoader(jpeg_ai, 1, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    
    len_db = len(db)
    #step = 0
   
    #it = tqdm(db)
    
    #for x_spt in it:
    for _, x_spt in enumerate(db):
          
        x_spt = x_spt.to(device)
        # update meta learner once
        y = encoder(x_spt)
        m, tau = calculate_m(x_spt, importance, round_straightthrough, device)
        y_telta = torch.mul(y, m)
        #y_telta = y
        # overfitting_loss = []
        # inner loop
        for i in range(inner_update):
            # forward the whole network
            y_telta.retain_grad()
            z, _s, _z = round_straightthrough(y_telta, b)
            y_hat = dequentize(z, _s, _z)
            x_hat = decoder(y_hat)
            # x_hat = decoder(y_telta)
            
            # losses
            # loss_M = calculate_loss_M(m, tau, device, 1)                     # impoortant map loss
            loss_d1 = calculate_msssim_loss(x_spt, x_hat)                            # MS-SSIM loss
            loss_d2 = MSE_loss(x_spt, x_hat)                                        # MSE loss
            loss_d3 = calculate_perceptual_loss(vgg16(x_spt), vgg16(x_hat), L1_loss)# perceptual loss
            # loss_r = probabilityModel(z)                                             # probability model loss
            #loss =  loss_M + loss_d1 + loss_d2 + loss_d3 + loss_r * lamb_r
            #loss = loss_d1 + loss_d2 + loss_d3 + loss_r * lamb_r
            loss = loss_d2 + loss_d1*lamb_msssim + loss_d3 * lamb_percep
            #overfitting_loss.append(loss)
            loss.backward(retain_graph=True)
            
            y_telta = y_telta - meta_inner_lr * y_telta.grad.data
            
     
        # plt.figure()
        # plt.plot(overfitting_loss, 'o-')
        # plt.show()
        # select_bias
        min_loss, rec_loss_d1, rec_loss_d2, rec_loss_d3, min_r = sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize
       
        for c in range(dec_bias.shape[0]):
            
            with torch.no_grad():
                # apply specific bias to decoder
                bias_c_decoder = copy.deepcopy(decoder)
                bias_c_decoder.conv1[0].bias = torch.nn.Parameter(torch.tensor(dec_bias[c, : d1]).to(device, dtype=torch.float))
                bias_c_decoder.conv2[0].bias = torch.nn.Parameter(torch.tensor(dec_bias[c, d1 : d1+ d2]).to(device, dtype=torch.float))
                bias_c_decoder.conv3.bias = torch.nn.Parameter(torch.tensor(dec_bias[c, d1 + d2 :]).to(device, dtype=torch.float))
                # forward the remaining network
                z, _s, _z = round_straightthrough(y_telta, b)
                y_hat = dequentize(z, _s, _z)
                
                x_hat = bias_c_decoder(y_hat)
                # x_hat = bias_c_decoder(y_telta)
                # losses
                loss_d1 = calculate_msssim_loss(x_spt, x_hat)                            # MS-SSIM loss
                loss_d2 = MSE_loss(x_spt, x_hat)                                    # MSE loss
                loss_d3 = calculate_perceptual_loss(vgg16(x_spt), vgg16(x_hat), L1_loss) # perceptual loss
                loss_r = probabilityModel(z)
                if lamb_msssim * loss_d1 + loss_d2 + loss_d3 * lamb_percep < min_loss:
                    min_loss = lamb_msssim * loss_d1 + loss_d2 + loss_d3 * lamb_percep
                    rec_loss_d1, rec_loss_d2, rec_loss_d3, min_r = loss_d1, loss_d2, loss_d3, loss_r/(np.log(2)*256*256)
                    bias_cluster = c
              
      
        # inference
        with torch.no_grad():
            # apply specific bias to decoder
            bias_c_decoder = copy.deepcopy(decoder)
            bias_c_decoder.conv1[0].bias = torch.nn.Parameter(torch.tensor(dec_bias[bias_cluster, : d1]).to(device, dtype=torch.float))
            bias_c_decoder.conv2[0].bias = torch.nn.Parameter(torch.tensor(dec_bias[bias_cluster, d1 : d1+ d2]).to(device, dtype=torch.float))
            bias_c_decoder.conv3.bias = torch.nn.Parameter(torch.tensor(dec_bias[bias_cluster, d1 + d2 :]).to(device, dtype=torch.float))
            # forward the remaining network
            z, _s, _z = round_straightthrough(y_telta, b)
            y_hat = dequentize(z, _s, _z)
            x_hat = bias_c_decoder(y_hat)
            
            PSNR, x_spt_RGB, x_hat_RGB = calculate_PSNR(x_spt[0], x_hat[0])
                    
            img = make_grid(torch.cat([x_spt, x_hat, x_spt_RGB, x_hat_RGB], 0).cpu(), nrow=2)
            plt.figure(1)
            plt.imshow(img.permute(1, 2, 0))
            #plt.figure(2)
            #plt.imshow(z[0].cpu().permute(1, 2, 0))
            plt.pause(0.5)
            
            print('MS-SSIM loss: {:.3f} | MSE: {:.3f} | perceptual loss: {:.3f} | bpp: {:.3f} |  PSNR: {:.3f}'
                  .format(rec_loss_d1, rec_loss_d2, rec_loss_d3,  min_r, PSNR))
            
        mse_total += rec_loss_d2
        msssim_total += rec_loss_d1
        perceptual_total += rec_loss_d3
        bpp_total += min_r
        PSNR_total += PSNR
        
    return msssim_total, mse_total, perceptual_total, bpp_total, PSNR_total
      


if __name__ == '__main__':
    
   
    # networks    
    encoder = Encoder().to(device)
    importance = Importance().to(device)
    decoder = Decoder().to(device)
    probabilityModel = ProbabilityModel().to(device)
    vgg16 = VGG16().to(device)
    
    # load weights
    arch = torch.load('/mnt/HDD3/weights_all/b8')
    encoder.load_state_dict(arch['encoder'])
    importance.load_state_dict(arch['importance'])
    decoder.load_state_dict(arch['decoder'])
    probabilityModel.load_state_dict(arch['probabilityModel'])
    
    # kmeans on decoder bias
    dec_bias = []
    dec_bias.append(list(decoder.conv1[0].bias.detach().cpu().numpy()) +
                        list(decoder.conv2[0].bias.detach().cpu().numpy()) +
                        list(decoder.conv3.bias.detach().cpu().numpy()))
    
    dec_bias_raw, d1, d2, d3 = arch['dec_bias_raw'], arch['d1'], arch['d2'], arch['d3']
    # kmeans = KMeans(n_clusters=256, random_state=0).fit(dec_bias_raw)
    # dec_bias_center = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_))
    
    kmeans = KMeans(n_clusters=255, random_state=0).fit(dec_bias_raw)
    dec_bias.extend(kmeans.cluster_centers_.tolist())
    dec_bias = np.array(dec_bias)
    
    # quantizer
    round_straightthrough = RoundStraightThrough().apply

    # losses
    L1_loss = nn.L1Loss(reduction='sum')
    MSE_loss = nn.MSELoss(reduction='sum')
    
    jpeg_ai = JPEG_AI('jpeg_ai', mode='test', n_way=1, k_shot=1, k_query=1, batchsz=50, resize=256)
    
    msssim_total, mse_total, perceptual_total, bpp_total, PSNR_total = finetune(jpeg_ai, encoder, importance, decoder, probabilityModel, 
                                                           device, L1_loss, MSE_loss,
                                                           vgg16, round_straightthrough, dec_bias, d1, d2, d3)
    print('='*80)
    print('msssim: {:.3f} | mse: {:.3f} | perceptual: {:.3f} | bpp: {:.3f} | PSNR: {:.3f}'.format(
          msssim_total.tolist()/50, mse_total.tolist()/50, perceptual_total.tolist()/50, bpp_total.tolist()/50, PSNR_total/50))
    