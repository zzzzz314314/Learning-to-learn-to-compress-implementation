import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans


from models import *
from prob import ProbabilityModel
from dataloader import *
from hyper import *
from util import *
from train import *


def main():
    
    import torch
    device = torch.device('cuda')
    
    writer = SummaryWriter('runs_final_b2')
    
    # keep the same result for every training
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    
    jpeg_ai = JPEG_AI('jpeg_ai', mode='train', n_way=1, k_shot=1, k_query=1, batchsz=batchsz, resize=256)
    
    # networks    
    encoder = Encoder().to(device)
    importance = Importance().to(device)
    decoder = Decoder().to(device)
    probabilityModel = ProbabilityModel().to(device)
    vgg16 = VGG16().to(device)
    
    
    # quantizer
    round_straightthrough = RoundStraightThrough().apply
    
    #print_summary(L2C, device)
    
    # optimizer
    params_all = list(encoder.parameters()) + list(importance.parameters()) + list(decoder.parameters()) + list(probabilityModel.parameters())
    init_opt = optim.Adam(params_all, lr=init_lr)
    meta_opt = optim.Adam(params_all, lr=meta_outer_lr)
    
   
    
    # losses
    L1_loss = nn.L1Loss(reduction='sum')
    MSE_loss = nn.MSELoss(reduction='sum')
    
    ''' start training '''

    # step1. normal training
    for epoch in range(init_epoch):
        train_init(jpeg_ai, encoder, importance, decoder, probabilityModel, 
                    writer, init_opt, epoch, device, L1_loss, MSE_loss, 
                    vgg16, round_straightthrough)
        
    torch.save({'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'importance': importance.state_dict(),
                'probabilityModel': probabilityModel.state_dict()},
                '/mnt/HDD3/weights_final/b2')
        
    # step2. meta training
    for epoch in range(meta_epoch):
        train_meta(jpeg_ai, encoder, importance, decoder, probabilityModel, 
                    writer, meta_opt, epoch, device, L1_loss, MSE_loss,
                    vgg16, round_straightthrough)
        
    torch.save({'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'importance': importance.state_dict(),
                'probabilityModel': probabilityModel.state_dict()},
                '/mnt/HDD3/weights_all/b2')

    
    # step3. overfit a set of decoder params
    dec_bias_raw, d1, d2, d3 = overfit_dec(jpeg_ai, encoder, importance, decoder, probabilityModel, 
                               writer, device, L1_loss, MSE_loss,
                               vgg16, round_straightthrough)
    
    torch.save({'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'importance': importance.state_dict(),
                'probabilityModel': probabilityModel.state_dict(),
                'dec_bias_raw': dec_bias_raw,
                'd1': d1,
                'd2': d2,
                'd3': d3},
                '/mnt/HDD3/weights_all/b2')
    
    #kmeans = KMeans(n_clusters=256, random_state=0).fit(dec_bias_raw)
    
if __name__ == '__main__':
    
    main()