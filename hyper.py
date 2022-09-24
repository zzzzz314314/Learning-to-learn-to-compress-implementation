init_epoch = 200
meta_epoch = 5

# number of innter loop update
inner_update = 4
# number of decoder update in overfitting decoder phase
dec_update = 10

init_lr = 0.0001
meta_inner_lr = 0.001
meta_outer_lr = 0.0001
overfit_dec_lr = 0.001

# number of tasks used in one meta-update
num_tasks = 6
# total of tasks in one epoch
batchsz = 340

# number of latent tensor's channel
c = 3
# number of quantization bits
b = 8

# for supressing probability model loss
lamb_r = 0.00001
# ref: SROBB: 1/1000
# ref: enhancenet: 1/50
lamb_percep = 1/1000
lamb_msssim = 100000
lamb_M = 1000000

# used in loss_M
nonzero_ratio = 0.8

import torch
device = torch.device('cuda')