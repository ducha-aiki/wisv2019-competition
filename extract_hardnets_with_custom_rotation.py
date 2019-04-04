import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
import csv
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from time import time
import os
import math
from HardNet import HardNet
from PIL import Image
import pandas as pd
from wiswUtils import (read_circle_patches,
                       crop_round_patches,
                       rotate_circle_patches,
                       resize_patches)

class OriNetFast(nn.Module):
    def __init__(self, PS = 16):
        super(OriNetFast, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 2, kernel_size=int(PS/4), stride=1,padding=1, bias = True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/4)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.9)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_rot_matrix = False):
        xy = self.features(self.input_norm(input)).view(-1,2) 
        angle = torch.atan2(xy[:,0] + 1e-8, xy[:,1]+1e-8);
        if return_rot_matrix:
            return get_rotation_matrix(angle)
        return angle
def get_batched_out(patches_np, model, out_dim = 128):
    n_patches = len(patches_np)
    bs = 128
    outs = []
    n_batches = int(n_patches / bs) + 1
    descriptors_for_net = np.zeros((n_patches, out_dim))
    for i in range(0, n_patches, bs):
        data_a = patches_np[i: i + bs, :, :, :]
        data_a = torch.from_numpy(data_a).cuda()
        with torch.no_grad():
            out_a = model(data_a)
        descriptors_for_net[i: i + bs,:] = out_a.data.cpu().numpy().reshape(-1, out_dim)
    return descriptors_for_net
    
    
def describe_with_custom_ori(fname, model, ori_model):
    model = model.cuda()
    cp = read_circle_patches(fname)
    rp = crop_round_patches(cp)
    patches_for_orinet = resize_patches(rp).astype(np.float32)
    angles = np.degrees(get_batched_out(patches_for_orinet, ori_model, 1).flatten())
    np.savetxt(fname.replace('big_patches', 'custom_angles'), angles, fmt='%4.3f')
    cp_rot = rotate_circle_patches(cp,angles)
    rp_rot = crop_round_patches(cp_rot)
    out_patches = resize_patches(rp_rot).astype(np.float32)
    descriptors_for_net = get_batched_out(out_patches, model, 128)
    descriptors_for_net = descriptors_for_net + 0.45;
    out = (descriptors_for_net * 210.).astype(np.int32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


if __name__ == '__main__':
    DO_CUDA=True
    model_weights = 'pretrained/HardNet++.pth'
    INPUT_DATA_DIR = 'input_data/'
    OUT_DIR = 'aux_data/orinet_hardnet'
    model = HardNet()
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    ori_model = OriNetFast(PS=32)
    ori_model_weights = 'pretrained/OriNet.pth'
    ori_checkpoint = torch.load(ori_model_weights)
    ori_model.load_state_dict(ori_checkpoint['state_dict'])
    ori_model.eval()
    
    if DO_CUDA:
        model = model.cuda()
        ori_model = ori_model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
        ori_model = ori_model.cpu()
    
    
    fnames = sorted([f for f in os.listdir(os.path.join(INPUT_DATA_DIR))  if f.endswith('csv') ])
    patches_fnames = [os.path.join(INPUT_DATA_DIR, f) for f in fnames if 'patches' in f]
    ori_fnames = [os.path.join(INPUT_DATA_DIR, f) for f  in fnames if 'ori' in f]
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    for idx, fn in enumerate(patches_fnames):
        t=time()
        out_fname = os.path.join(OUT_DIR, fn.split('/')[-1].replace('big_patches', 'orinethardnet'))
        if os.path.isfile(out_fname):
            print (out_fname, 'exists, skipping')
            continue
        desc = describe_with_custom_ori(fn, model, ori_model)
        np.savetxt(out_fname, desc, delimiter=' ', fmt='%d')
        print (fn, 'in',  time()-t, 'sec', idx, 'out of', len(patches_fnames))
    print ('Done')
        
