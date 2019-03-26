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

def read_circle_patches(fname, rot_ang = 0):
    #patches = np.loadtxt(fname, delimiter=',') #24 sec to read
    #patches = pd.read_csv(fname,header=None,dtype=np.uint8).values #6 sec to read. Still huge :(
    patches = np.fromfile(fname, sep =' ', dtype=np.uint8).reshape(-1,137*137) # Fastest, but needs preprocessing in bash: for i in *.csv; do sed -i 's/,/ /g' "$i"; done
    num, area = patches.shape
    PS = int(np.sqrt(area))
    assert PS == 137
    patches = np.reshape(patches, (num,1, PS, PS))
    return patches

def crop_round_patches(circle_patches, cropsize=97):
    num,ch,h,w = circle_patches.shape
    assert h == 137
    assert w == 137
    PS = h
    PS_crop = cropsize
    pad = (PS - PS_crop)//2
    crop_patches = circle_patches[:,:,pad:pad+PS_crop,pad:pad+PS_crop]
    return crop_patches

def rotate_circle_patches(cp, rot_angles):
    ropatches = np.ndarray(cp.shape, dtype=np.uint8)
    for i in range(len(cp)):
        ropatches[i,0,:,:] = np.array(Image.fromarray(cp[i,0,:,:]).rotate(-rot_angles[i], resample=Image.BICUBIC))
    return ropatches

def resize_patches(rp, PS=32):
    num,ch,h,w = rp.shape
    out_patches = np.ndarray((num,ch, PS,PS), dtype=np.uint8)
    for i in range(len(rp)):
        out_patches[i,0,:,:] = np.array(Image.fromarray(rp[i,0,:,:]).resize((PS,PS), resample=Image.LANCZOS))
    return out_patches

def describe_with_default_ori(fname, model):
    model = model.cuda()
    cp = read_circle_patches(fname)
    angles = np.loadtxt(fname.replace('big_patches', 'ori'))
    cp_rot = rotate_circle_patches(cp,angles)
    rp_rot = crop_round_patches(cp_rot)
    out_patches = resize_patches(rp_rot).astype(np.float32)
    n_patches = len(out_patches)
    bs = 128
    outs = []
    n_batches = int(n_patches / bs) + 1
    descriptors_for_net = np.zeros((n_patches, 128))
    for i in range(0, n_patches, bs):
        data_a = out_patches[i: i + bs, :, :, :]
        data_a = torch.from_numpy(data_a).cuda()
        with torch.no_grad():
            out_a = model(data_a)
        descriptors_for_net[i: i + bs,:] = out_a.data.cpu().numpy().reshape(-1, 128)
    descriptors_for_net = descriptors_for_net + 0.45;
    out = (descriptors_for_net * 210.).astype(np.int32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

if __name__ == '__main__':
    DO_CUDA=True
    model_weights = 'pretrained/HardNet++.pth'
    INPUT_DATA_DIR = 'input_data/'
    OUT_DIR = 'output_data/plain_hardnet'
    model = HardNet()
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if DO_CUDA:
        model = model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
    
    fnames = sorted([f for f in os.listdir(os.path.join(INPUT_DATA_DIR))  if f.endswith('csv') ])
    patches_fnames = [os.path.join(INPUT_DATA_DIR, f) for f in fnames if 'patches' in f]
    ori_fnames = [os.path.join(INPUT_DATA_DIR, f) for f  in fnames if 'ori' in f]
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    for idx, fn in enumerate(patches_fnames):
        t=time()
        out_fname = os.path.join(OUT_DIR, fn.split('/')[-1].replace('big_patches', 'hardnet'))
        if os.path.isfile(out_fname):
            print (out_fname, 'exists, skipping')
        desc = describe_with_default_ori(fn, model)
        np.savetxt(out_fname, desc, delimiter=' ', fmt='%d')
        print (fn, 'in',  time()-t, 'sec', idx, 'out of', len(patches_fnames))
    print ('Done')
        
