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
                       resize_patches,
                       describe_with_default_ori)

if __name__ == '__main__':
    DO_CUDA=True
    model_weights = 'pretrained/HardNet++.pth'
    INPUT_DATA_DIR = 'input_data/'
    OUT_DIR = 'aux_data/plain_hardnet'
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
        
