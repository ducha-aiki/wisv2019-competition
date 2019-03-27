import numpy as np
import torch
import os 
from time import time

from wiswUtils import (greedy_iterative_nns_slow,
                       match_fname,
                       distance_matrix_vector)

pairs_list =[    'DD1',      'DD2',
    'DD1',      'DD3',
    'DD1',      'DD4',
    'DD1',      'DD5',
    'DD1',      'DD6',
    'bark1',    'bark2',
    'bark1',    'bark3',
    'bark1',    'bark4',
    'bark1',    'bark5',
    'bark1',    'bark6',
    'boat1',    'boat2',
    'boat1',    'boat3',
    'boat1',    'boat4',
    'boat1',    'boat5',
    'boat1',    'boat6',
    'chatnoir1','chatnoir2',
    'chatnoir1','chatnoir3',
    'chatnoir1','chatnoir4',
    'chatnoir1','chatnoir5',
    'chatnoir1','chatnoir6',
    'graf1',    'graf2',
    'graf1',    'graf3',
    'graf1',    'graf4',
    'graf1',    'graf5',
    'graf1',    'graf6',
    'duckhunt1','duckhunt2',
    'duckhunt1','duckhunt3',
    'duckhunt1','duckhunt4',
    'duckhunt1','duckhunt5',
    'duckhunt1','duckhunt6',
    'floor1',   'floor2',
    'floor1',   'floor3',
    'floor1',   'floor4',
    'floor1',   'floor5',
    'floor1',   'floor6',
    'mario1',   'mario2',
    'mario1',   'mario3',
    'mario1',   'mario4',
    'mario1',   'mario5',
    'mario1',   'mario6',
    'marilyn1', 'marilyn2',
    'marilyn1', 'marilyn3',
    'marilyn1', 'marilyn4',
    'marilyn1', 'marilyn5',
    'marilyn1', 'marilyn6',
    'op1',      'op2',
    'op1',      'op3',
    'op1',      'op4',
    'op1',      'op5',
    'op1',      'op6',
    'outside1', 'outside2',
    'outside1', 'outside3',
    'outside1', 'outside4',
    'outside1', 'outside5',
    'outside1', 'outside6',
    'posters1', 'posters2',
    'posters1', 'posters3',
    'posters1', 'posters4',
    'posters1', 'posters5',
    'posters1', 'posters6',
    'screen1',  'screen2',
    'screen1',  'screen3',
    'screen1',  'screen4',
    'screen1',  'screen5',
    'screen1',  'screen6',
    'wall1',    'wall2',
    'wall1',    'wall3',
    'wall1',    'wall4',
    'wall1',    'wall5',
    'wall1',    'wall6',
    'spidey1',  'spidey2',
    'spidey1',  'spidey3',
    'spidey1',  'spidey4',
    'spidey1',  'spidey5',
    'spidey1',  'spidey6',
    'dc0',      'dc1',    
    'dc0',      'dc2',    
    'dc1',      'dc2',               
    'castle0',  'castle1',    
    'castle0',  'castle2',    
    'castle1',  'castle2',   
    'kermit0',  'kermit1',    
    'kermit0',  'kermit2',    
    'kermit1',  'kermit2',               
    'sponge0',  'sponge1',    
    'sponge0',  'sponge2',    
    'sponge1',  'sponge2',           
    'shelf0',   'shelf1',    
    'shelf0',   'shelf2',    
    'shelf1',   'shelf2',           
    'tribal0',  'tribal1',    
    'tribal0',  'tribal2',    
    'tribal1',  'tribal2',           
    'teddy0',   'teddy1',    
    'teddy0',   'teddy2',    
    'teddy1',   'teddy2',           
    'pen0',     'pen1',    
    'pen0',     'pen2',    
    'pen1',     'pen2',           
    'et0',      'et1',    
    'et0',      'et2',    
    'et1',      'et2',           
    'desk0',    'desk1',    
    'desk0',    'desk2',    
    'desk1',    'desk2',           
    'corridor0','corridor1',    
    'corridor0','corridor2',    
    'corridor1','corridor2',           
    'dtua0',    'dtua1',    
    'dtua0',    'dtua2',    
    'dtua1',    'dtua2',           
    'dtub0',    'dtub1',    
    'dtub0',    'dtub2',    
    'dtub1',    'dtub2',           
    'dtuc0',    'dtuc1',    
    'dtuc0',    'dtuc2',    
    'dtuc1',    'dtuc2',           
    'dtud0',    'dtud1',    
    'dtud0',    'dtud2',    
    'dtud1',    'dtud2',           
    'dtuf0',    'dtuf1',    
    'dtuf0',    'dtuf2',    
    'dtuf1',    'dtuf2',           
    'dtug0',    'dtug1',    
    'dtug0',    'dtug2',    
    'dtug1',    'dtug2',           
    'fountain0','fountain1',    
    'fountain0','fountain2',    
    'fountain1','fountain2',               
    'herzjesu0','herzjesu1',    
    'herzjesu0','herzjesu2',    
    'herzjesu1','herzjesu2',           
    'build0',   'build1',    
    'cart0',    'cart1',    
    'church0',  'church1',    
    'dante0',   'dante1',    
    'facade0',  'facade1',    
    'frame0',   'frame1',    
    'groupsac0','groupsac1',    
    'horse0',   'horse1',    
    'plant0',   'plant1',    
    'rooster0', 'rooster1',    
    'scale0',   'scale1',    
    'webcam0',  'webcam1',    
    'standing0','standing1',    
    'statue0',  'statue1',    
    'valbonne0','valbonne1',    
    'valencia0','valencia1']

if __name__ == '__main__':
    pairs = []
    for i in range(len(pairs_list)//2):
        pairs.append([pairs_list[2*i],pairs_list[2*i+1]])
    DESCR_DIR='aux_data/orinet_hardnet'
    MATCHES_DIR='output_data/'
    DESC_NAME = 'orinethardnet'
    for pair in pairs:
        fn1, fn2, mfn = match_fname(pair, DESCR_DIR, DESC_NAME, MATCHES_DIR)
        if not os.path.isdir(MATCHES_DIR):
            os.makedirs(MATCHES_DIR)
        needs_matching = not os.path.isfile(mfn)
        if True:#needs_matching:
            t = time()
            d1 = torch.from_numpy(np.nan_to_num(np.loadtxt(fn1).astype(np.float32))).cuda()
            d2 = torch.from_numpy(np.nan_to_num(np.loadtxt(fn2).astype(np.float32))).cuda()
            with torch.no_grad():
                dmv = distance_matrix_vector(d1, d2)
                out = greedy_iterative_nns_slow(dmv)
                vals, idxs = torch.sort(out[:,2])
                el = time() - t
                print ( pair, el, " sec")
                np.savetxt(mfn, out[idxs].detach().cpu().numpy(), delimiter=' ', fmt='%5.3f')
        else:
            print (pair, 'exist, skipping')
    print ('Matching done!')


