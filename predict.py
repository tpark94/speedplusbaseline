import torch

import os
import numpy as np
import json

from utils.utils import set_all_seeds
from cfgs.cfg_park2019 import cfg

# root = '/Users/taehapark/SLAB'
root   = '/media/shared/Jeff/SLAB'
device = torch.device('cuda:0')

# ----- SEED
set_all_seeds(2021, True)

# ----- Camera
with open('/home/jeffpark/SLAB/Dataset/speedplus/camera.json', 'r') as f:
    cam = json.load(f)
cameraMatrix = np.array(cam['K'], dtype=np.float32)
distCoeffs   = np.array(cam['dist_coeffs'], dtype=np.float32)

# ---------------------------------------------------------------------------- #
# # Park et al., ASC2019
# from nets.park2019.predict import predict

# # - Pretrained weights of [PARK19]
# odn_dict = 'checkpoints/park2019/odn/speedplus_synthetic_1/final.pth'
# krn_dict = 'checkpoints/park2019/krn/speedplus_synthetic_1/final.pth'

# # - CSV for data to predict
# cfg.val_csv = '/home/jeffpark/SLAB/Dataset/speedplus/sunlamp/splits_park2019/sunlamp.csv'

# # Evaluation on validation set
# krn_only = False
# err_iou, err_t, err_q, speed = predict(odn_dict, krn_dict, cfg, cameraMatrix, distCoeffs, device, krn_only)

# ---------------------------------------------------------------------------- #
# Park et al., ASC2019
from nets.park2019.predict import predict

eval = 'sunlamp'

eIoU, eT, eR, spd = [], [], [], []
for ii in range(1,6):
    # - Weights
    odn_dict = 'checkpoints/park2019/odn/speedplus_{}_{}/final.pth'.format(eval, ii)
    krn_dict = 'checkpoints/park2019/krn/speedplus_{}_{}/final.pth'.format(eval, ii)

    # - CSV for data to predict
    cfg.val_csv = '{}/splits_park2019/{}.csv'.format(eval, eval)

    # Evaluation on validation set
    krn_only = False
    err_iou, err_t, err_q, speed = predict(odn_dict, krn_dict, cfg, cameraMatrix, distCoeffs, device, krn_only)

    eIoU.append(err_iou.avg)
    eT.append(err_t.avg)
    eR.append(err_q.avg)
    spd.append(speed.avg)

print('eIoU:  %.2f +/- %.2f [m]' %   (np.mean(eIoU), np.std(eIoU)))
print('eT:    %.2f +/- %.2f [m]' %   (np.mean(eT), np.std(eT)))
print('eR:    %.2f +/- %.2f [deg]' % (np.mean(eR), np.std(eR)))
print('SPEED: %.2f +/- %.2f [-]' %   (np.mean(spd), np.std(spd)))


