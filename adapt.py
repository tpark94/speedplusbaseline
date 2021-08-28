import torch

import os

from utils.utils import set_all_seeds, load_surface_points

from da.dann import train_dann_odn, train_dann_krn
from cfgs.cfg_dann_park2019 import cfg

if __name__ == '__main__':

    # GPU ID for this process
    device    = torch.device('cuda:0') if cfg.use_cuda else torch.device('cpu')

    # Seeds for this process
    set_all_seeds(2021, True)

    # KRN stuff
    corners3D  = load_surface_points('utils/tangoPoints.mat')
    cameraJson = os.path.join(cfg.dataroot, 'camera.json')

    # ---------------------------------------------------------------------------- #
    # DA to Lightbox (ODN)
    cfg.lr = 0.0005
    cfg.save_epoch = 20
    cfg.test_epoch = 20
    cfg.max_epochs = 200
    cfg.lr_decay_alpha = 0.95 ** (1/10)
    cfg.savedir   = 'checkpoints/park2019/dann_odn/speedplus_lightbox_1'
    cfg.target_train_csv = 'lightbox/splits_park2019/lightbox.csv'
    cfg.target_val_csv   = 'lightbox/splits_park2019/lightbox.csv'
    train_dann_odn(cfg, device)

    # ---------------------------------------------------------------------------- #
    # DA to Lightbox (KRN)
    # cfg.lr = 0.001
    # cfg.save_epoch = 50
    # cfg.test_epoch = 50
    # cfg.max_epochs = 750
    # cfg.lr_decay_alpha = 0.95 ** (1/10)
    # cfg.savedir   = 'checkpoints/park2019/dann_krn/speedplus_lightbox_1'
    # cfg.target_train_csv = 'lightbox/splits_park2019/lightbox.csv'
    # cfg.target_val_csv   = 'lightbox/splits_park2019/lightbox.csv'
    # train_dann_krn(cfg, cameraJson, corners3D, device)
