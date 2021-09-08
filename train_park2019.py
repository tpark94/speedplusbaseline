import torch
import os

from utils.utils import set_all_seeds, load_surface_points

from nets.park2019.train_odn import train_odn
from nets.park2019.train_krn import train_krn
from cfgs.cfg_park2019       import cfg

if __name__ == '__main__':

    # GPU ID for this process
    device    = torch.device('cuda:0') if cfg.use_cuda else torch.device('cpu')

    # Seeds for this process
    set_all_seeds(2021, True)

    # KRN stuff
    corners3D  = load_surface_points('utils/tangoPoints.mat')
    cameraJson = os.path.join(cfg.dataroot, 'camera.json')

    # ---------------------------------------------------------------------------- #
    cfg.batch_size = 24
    cfg.lr = 0.0005

    # -- Train ODN (Synthetic)
    cfg.savedir    = 'checkpoints/park2019/odn/speedplus_synthetic_1'
    cfg.train_csv  = 'synthetic/splits_park2019/train.csv'
    cfg.val_csv    = 'synthetic/splits_park2019/test.csv'
    cfg.save_epoch = 1
    cfg.test_epoch = 5
    cfg.max_epochs = 20
    cfg.lr_decay_alpha = 0.95
    train_odn(cfg, device)

    # # -- Train ODN (Lightbox) on 5 random splits
    # cfg.save_epoch = 10
    # cfg.test_epoch = 20
    # cfg.max_epochs = 160
    # cfg.lr_decay_alpha = 0.95 ** (1/10)
    # for ii in range(1,6):
    #     cfg.savedir   = 'checkpoints/park2019/odn/speedplus_lightbox_{}'.format(ii)
    #     cfg.train_csv = 'lightbox/splits_park2019/train_{}.csv'.format(ii)
    #     cfg.val_csv   = 'lightbox/splits_park2019/test_{}.csv'.format(ii)
    #     train_odn(cfg, device)

    # # -- Train ODN (Sunlamp) on 5 random splits
    # cfg.save_epoch = 30
    # cfg.test_epoch = 60
    # cfg.max_epochs = 480
    # cfg.lr_decay_alpha = 0.95 ** (1/30)
    # for ii in range(1,6):
    #     cfg.savedir   = 'checkpoints/park2019/odn/speedplus_sunlamp_{}'.format(ii)
    #     cfg.train_csv = 'sunlamp/splits_park2019/train_{}.csv'.format(ii)
    #     cfg.val_csv   = 'sunlamp/splits_park2019/test_{}.csv'.format(ii)
    #     train_odn(cfg, device)

    # ---------------------------------------------------------------------------- #
    cfg.batch_size = 48
    cfg.lr = 0.001

    # -- Train KRN (Synthetic)
    cfg.savedir    = 'checkpoints/park2019/krn/speedplus_synthetic_2'
    cfg.train_csv  = 'synthetic/splits_park2019/train.csv'
    cfg.val_csv    = 'synthetic/splits_park2019/test.csv'
    cfg.save_epoch = 5
    cfg.test_epoch = 5
    cfg.max_epochs = 75
    cfg.lr_decay_alpha = 0.95
    train_krn(cfg, cameraJson, corners3D, device)

    # # -- Train KRN (Lightbox) on 5 random splits
    # cfg.save_epoch = 50
    # cfg.test_epoch = 50
    # cfg.max_epochs = 750
    # cfg.lr_decay_alpha = 0.95 ** (1/10)
    # for ii in range(1,6):
    #     cfg.savedir   = 'checkpoints/park2019/krn/speedplus_lightbox_{}'.format(ii)
    #     cfg.train_csv = 'lightbox/splits_park2019/train_{}.csv'.format(ii)
    #     cfg.val_csv   = 'lightbox/splits_park2019/train_{}.csv'.format(ii)
    #     train_krn(cfg, cameraJson, corners3D, device)

    # # -- Train KRN (Sunlamp) on 5 random splits
    # cfg.save_epoch = 150
    # cfg.test_epoch = 150
    # cfg.max_epochs = 2250
    # cfg.lr_decay_alpha = 0.95 ** (1/30)
    # for ii in range(1,6):
    #     cfg.savedir   = 'checkpoints/park2019/krn/speedplus_sunlamp_{}'.format(ii)
    #     cfg.train_csv = 'sunlamp/splits_park2019/train_{}.csv'.format(ii)
    #     cfg.val_csv   = 'sunlamp/splits_park2019/test_{}.csv'.format(ii)
    #     train_krn(cfg, cameraJson, corners3D, device)