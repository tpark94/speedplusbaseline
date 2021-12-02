'''
MIT License

Copyright (c) 2021 SLAB Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.tensorboard import SummaryWriter

import os
import os.path as osp
import json
import logging

from config import cfg
from src.nets.build     import get_model, get_optimizer
from src.datasets.build import make_dataloader
from src.core.dann      import train_dann_single_epoch_krn
from src.core.inference import valid_krn
from src.utils.utils    import setup_logger, set_all_seeds, \
                               save_checkpoint, load_checkpoint, \
                               load_tango_3d_keypoints, load_camera_intrinsics

logger = logging.getLogger(__name__)

def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() and cfg.use_cuda else torch.device('cpu')

    # Make sure dann option is on
    # DANN only implemented for KRN
    assert cfg.dann and cfg.model_name=='krn'

    # Seeds
    set_all_seeds(2021, cfg, True)

    # Logger
    setup_logger('train')

    # Save directory
    if not osp.exists(cfg.savedir): os.makedirs(cfg.savedir)
    logger.info('Checkpoints will be saved to {}'.format(cfg.savedir))

    # Tensorboard log directory
    if not osp.exists(cfg.logdir): os.makedirs(cfg.logdir)
    writer = SummaryWriter(cfg.logdir)
    logger.info('Tensorboard logs will be saved to {}'.format(cfg.logdir))

    # Save current config
    with open(osp.join(cfg.savedir, 'config.txt'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Pose estimation CNN
    model = get_model(cfg)

    # Optimizer
    optimizer = get_optimizer(cfg, model)

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha
    )

    # Load checkpoint
    checkpoint_file = osp.join(cfg.savedir, 'checkpoint.pth.tar')
    if cfg.auto_resume and osp.exists(checkpoint_file):
        last_epoch, best_speed = load_checkpoint(checkpoint_file, model, optimizer, device)
        begin_epoch = last_epoch
        best_perf   = begin_epoch
    else:
        begin_epoch = 0
        last_epoch = -1
        # best_speed = 1e10
        best_perf   = begin_epoch

    # Model to device
    model.to(device)

    # No mixed-precision training for DANN
    # (Requires some modification to GradientReversalLayer)
    scaler = None

    # Dataloader
    source_train_loader = make_dataloader(cfg, is_train=True,  is_source=True,  load_labels=True)
    target_train_loader = make_dataloader(cfg, is_train=True,  is_source=False, load_labels=False)
    target_test_loader  = make_dataloader(cfg, is_train=False, is_source=False, load_labels=True)

    # PnP-related items
    corners3D = load_tango_3d_keypoints(cfg.keypts_3d_model)
    cameraMatrix, distCoeffs = load_camera_intrinsics(
                osp.join(cfg.dataroot, cfg.dataname, 'camera.json'))

    # Main loop
    for epoch in range(begin_epoch, cfg.max_epochs):
        # Train an epoch
        train_dann_single_epoch_krn(
                epoch, cfg, model, source_train_loader, target_train_loader,
                optimizer, writer, device)

        # Update LR scheduler
        lr_scheduler.step()

        # Test
        if (epoch+1) % cfg.test_epoch == 0 and cfg.test_epoch > 0:
            valid_krn(epoch, cfg, model, target_test_loader,
                      cameraMatrix, distCoeffs, corners3D, writer, device, None)

        # Save best models every epoch
        perf = epoch+1
        if perf > best_perf:
            best_perf = perf
            is_best = True
        else:
            is_best = False

        # Save
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.model_name,
            'state_dict': model.state_dict(),
            'best_score': best_perf,
            'optimizer': optimizer.state_dict(),
        }, is_best, cfg.savedir)

    # Close tensorboard
    writer.close()

if __name__=='__main__':
    main()