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
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import numpy as np
import time

from src.utils.utils import AverageMeter, report_progress
from src.utils.visualize import imshow, plot_2D_bbox, scatter_keypoints

def train_dann_single_epoch_krn(epoch, cfg, model, dataloader_source, dataloader_target,
                                optimizer, writer, device, scaler=None):

    # logger = logging.getLogger("Training")
    training_time_meter = AverageMeter('ms')
    loss_pose_meter   = AverageMeter('-')
    loss_source_meter = AverageMeter('-')
    loss_target_meter = AverageMeter('-')

    # switch to train mode
    model.train()

    # Current learning rate
    for pg in optimizer.param_groups:
        lr = pg['lr']

    # from both data loaders
    batches   = zip(dataloader_source, dataloader_target)
    n_batches = min(len(dataloader_source), len(dataloader_target))

    for idx, ((source, label), target) in enumerate(batches):
        B  = source.size(0)
        ts = time.time()

        # Debug (uncomment)
        # imshow(target[0])
        # x = label[0, [0,2,4,6,8,10,12,14,16,18,20]]
        # y = label[0, [1,3,5,7,9,11,13,15,17,19,21]]
        # scatter_keypoints(source[0], x, y, True)

        # To device
        source = source.to(device)
        label  = label.to(device)
        target = target.to(device)

        # Zero gradient
        optimizer.zero_grad(set_to_none=True)

        # Domain classifier loss factor
        p = float(idx + epoch * n_batches) / cfg.max_epochs / n_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Feed-forward (source)
        loss_source, domain_source_pred = model(source, y=label, alpha=alpha)
        loss_pose_source, sm = loss_source

        # Domain loss (1: source, 0: target)
        loss_domain_source = nn.functional.binary_cross_entropy_with_logits(
                                domain_source_pred, torch.ones(B).to(device), reduction='mean'
        )
        # Feed-forward (target)
        _, domain_target_pred = model(target, alpha=alpha)
        loss_domain_target = nn.functional.binary_cross_entropy_with_logits(
                                domain_target_pred, torch.zeros(B).to(device), reduction='mean'
        )

        # Backprop
        loss = loss_pose_source + loss_domain_source + loss_domain_target
        loss.backward()

        # Compute & update gradient
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # measure elapsed time & loss
        training_time_meter.update((time.time() - ts)*1000, B)
        loss_pose_meter.update(float(loss_pose_source), B)
        loss_source_meter.update(float(loss_domain_source), B)
        loss_target_meter.update(float(loss_domain_target), B)

        # Report
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=n_batches,
                        time=training_time_meter, is_train=True,
                        loss_pose=loss_pose_meter, loss_source=loss_source_meter, loss_target=loss_target_meter)

    # log
    if writer is not None:
        writer.add_scalar('train/loss_pose',   loss_pose_meter.avg, epoch)
        writer.add_scalar('train/loss_source', loss_source_meter.avg, epoch)
        writer.add_scalar('train/loss_target', loss_target_meter.avg, epoch)
