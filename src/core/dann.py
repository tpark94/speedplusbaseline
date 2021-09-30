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
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

import numpy as np
import time

from src.utils.utils import AverageMeter, report_progress
from src.utils.visualize import imshow, plot_2D_bbox, scatter_keypoints

def train_dann_single_epoch_krn(epoch, cfg, model,dataloader_source, dataloader_target,
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

    # Domain loss
    domain_loss = nn.NLLLoss().to(device)

    for idx, ((source, label), (target, _)) in enumerate(batches):
        B  = source.size(0)
        ts = time.time()

        # Debug (uncomment)
        # imshow(target[0])
        # scatter_keypoints(source[0], label[0,:,0], label[0,:,1], True)

        # To device
        source = source.to(device)
        label  = label.to(device)
        target = target.to(device)

        # Domain classifier loss factor
        p = float(idx + epoch * n_batches) / cfg.max_epochs / n_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Feed-forward (source)
        if scaler is not None and cfg.use_cuda:
            with autocast():
                loss_source, domain_source_pred = model(source, y=label, alpha=alpha)
                loss_pose_source, sm = loss_source

                # Domain loss (1: source, 0: target)
                loss_domain_source = domain_loss(domain_source_pred, torch.zeros(B).long().to(device))

                # Feed-forward (target)
                _, domain_target_pred = model(target, alpha=alpha)
                loss_domain_target = domain_loss(domain_target_pred, torch.ones(B).long().to(device))

                # Backprop
                loss = loss_pose_source + loss_domain_source + loss_domain_target
        else:
            loss_source, domain_source_pred = model(source, y=label, alpha=alpha)
            loss_pose_source, sm = loss_source

            # Domain loss (1: source, 0: target)
            loss_domain_source = domain_loss(domain_source_pred, torch.zeros(B).long().to(device))

            # Feed-forward (target)
            _, domain_target_pred = model(target, alpha=alpha)
            loss_domain_target = domain_loss(domain_target_pred, torch.ones(B).long().to(device))

            # Backprop
            loss = loss_pose_source + loss_domain_source + loss_domain_target

        # Zero gradient
        optimizer.zero_grad(set_to_none=True)

        # Compute & update gradient
        if scaler is not None:
            # Use mixed-precision
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()
        else:
            loss.backward()
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
