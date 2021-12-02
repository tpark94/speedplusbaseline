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

import logging
import time
import random

from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.cuda.amp import autocast

from src.nets.spn        import softmax_cross_entropy_with_logits
from src.utils.utils     import AverageMeter, report_progress
from src.utils.visualize import imshow, plot_2D_bbox, scatter_keypoints

logger = logging.getLogger("Training")

def train_single_epoch_krn(epoch, cfg, model, data_loader, optimizer,
                       writer, device, styleAugmentor=None, scaler=None):
    training_time_meter = AverageMeter('ms')
    loss_x_meter = AverageMeter('-')
    loss_y_meter = AverageMeter('-')

    # switch to train mode
    model.train()

    # Current learning rate
    for pg in optimizer.param_groups:
        lr = pg['lr']

    # Loop through dataloader
    for idx, (images, target) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])
        # scatter_keypoints(images[0], target[0,0], target[0,1], True)

        # To device
        images = images.to(device)
        target = target.to(device)

        # Randomize texture?
        if styleAugmentor is not None and random.random() < cfg.texture_ratio:
            images = styleAugmentor(images)
            # imshow(images[0].cpu())

        # compute output
        if scaler is not None and cfg.use_cuda:
            # Use mixed-precision
            with autocast():
                loss, summary = model(images, target)
        else:
            loss, summary = model(images, target)

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

        # measure elapsed time & record loss
        training_time_meter.update((time.time() - start)*1000, B)
        loss_x_meter.update(summary['loss_x'], B)
        loss_y_meter.update(summary['loss_y'], B)

        # report training progress
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=training_time_meter, is_train=True, loss_x=loss_x_meter, loss_y=loss_y_meter)

    # log to tensorboard
    if writer is not None:
        writer.add_scalar('train/loss_x', loss_x_meter.avg, epoch)
        writer.add_scalar('train/loss_y', loss_y_meter.avg, epoch)

def train_single_epoch_spn(epoch, cfg, model, data_loader, optimizer,
                       writer, device, styleAugmentor=None, scaler=None):
    training_time_meter = AverageMeter('ms')
    loss_class_meter  = AverageMeter('-')
    loss_weight_meter = AverageMeter('-')

    # switch to train mode
    model.train()

    # Current learning rate
    for pg in optimizer.param_groups:
        lr = pg['lr']

    # Loop through dataloader
    for idx, (images, yClasses, yWeights) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])

        # To device
        images   = images.to(device)
        yClasses = yClasses.to(device)
        yWeights = yWeights.to(device)

        # Randomize texture?
        if styleAugmentor is not None and random.random() < cfg.texture_ratio:
            images = styleAugmentor(images)
            # imshow(images[0].cpu())

        # compute output
        if scaler is not None and cfg.use_cuda:
            # Use mixed-precision
            with autocast():
                classes, weights = model(images)

                # Attitude classification / Relative attitude loss
                loss_class   = softmax_cross_entropy_with_logits(classes, yClasses, reduction='mean')
                loss_regress = softmax_cross_entropy_with_logits(weights, yWeights, reduction='mean')

                # Fina loss
                loss = loss_class + 10.0 * loss_regress
        else:
            classes, weights = model(images)

            # Attitude classification / Relative attitude loss
            loss_class   = softmax_cross_entropy_with_logits(classes, yClasses, reduction='mean')
            loss_regress = softmax_cross_entropy_with_logits(weights, yWeights, reduction='mean')

            # Fina loss
            loss = loss_class + 10.0 * loss_regress

        # Zero gradient
        optimizer.zero_grad(set_to_none=True)

        # Compute & update gradient
        if scaler is not None:
            # Use mixed-precision
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            clip_grad_value_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()
        else:
            loss.backward()
            clip_grad_value_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()

        # measure elapsed time & loss
        training_time_meter.update((time.time() - start)*1000, B)
        loss_class_meter.update(float(loss_class.detach().cpu()), B)
        loss_weight_meter.update(float(loss_regress.detach().cpu()), B)

        # report training progress
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=training_time_meter, is_train=True, loss_c=loss_class_meter, loss_r=loss_weight_meter)

    # log to tensorboard
    if writer is not None:
        writer.add_scalar('train/loss_c', loss_class_meter.avg, epoch)
        writer.add_scalar('train/loss_r', loss_weight_meter.avg, epoch)
