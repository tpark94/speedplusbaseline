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
import os.path as osp

import torch

from src.utils.utils     import pnp, weighted_mean_quaternion, \
                                AverageMeter, report_progress
from src.utils.metrics   import *
from src.utils.visualize import imshow, plot_2D_bbox, scatter_keypoints

logger = logging.getLogger("Testing")

def valid_krn(epoch, cfg, model, data_loader, cameraMatrix, distCoeffs, corners3D, writer, device, qClass=None):
    test_time_meter = AverageMeter('ms')
    err_q_meter     = AverageMeter('deg')
    err_t_meter     = AverageMeter('m')
    speed_meter     = AverageMeter('-')
    speed_meter_th  = AverageMeter('-')
    acc_meter       = AverageMeter('%')

    err_q_all = []
    err_t_all = []
    speed_all = []

    # switch to eval mode
    model.eval()

    # Loop through dataloader
    for idx, (images, bbox, q_gt, t_gt) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])
        # print(bbox[0])

        # To device
        images = images.to(device)
        with torch.no_grad():
            x_pr, y_pr = model(images)

        # Debug (uncomment)
        # scatter_keypoints(images[0].cpu(), x_pr.cpu(), y_pr.cpu(), normalized=True)

        for b in range(B):
            # Post-processing
            q_pr, t_pr = keypts_to_pose(x_pr[b], y_pr[b], bbox[b].numpy(), corners3D, cameraMatrix, distCoeffs)

            # q_gt to [qvec, qcos]
            q_gt_i = q_gt[b, [1,2,3,0]].numpy()
            t_gt_i = t_gt[b].numpy()

            # Metrics
            err_q = error_orientation(q_pr, q_gt_i) # [deg]
            err_t = error_translation(t_pr, t_gt_i)
            speed_raw, acc = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=False)
            speed_mod, _   = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=True,
                    rotThresh=0.169, posThresh=0.002173)

        # Update
        test_time_meter.update((time.time()-start)*1000, B)
        err_q_meter.update(err_q, B)
        err_t_meter.update(err_t, B)
        speed_meter.update(speed_raw, B)
        speed_meter_th.update(speed_mod, B)
        acc_meter.update(acc*100, B)

        # Save
        err_q_all.append(err_q)
        err_t_all.append(err_t)
        speed_all.append(speed_mod)

        report_progress(epoch=epoch, lr=np.nan, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=test_time_meter, is_train=False, eT=err_t_meter, eR=err_q_meter,
                        speed=speed_meter, acc=acc_meter)

    # log to Tensorboard
    if writer is not None:
        writer.add_scalar('Valid/err_q [deg]', err_q_meter.avg, epoch)
        writer.add_scalar('Valid/err_t [m]',   err_t_meter.avg, epoch)
        writer.add_scalar('Valid/speed (raw) [-]', speed_meter.avg, epoch)
        writer.add_scalar('Valid/speed (thr) [-]', speed_meter_th.avg, epoch)

    # Write overall results
    writefn = osp.join(cfg.logdir, cfg.resultfn)
    with open(writefn, 'w') as f:
        f.write('eR:          {:.5f} [deg]\n'.format(err_q_meter.avg))
        f.write('eT:          {:.5f} [m]\n'.format(err_t_meter.avg))
        f.write('speed (raw): {:.5f} [-]\n'.format(speed_meter.avg))
        f.write('speed (thr): {:.5f} [-]\n'.format(speed_meter_th.avg))
    logger.info('Test results written to {}'.format(writefn))

    # Write performances for each items
    with open(osp.join(cfg.logdir, 'err_q.txt'), 'w') as f:
        for eq in err_q_all:
            f.write('{:.5f}\n'.format(eq))

    with open(osp.join(cfg.logdir, 'err_t.txt'), 'w') as f:
        for et in err_t_all:
            f.write('{:.5f}\n'.format(et))

    with open(osp.join(cfg.logdir, 'speed.txt'), 'w') as f:
        for spd in speed_all:
            f.write('{:.5f}\n'.format(spd))

    return err_q_meter.avg, err_t_meter.avg, speed_meter.avg, speed_meter_th.avg

def valid_spn(epoch, cfg, model, data_loader, cameraMatrix, distCoeffs, corners3D, writer, device, qClass):
    test_time_meter = AverageMeter('ms')
    err_q_meter     = AverageMeter('deg')
    err_t_meter     = AverageMeter('m')
    speed_meter     = AverageMeter('-')
    speed_meter_th  = AverageMeter('-')
    acc_meter       = AverageMeter('%')

    # switch to eval mode
    model.eval()

    # ! Temporary
    q_all = []

    # Loop through dataloader
    for idx, (images, bbox, q_gt, t_gt) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])
        # print(bbox[0])

        # Feed-forward
        with torch.no_grad():
            _, weights = model(images.to(device))

            # Post-processing (Orientation only)
            topWeights, topClasses = torch.topk(weights, cfg.num_neighbors, dim=1)
            topWeights = torch.softmax(topWeights, dim=1)

        for b in range(B):
            # Predicted quaternion classes
            qs_pr = qClass[topClasses[b].cpu()] # [N x 4]
            qs_pr = qs_pr[:,[1,2,3,0]] # [qvec, qcos]

            # Weighted mean
            q_pr = weighted_mean_quaternion(qs_pr, topWeights.cpu().squeeze())

            # q_gt to [qvec, qcos]
            q_gt_i = q_gt[b, [1,2,3,0]].numpy()
            t_gt_i = t_gt[b].numpy()

            # Metrics
            err_q = error_orientation(q_pr, q_gt_i) # [deg]
            #err_t = error_translation(t_pr, t_gt_i)
            #speed_raw, acc = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=False)
            #speed_mod, _ = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=True,
            #        rotThresh=0.169, posThresh=0.002173)

            # TODO: Implement translation estimation in Python
            # For now, save predicted quaternion and run separate MATLAB function
            q_all.append(q_pr[[3,0,1,2]])

        # Update
        test_time_meter.update((time.time()-start)*1000, B)
        err_q_meter.update(err_q, B)
        #err_t_meter.update(err_t, B)
        #speed_meter.update(speed_raw, B)
        #speed_meter_th.update(speed_mod, B)
        #acc_meter.update(acc*100, B)

        report_progress(epoch=epoch, lr=np.nan, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=test_time_meter, is_train=False, eT=err_t_meter, eR=err_q_meter,
                        speed=speed_meter, acc=acc_meter)

    # log
    if writer is not None:
        writer.add_scalar('Valid/err_q [deg]', err_q_meter.avg, epoch)
        #writer.add_scalar('Valid/err_t [m]',   err_t_meter.avg, epoch)
        #writer.add_scalar('Valid/speed (raw) [-]', speed_meter.avg, epoch)
        #writer.add_scalar('Valid/speed (thr) [-]', speed_meter_th.avg, epoch)

    # return err_q_meter.avg, err_t_meter.avg, speed_meter.avg, speed_meter_th.avg

    # ! Temporary: return q_all instead of err_t
    return err_q_meter.avg, q_all, None, None

def keypts_to_pose(x_pr, y_pr, roi, corners3D, cameraMatrix, distCoeffs=np.zeros((1,5))):
    # x_pr, y_pr  [torch.Tensor]  11,
    # roi         [numpy.ndarray]
    corners2D_pr = torch.cat((x_pr.unsqueeze(0), y_pr.unsqueeze(0)), dim=0) # [2 x 11]
    corners2D_pr = corners2D_pr.cpu().t().numpy() # [11 x 2]

    # Apply RoI
    xmin, xmax, ymin, ymax = roi
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * (xmax-xmin) + xmin
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * (ymax-ymin) + ymin

    # Compute [R|t] by pnp
    q_pr, t_pr = pnp(corners3D, corners2D_pr, cameraMatrix, distCoeffs)

    return q_pr, t_pr