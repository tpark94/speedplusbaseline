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

import numpy as np
import random
import os
import sys
import logging
import cv2
import json
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

logger= logging.getLogger(__name__)

# -----------------------------------------------------------------------
# For reporting & logging
class AverageMeter(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self, unit='-'):
        self.reset()
        self.unit = unit

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count != 0 else 0

def setup_logger(phase):
    # # Where to?
    # time_str = time.strftime('%Y-%m-%d-%H-%M')
    # log_file = '{}_{}.log'.format(phase, time_str)
    # final_log_file = os.path.join(final_output_dir, log_file)

    # Configure logger
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=head, datefmt='%Y/%m/%d %H:%M:%S')
    # logging.basicConfig(filename=str(final_log_file),
    #                     format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # console = logging.StreamHandler()
    # logging.getLogger('').addHandler(console)

    return logger

def report_progress(epoch, lr, epoch_iter, epoch_size, time, is_train=True, **kwargs):
    # Construct message
    blength = 30
    percent = float(epoch_iter / epoch_size)
    arrow   = '█' * int(round(percent * blength))
    spaces  = ' ' * (blength - len(arrow))
    msg = "\rTraining " if is_train else "\rTesting "

    msg += "{epoch:03d} (lr: {lr:.5f}): {iter:04d}/{esize:04d} [{prog}{percent:03d}%] [{time_v:.0f} ({time_a:.0f}) ms] ".format(
        epoch=epoch, lr=lr, iter=epoch_iter, esize=epoch_size, time_v=time.val, time_a=time.avg,
        prog=arrow+spaces, percent=round(percent*100))

    # Add losses to report
    for key, item in kwargs.items():
        if item is not None:
            msg += '{}: {:.2f} ({:.2f}) [{}] '.format(key, item.val, item.avg, item.unit)

    # Report loss
    sys.stdout.write(msg)
    sys.stdout.flush()

    # To next line if at the end of the epoch
    if epoch_iter == epoch_size:
        sys.stdout.write('\n')
        sys.stdout.flush()

# -----------------------------------------------------------------------
# Functions regarding model training
def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    logger.info('Checkpoint saved to {}'.format(os.path.join(output_dir, filename)))

    if is_best and 'state_dict' in states:
        torch.save(
            states['state_dict'],
            os.path.join(output_dir, 'model_best.pth.tar')
        )
        logger.info('Best model saved to {}'.format(os.path.join(output_dir, 'model_best.pth.tar')))

def load_checkpoint(checkpoint_file, model, optimizer, device):
    load_dict = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(load_dict['state_dict'], strict=True)

    # Optimizer state_dict saved as CUDA tensor, must load them manually in CUDA
    if optimizer is not None:
        optimizer.load_state_dict(load_dict['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    logger.info('Checkpoint loaded from {} at epoch {}'.format(checkpoint_file, load_dict['epoch']))

    return load_dict['epoch'], load_dict['best_score']

# -----------------------------------------------------------------------
# Functions regarding 3D keypoints projection
def weighted_mean_quaternion(qs, weights=None):
    if qs.shape[1] != 4:
        qs = np.transpose(qs) # [N x 4]

    # Weights?
    if weights is None:
        weights = np.ones((qs.shape[0],), dtype=np.float32)

    # To R
    Rs = R.from_quat(qs)

    # Weighted average
    q = Rs.mean(weights).as_quat() # [1 x 4]

    return q

def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def project_keypoints(q, r, K, dist, keypoints):
    """ Projecting 3D keypoints to 2D
        q: quaternion (np.array)
        r: position   (np.array)
        K: camera intrinsic (3,3) (np.array)
        dist: distortion coefficients (5,) (np.array)
        keypoints: N x 3 or 3 x N (np.array)
    """
    # Make sure keypoints are 3 x N
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    # transformation to image frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:] # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
    x  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
    y  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((K[0,0]*x + K[0,2], K[1,1]*y + K[1,2]))

    return points2D

def pnp(points_3D, points_2D, cameraMatrix, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=False):
    if distCoeffs is None:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    points_3D = np.ascontiguousarray(points_3D).reshape((-1,1,3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1,1,2))

    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D,
                               cameraMatrix,
                               distCoeffs, rvec, tvec, useExtrinsicGuess,
                               flags=cv2.SOLVEPNP_EPNP)

    R_pr, _ = cv2.Rodrigues(R_exp)

    q = R.from_matrix(R_pr).as_quat() # [qvec, qcos]

    return q, np.squeeze(t)

# -----------------------------------------------------------------------
# Functions regarding 3D model loading
def load_tango_3d_keypoints(mat_dir):
    vertices  = loadmat(mat_dir)['tango3Dpoints_refined'] # [3 x 11]
    corners3D = np.transpose(np.array(vertices, dtype=np.float32))

    return corners3D

def load_camera_intrinsics(camera_json):
    with open(camera_json) as f:
        cam = json.load(f)
    cameraMatrix = np.array(cam['cameraMatrix'], dtype=np.float32)
    distCoeffs   = np.array(cam['distCoeffs'], dtype=np.float32)

    return cameraMatrix, distCoeffs

# -----------------------------------------------------------------------
# Miscellaneous functions.
def set_all_seeds(seed, cfg, use_cuda):
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled       = True

def compute_mean_std(val_loader):
    mu  = np.zeros(3)
    std = np.zeros(3)
    for idx, (x,_,_) in enumerate(val_loader):
        x = x.numpy()
        mu  += np.mean(x, axis=(0,2,3))
        std += np.std(x, axis=(0,2,3))
    return mu/len(val_loader), std/len(val_loader)

def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum(p.numel() for p in model_parameters)