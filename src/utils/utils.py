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
    ''' Compute weighted mean of N unit quaternions.
    Arguments:
        qs: (N, 4) or (4, N) numpy.ndarray - unit quaternions (scalar-first)
    Returns:
        q: (4,) numpy.ndarray - weighted mean unit quaternion (scalar-first)
    '''
    # Size check
    if qs.shape[1] != 4:
        qs = np.transpose(qs) # (N,4)

    # Scipy uses scalar-last convention
    qs = qs[:,[1,2,3,0]]

    # Weights?
    if weights is None:
        weights = np.ones((qs.shape[0],), dtype=np.float32)

    # Quaternions to rotation matrices
    Rs = R.from_quat(qs)

    # Weighted average
    q = Rs.mean(weights).as_quat() # (4,)

    # Back to scalar-first convention
    q = q[[3,0,1,2]]

    return q

def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. 
    Arguments:
        q: (4,) numpy.ndarray - unit quaternion (scalar-first)
    Returns:
        dcm: (3,3) numpy.ndarray - corresponding DCM
    """

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

def project_keypoints(q_vbs2tango, r_Vo2To_vbs, cameraMatrix, distCoeffs, keypoints):
    ''' Project keypoints.
    Arguments:
        q_vbs2tango:  (4,) numpy.ndarray - unit quaternion from VBS to Tango frame
        r_Vo2To_vbs:  (3,) numpy.ndarray - position vector from VBS to Tango in VBS frame (m)
        cameraMatrix: (3,3) numpy.ndarray - camera intrinsics matrix
        distCoeffs:   (5,) numpy.ndarray - camera distortion coefficients in OpenCV convention
        keypoints:    (3,N) or (N,3) numpy.ndarray - 3D keypoint locations (m)
    Returns:
        points2D: (2,N) numpy.ndarray - projected points (pix)
    '''
    # Size check (3,N)
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    # transformation to image frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q_vbs2tango)),
                          np.expand_dims(r_Vo2To_vbs, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:] # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + distCoeffs[0]*r2 + distCoeffs[1]*r2*r2 + distCoeffs[4]*r2*r2*r2
    x  = x0*cdist + distCoeffs[2]*2*x0*y0 + distCoeffs[3]*(r2 + 2*x0*x0)
    y  = y0*cdist + distCoeffs[2]*(r2 + 2*y0*y0) + distCoeffs[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((cameraMatrix[0,0]*x + cameraMatrix[0,2],
                          cameraMatrix[1,1]*y + cameraMatrix[1,2]))

    return points2D

def pnp(points_3D, points_2D, cameraMatrix, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=False):
    ''' Perform EPnP using OpenCV.
    Arguments:
        points_3D:  (N,3) numpy.ndarray - 3D coordinates
        points_2D:  (N,2) numpy.ndarray - 2D coordinates
        ...
    Returns:
        q_pr: (4,) numpy.ndarray - unit quaternion (scalar-first)
        t_pr: (3,) numpy.ndarray - position vector (m)
    '''
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

    # Rotation matrix to quaternion
    q_pr = R.from_matrix(np.squeeze(R_pr)).as_quat()

    # Convert to scalar-first
    q_pr = q_pr[[3,0,1,2]]

    return q_pr, np.squeeze(t)

# -----------------------------------------------------------------------
# Functions regarding 3D model loading
def load_tango_3d_keypoints(mat_dir):
    vertices  = loadmat(mat_dir)['tango3Dpoints'] # [3 x 11]
    corners3D = np.transpose(np.array(vertices, dtype=np.float32)) # [11 x 3]

    return corners3D

def load_camera_intrinsics(camera_json):
    with open(camera_json) as f:
        cam = json.load(f)
    cameraMatrix = np.array(cam['cameraMatrix'], dtype=np.float32)
    distCoeffs   = np.array(cam['distCoeffs'],   dtype=np.float32)

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