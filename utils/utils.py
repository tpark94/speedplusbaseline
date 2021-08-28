import torch
import numpy as np
import random
import time
import os
import sys

from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------
# For reporting training progress
class AverageMeter(object):
    """ Computes and stores the average and current value.
        Copied from https://github.com/locuslab/smoothing/blob/master/code/train_utils.py
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
        self.avg    = self.sum / self.count

# -----------------------------------------------------------------------
# Functions regarding model training
def save_checkpoint(epoch, cfg, model, optimizer, last=False):
    checkpoint_dir = create_dir_name(cfg.projroot, cfg.savedir)

    # Path to save weights
    if last:
        filename = os.path.join(checkpoint_dir, 'final.pth')
    else:
        filename = os.path.join(checkpoint_dir, 'e{:04d}.pth'.format(epoch))

    # Save weight
    torch.save({'epoch': epoch,
                'weights': model.state_dict(),
                'optim': optimizer.state_dict()}, filename)

    # Print
    logging('Weights saved to ' + filename)

def load_checkpoint(epoch, cfg, model, optimizer, device):
    path = os.path.join(cfg.projroot, cfg.savedir, "e{:04d}.pth".format(epoch))
    load_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(load_dict['weights'], strict=True)

    # Optimizer state_dict saved as CUDA tensor, must load them manually in CUDA
    optimizer.load_state_dict(load_dict['optim'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    logging('Checkpoints loaded from {:s}'.format(path))

def initialize_conv_weights(model):
    for key in model.state_dict():
        if key.split('.')[-1] == 'weight':
            if model.state_dict()[key].dim() == 4:
                # Conv2d
                torch.nn.init.xavier_uniform_(model.state_dict()[key])

# -----------------------------------------------------------------------
# Functions regarding 3D keypoints projection
def get_camera_intrinsic(camera):
    """ pointgrey: camera model used for SPEED
        prisma:    camera model for PRISMA mission images
    """
    K = np.zeros((3, 3), dtype=np.float32)

    if camera == 'prisma':
        # Camera matrix
        K[0, 0], K[0, 2] = 2323.604651162790, 376.0
        K[1, 1], K[1, 2] = 2323.604651162790, 290.0

        # Screen pixel sizes (width, height)
        N = [ 752, 580 ]

    elif camera == 'pointgrey':
        K[0, 0], K[0, 2] = 3003.412969283277, 960.0
        K[1, 1], K[1, 2] = 3003.412969283277, 600.0

        # Screen pixel sizes (width, height)
        N = [ 1920, 1200 ]

    else:
        raise Exception('Not a valid camera')

    K[2, 2] = 1.0
    return K, N

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
        K: camera intrinsic (3x3 np.array)
        dist:
        keypoints: N x 3 (np.array)
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

# -----------------------------------------------------------------------
# Functions regarding 3D model loading
def load_surface_points(mat_dir):
    vertices  = loadmat(mat_dir)['tango3Dpoints_refined']
    corners3D = np.array(np.transpose(vertices), dtype='float32')

    return corners3D

# -----------------------------------------------------------------------
# Miscellaneous functions.
def set_all_seeds(seed, use_cuda):
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = True

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

def logging(message):
    print('%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), message))

def _report(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def report_progress(epoch, lr, epoch_iter, epoch_size, time, mode='train', **kwargs):
    # Construct message
    blength = 30
    percent = float(epoch_iter / epoch_size)
    arrow   = '=' * int(round(percent * blength))
    spaces  = ' ' * (blength - len(arrow))
    if mode == 'train':
        msg = "\rTraining "
    elif mode == 'val':
        msg = "\rValidating "

    msg += "{epoch:03d} (lr: {lr:.5f}): {iter:04d}/{esize:04d} [{prog}{percent:03d}%] [{time_v:.2f} ({time_a:.2f}) ms] ".format(
        epoch=epoch, lr=lr, iter=epoch_iter, esize=epoch_size, time_v=time.val, time_a=time.avg,
        prog=arrow+spaces, percent=round(percent*100))

    # Add losses to report
    for key, item in kwargs.items():
        if item is not None:
            msg += '{}: {:.2f} ({:.2f}) [{}] '.format(key, item.val, item.avg, item.unit)

    # Report loss
    _report(msg)

    # To next line if at the end of the epoch
    if epoch_iter == epoch_size: _report('\n')

def create_dir_name(*args):
    # Create directory name similar to fullfile(),
    # then make the directory if it doesn't exist
    name = ""
    for arg in args:
        name = os.path.join(name, arg)

    if not os.path.exists(name):
        os.makedirs(name)

    return name

def log_csv(file, epoch, *args):
    msg = "{}".format(epoch)
    for item in args:
        msg += ',{}'.format(item)

    f = open(file+'.csv', 'a')
    f.write(msg+"\n")
    f.close()