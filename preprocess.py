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

import os
import json
import numpy as np
import argparse
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.utils.utils import load_tango_3d_keypoints, load_camera_intrinsics, project_keypoints

PROJROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplusbaseline',
               'linux': '/media/shared/Jeff/SLAB/speedplusbaseline'}

DATAROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplus/data/datasets',
               'linux': '/home/jeffpark/SLAB/Dataset'}

parser = argparse.ArgumentParser('Generating CSV files')
parser.add_argument('--projroot',   type=str, default=PROJROOTDIR['linux'])
parser.add_argument('--dataroot',   type=str, default=DATAROOTDIR['linux'])
parser.add_argument('--dataname',   type=str, default='speedplus')
parser.add_argument('--model_name', type=str, default='krn')
parser.add_argument('--num_keypoints',   type=int,   default=11)
parser.add_argument('--num_neighbors',   type=int,   default=5)
parser.add_argument('--keypts_3d_model', type=str, default='src/utils/tangoPoints.mat')
parser.add_argument('--attitude_class',  type=str, default='src/utils/attitudeClasses.mat')
parser.add_argument('--domain',   type=str, default='synthetic')
parser.add_argument('--jsonfile', type=str, default='train.json')
parser.add_argument('--csvfile',  type=str, default='splits_krn/train.csv')

cfg = parser.parse_args()

def json2csv(cfg,):
    ''' Create CSV file from JSON file. '''
    assert cfg.model_name=='krn' or cfg.model_name=='spn', 'Model must be either krn or spn'

    # Read labels
    jsonfile = os.path.join(cfg.dataroot, cfg.dataname, cfg.domain, cfg.jsonfile)
    print('Reading from {} ...'.format(jsonfile))
    with open(jsonfile, 'r') as f:
        labels = json.load(f) # list

    # Read camera
    camerafile = os.path.join(cfg.dataroot, cfg.dataname, 'camera.json')
    cameraMatrix, distCoeffs = load_camera_intrinsics(camerafile)

    # Read Tango 3D keypoints
    tango3dmodelfile = os.path.join(cfg.projroot, cfg.keypts_3d_model)
    keypts3d = load_tango_3d_keypoints(tango3dmodelfile) # (11, 3) [m]

    # For SPN, read attitude classes and pre-process
    if cfg.model_name=='spn':
        attClassFile = os.path.join(cfg.projroot, cfg.attitude_class)
        assert attClassFile is not None, '.mat file for attitude classes must be provided for SPN'
        attClasses = loadmat(attClassFile)['qClass'] # [Nclasses x 4]

    # Open CSV file
    outcsvfile = os.path.join(cfg.dataroot, cfg.dataname, cfg.domain, cfg.csvfile)
    csv = open(outcsvfile, 'w')
    print('Writing to {}'.format(outcsvfile))

    for idx in tqdm(range(len(labels))):
        # Filename & pose labels
        filename    = os.path.join(cfg.domain, 'images', labels[idx]['filename'])
        q_vbs2tango = np.array(labels[idx]['q_vbs2tango_true'], dtype=np.float32)
        r_Vo2To_vbs = np.array(labels[idx]['r_Vo2To_vbs_true'], dtype=np.float32)

        # Project Tango keypoints
        keypts2d = project_keypoints(q_vbs2tango, r_Vo2To_vbs, cameraMatrix, distCoeffs, keypts3d) # (2, 11)

        # Bounding box based on projected keypoints
        xmin = np.amin(keypts2d[0])
        xmax = np.amax(keypts2d[0])
        ymin = np.amin(keypts2d[1])
        ymax = np.amax(keypts2d[1])
        bbox = [xmin, xmax, ymin, ymax]

        # Write filename, bbox, and pose labels (common to both SPN and KRN)
        row = [filename] + bbox + q_vbs2tango.tolist() + r_Vo2To_vbs.tolist()

        if cfg.model_name == 'krn':
            # Keypoints into 1-D array (x0, y0, x1, y1, ...)
            keypts2d = np.reshape(np.transpose(keypts2d), (2*cfg.num_keypoints,))
            row = row + keypts2d.tolist()
        else:
            # Orientation into adjacent attitude classes and weights
            [attClassLabel, attWeightVector] = _get_quat_bins(q_vbs2tango, attClasses, cfg.num_neighbors)
            row = row + attClassLabel.tolist() + attWeightVector.tolist()

        # Write to CSV
        row = ', '.join([str(e) for e in row])

        # Write
        csv.write(row + '\n')

    csv.close()

def _get_quat_bins(qPose, qClass, numNeighbors):
    ''' For each quaternion in `qPose`, find the nearest quaternion in `qClass` and
        the weights associated with distance to each class. Based on the original MATLAB 
        implementation by Sumant Sharma.
    Arguments:
        qPose:  (4,)  array of unit quaternion (scalar-first)
        qClass: (N,4) matrix of unit quaternion classes (scalar-first)
        numNeighbors: (int) number of `qClass` covering each of `qOise`
    Returns:
        nClasses: (numNeighbors,) array of closest quaternion classes
        nWeights: (numNeighbors,) array of weights of each quaternion classes
    '''
    # Quaternion classes into scalar-last for scipy
    q      = R.from_quat(qPose[[1,2,3,0]])
    qClass = R.from_quat(qClass[:,[1,2,3,0]])

    # Orientation diff. w.r.t. each entries in qClass [numClasses, 4]
    qDiff = q.inv() * qClass
    qDiff = qDiff.as_quat()

    # Angular distance w.r.t. each entries of qClass [rad]
    angleVec = 2 * np.arccos(np.abs(qDiff[:,-1])) # scalar-last

    # nAngles:  Smallest angular distances [rad]
    # nCLasses: Class indices of such entries
    sortIdx  = np.argsort(angleVec)
    nClasses = sortIdx[:numNeighbors]
    nAngles  = angleVec[nClasses]

    # Angular distances into weights
    nWeights = 1.0 - nAngles / np.pi**2 # (numNeighbors,)
    nWeights = nWeights / np.sum(nWeights)

    return nClasses, nWeights


if __name__=='__main__':

    json2csv(cfg)
