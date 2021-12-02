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

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.utils import quat2dcm, project_keypoints

def compute_position_spn(q_vbs2tango, bbox, corners3D, cameraMatrix, distCoeffs=np.zeros((1,5))):
    ''' Compute position vector for SPN model
    Arguments:
        q_vbs2tango: (4,) numpy.ndarray - predicted unit quaternion (scalar-first)
        bbox:        (4,) numpy.ndarray - bounding box [xmin, xmax, ymin, ymax] (pix)
        ...
    Returns:
        r_Vo2To_vbs: (3,) numpy.ndarray - predicted position vector (m)
    '''
    maxModelLength = 1.246 # [m] for Tango

    # Bounding box decomposition
    xmin, ymin, width, height = bbox[0], bbox[2], bbox[1]-bbox[0], bbox[3]-bbox[2]

    # Initial position guess based on similar triangles
    boxSize   = np.sqrt(width**2 + height**2)
    boxCenter = np.array([xmin + width/2.0, ymin + height/2.0])
    offsetPx  = np.array([boxCenter[0] - cameraMatrix[0,2],
                          boxCenter[1] - cameraMatrix[1,2]])
    az = np.arctan(offsetPx[0]/cameraMatrix[0,0]) # [rad]
    el = np.arctan(offsetPx[1]/cameraMatrix[1,1])
    range_wge = cameraMatrix[0,0] * maxModelLength / boxSize # [m]
    Ry = R.from_euler('y', -az).as_matrix()
    Rx = R.from_euler('x', -el).as_matrix()
    r_Vo2To_vbs = Ry @ Rx @ np.reshape(np.array([0, 0, range_wge]), (3,1))

    # NEWTON's METHOD
    maxIter = 50
    tolerance = 5e-10
    iter = 0
    dx = 1 + 1e-15

    # Initialize betas
    beta_old = np.squeeze(r_Vo2To_vbs)

    while dx > tolerance and iter <= maxIter:
        # Compute extreme reprojected points in VBS frame
        r_Vo2X_vbs = _compute_extremal_points(q_vbs2tango, beta_old, corners3D, cameraMatrix) # [4 x 3]

        # Compute update to beta
        r = _calc_residuals(r_Vo2X_vbs, cameraMatrix, distCoeffs, beta_old, bbox)
        J = _calc_jacobian(r_Vo2X_vbs, cameraMatrix, distCoeffs, beta_old)
        beta_new = beta_old - np.squeeze(np.linalg.inv(np.transpose(J) @ J) @ np.transpose(J) @ np.reshape(r, (4,1)))

        # Compute change between new and oldbeta
        dx = np.linalg.norm(beta_new - beta_old)

        # Updates
        iter = iter + 1
        beta_old = beta_new

    r_Vo2To_vbs = beta_new

    return r_Vo2To_vbs

def _compute_extremal_points(q_vbs2tango, r_Vo2To_vbs, tangoPoints, cameraMatrix):
    ''' Compute the extremal points of the Tango model given orientation and position estimates '''
    reprImagePoints = project_keypoints(q_vbs2tango, r_Vo2To_vbs,
                                cameraMatrix, np.zeros((5,)), tangoPoints)
    idx1 = np.argmin(reprImagePoints[0]) # xmin
    idx2 = np.argmin(reprImagePoints[1]) # ymin
    idx3 = np.argmax(reprImagePoints[0]) # xmax
    idx4 = np.argmax(reprImagePoints[1]) # ymax

    if tangoPoints.shape[0] != 3:
        tangoPoints = np.transpose(tangoPoints)
    tangoPoints_vbs = np.transpose(quat2dcm(q_vbs2tango)) @ tangoPoints

    r_Vo2X_vbs = np.zeros((4, 3))
    r_Vo2X_vbs[0] = tangoPoints_vbs[:,idx1] # left-most point
    r_Vo2X_vbs[1] = tangoPoints_vbs[:,idx3] # right-most point
    r_Vo2X_vbs[2] = tangoPoints_vbs[:,idx2] # top-most point
    r_Vo2X_vbs[3] = tangoPoints_vbs[:,idx4] # bottom-most point

    return r_Vo2X_vbs

def _calc_residuals(r_Vo2X_vbs, cameraMatrix, distCoeffs, r_Vo2To_vbs, bbox):
    ''' Compute residuals of projected extremal points against the bounding box '''
    Tx, Ty, Tz = r_Vo2To_vbs
    Bx1, Bx2, By1, By2 = bbox

    xs, ys = [], []
    for ii in range(4):
        # Project
        Rx, Ry, Rz = r_Vo2X_vbs[ii]
        x0 = (Rx + Tx) / (Rz + Tz)
        y0 = (Ry + Ty) / (Rz + Tz)

        # Distortion
        r2 = x0*x0 + y0*y0
        cdist = 1 + distCoeffs[0]*r2 + distCoeffs[1]*r2*r2 + distCoeffs[4]*r2*r2*r2
        x  = x0*cdist + distCoeffs[2]*2*x0*y0 + distCoeffs[3]*(r2 + 2*x0*x0)
        y  = y0*cdist + distCoeffs[2]*(r2 + 2*y0*y0) + distCoeffs[3]*2*x0*y0

        # Apply camera
        xs.append(cameraMatrix[0,0]*x + cameraMatrix[0,2])
        ys.append(cameraMatrix[1,1]*y + cameraMatrix[1,2])

    # Residuals
    r1 = xs[0] - Bx1
    r2 = xs[1] - Bx2
    r3 = ys[2] - By1
    r4 = ys[3] - By2

    return np.array([r1, r2, r3, r4])

def _calc_jacobian(r_Vo2X_vbs, cameraMatrix, distCoeffs, r_Vo2To_vbs):
    ''' Compute jacobian of the residuals.
        Camera distortion coefficients are neglected at the moment.
    '''
    fx, fy = cameraMatrix[0,0], cameraMatrix[1,1]
    Tx, Ty, Tz = r_Vo2To_vbs
    Rx_left, Rz_left = r_Vo2X_vbs[0,0], r_Vo2X_vbs[0,2]
    Rx_right, Rz_right = r_Vo2X_vbs[1,0], r_Vo2X_vbs[1,2]
    Ry_top, Rz_top = r_Vo2X_vbs[2,1], r_Vo2X_vbs[2,2]
    Ry_bot, Rz_bot = r_Vo2X_vbs[3,1], r_Vo2X_vbs[3,2]

    # Left-most image feature
    dr1db1 = fx / (Rz_left + Tz)
    dr1db2 = 0
    dr1db3 = -fx * (Rx_left + Tx) / (Rz_left + Tz)**2

    # Right-most iamge feature
    dr2db1 = fx / (Rz_right + Tz)
    dr2db2 = 0
    dr2db3 = -fx * (Rx_right + Tx) / (Rz_right + Tz)**2

    # Top-most image feature
    dr3db1 = 0
    dr3db2 = fy / (Rz_top + Tz)
    dr3db3 = -fy * (Ry_top + Ty) / (Rz_top + Tz)**2

    # Bottom-most image feature
    dr4db1 = 0
    dr4db2 = fy / (Rz_bot + Tz)
    dr4db3 = -fy * (Ry_bot + Ty) / (Rz_bot + Tz)**2

    # Jacobian
    J = np.array([[dr1db1, dr1db2, dr1db3],
                  [dr2db1, dr2db2, dr2db3],
                  [dr3db1, dr3db2, dr3db3],
                  [dr4db1, dr4db2, dr4db3]], dtype=np.float32)

    return J