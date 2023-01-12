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

def error_translation(t_pr, t_gt):
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))

    return np.sqrt(np.sum(np.square(t_gt - t_pr)))

def error_orientation(q_pr, q_gt):
    # q must be [qvec, qcos]
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    qdot = np.abs(np.dot(q_pr, q_gt))
    qdot = np.minimum(qdot, 1.0)
    return np.rad2deg(2*np.arccos(qdot)) # [deg]

def speed_score(t_pr, q_pr, t_gt, q_gt, applyThresh=True, rotThresh=0.5, posThresh=0.005):
    # rotThresh: rotation threshold [deg]
    # posThresh: normalized translation threshold [m/m]
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(q_pr, q_gt) # [deg]

    t_gt = np.reshape(t_gt, (3,))
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_r = np.deg2rad(err_q)

    # Check if within threshold
    if applyThresh and err_q < rotThresh:
        speed_q = 0.0

    if applyThresh and speed_t < posThresh:
        speed_t = 0.0

    speed = speed_t + speed_q

    # Accuracy of within threshold
    acc   = float(err_q < rotThresh and speed_t < posThresh)

    return speed, acc