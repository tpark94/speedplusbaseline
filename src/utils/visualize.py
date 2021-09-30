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

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import patches as patches

import cv2
import numpy as np

def to_numpy_image(image):
    return image.mul(255).clamp(0,255).permute(1,2,0).byte().cpu().numpy()

def imshow(image, savefn=None):
    # image: torch.tensor (3 x H x W)

    # image to numpy.array
    image = to_numpy_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # plot
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    if savefn is not None:
        plt.savefig(savefn, bbox_inches='tight', pad_inches=0)

def plot_2D_bbox(tensor, bbox, xywh=True):
    # bbox: [xmin, xmax, ymin, ymax]

    # Processing
    data = to_numpy_image(tensor)

    if xywh:
        x, y, w, h = bbox
        xmin, xmax, ymin, ymax = x-w/2, x+w/2, y-h/2, y+h/2
    else:
        xmin, xmax, ymin, ymax = bbox

    # figure
    fig = plt.figure()
    plt.imshow(data)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 
                color='g', linewidth=1.5)
    plt.show()

def scatter_keypoints(tensor, x_pr, y_pr, normalized=False):
    # Tensor:     C x H x W
    # Keypoints : Nc x 2
    _, h, w = tensor.shape
    data  = to_numpy_image(tensor)

    if normalized:
        x_pr = x_pr * w
        y_pr = y_pr * h

    # figure
    fig = plt.figure()
    plt.imshow(data)
    plt.scatter(x_pr , y_pr, c='lime', marker='+')
    plt.show()