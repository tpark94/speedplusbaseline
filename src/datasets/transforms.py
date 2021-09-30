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

import cv2
import numpy as np

import torchvision.transforms.functional as T

class BrightnessContrast(object):
    def __init__(self, alpha=(0.5, 2.0), beta=(-25, 25)):
        self.alpha = alpha
        self.beta  = beta

    def __call__(self, image, bbox):
        a = np.random.rand() * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        b = np.random.rand() * (self.beta[1]  - self.beta[0])  + self.beta[0]
        image = cv2.convertScaleAbs(image, alpha=a, beta=b)
        return image, bbox

class GaussianNoise(object):
    def __init__(self, mean=0, std=25):
        self.mean = mean
        self.std  = std

    def __call__(self, image, bbox):
        noise = np.zeros(image.shape, np.uint8)
        cv2.randn(noise, mean=self.mean, stddev=self.std)
        image = cv2.add(image, noise)
        return image, bbox

class RandomCrop(object):
    def __init__(self, output_shape, is_train=True):
        self.shape    = output_shape
        self.is_train = is_train

    def __call__(self, image, bbox):
        # bbox: [xmin, xmax, ymin, ymax] (pix)
        xmin, xmax, ymin, ymax = bbox
        w, h = xmax-xmin, ymax-ymin
        x, y = xmin+w/2.0, ymin+h/2.0

        # Original image shape
        org_h, org_w, _ = image.shape

        # Make sure the RoI is SQUARE, as is most input to the CNN
        roi_size = max((w, h))

        # Since the 2D bounding box is supposedly very "tight",
        # Give some extra room in both directions
        if self.is_train:
            # Enlarge tight RoI by random factor within [1, 1.5]
            roi_size  = (1 + 0.5*np.random.rand()) * roi_size

            # Shift expanded RoI by random factor as well
            # Factor within range of [-f*roi_size, +f*roi_size]
            fx = 0.1 * (np.random.rand()*2 - 1) * roi_size
            fy = 0.1 * (np.random.rand()*2 - 1) * roi_size
        else:
            # For testing, just enlarge by fixed amount
            roi_size = (1 + 0.2) * roi_size
            fx = fy = 0

        # Construct new RoI
        xmin = max(0, int(x - roi_size/2.0 + fx))
        xmax = min(org_w, int(x + roi_size/2.0 + fx))
        ymin = max(0, int(y - roi_size/2.0 + fy))
        ymax = min(org_h, int(y + roi_size/2.0 + fy))

        bbox = np.array([xmin, xmax, ymin, ymax], dtype=np.float32)

        # Crop and resize
        image = cv2.resize(image[ymin:ymax, xmin:xmax], self.shape)

        return image, bbox

class ResizeCrop(object):
    def __init__(self, output_shape):
        self.shape = output_shape

    def __call__(self, image, bbox):
        # bbox: [xmin, xmax, ymin, ymax] (pix)
        xmin, xmax, ymin, ymax = bbox

        # Original image shape
        org_h, org_w, _ = image.shape

        # Make sure bbox is within image frame
        xmin = max(0, int(xmin))
        xmax = min(org_w, int(xmax))
        ymin = max(0, int(ymin))
        ymax = min(org_h, int(ymax))

        # Crop and resize
        image = cv2.resize(image[ymin:ymax, xmin:xmax], self.shape)

        return image, bbox

class ToTensor(object):
    def __call__(self, image, bbox):
        return T.to_tensor(image), bbox

class RandomApply(object):
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, bbox):
        for t in self.transforms:
            if np.random.rand() < self.p:
                image, bbox = t(image, bbox)

        return image, bbox

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox

def build_transforms(model_name, input_size, p_aug=0.5, is_train=True):
    # First, resize & crop
    if model_name == 'krn':
        transforms = [RandomCrop(input_size, is_train)]
    elif model_name == 'spn':
        transforms = [ResizeCrop(input_size)]

    # Add augmentation if training for KRN, skip if not
    if is_train and model_name == 'krn':
        augmentations = [
            RandomApply(
                [BrightnessContrast(alpha=(0.5,2.0), beta=(-25,25)),
                 GaussianNoise(std=25)],
            p=p_aug)
        ]
        transforms = transforms + augmentations

    # To tensor
    transforms = transforms + [ToTensor()]

    # Compose and return
    return Compose(transforms)