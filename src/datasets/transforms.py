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
import torchvision.transforms.functional as T

'''
As noted in https://pytorch.org/tutorials/beginner/data_loading_tutorial.html,
it's best to use torch to generate random numbers, otherwise each worker
must be initialized with different random seeds when using, e.g., numpy,
to prevent all images in the same batch ending up with identical augmentations.
'''

class Rotate(object):
    ''' Rotate image randomly by 90 deg intervals '''
    def __call__(self, data, bbox, keypts):
        angle = 90*float(torch.randint(1, 4, (1,)))

        # Rotate image
        data = T.rotate(data, angle)

        # Rotate keypoints
        x, y = keypts[0].clone(), keypts[1].clone()
        if angle == 90: # 90 deg
            keypts[0], keypts[1] = y, 1.0 - x
        elif angle == 180: # 180 deg
            keypts[0], keypts[1] = 1.0 - x, 1.0 - y
        elif angle == 270: # 270 deg
            keypts[0], keypts[1] = 1.0 - y, x

        return data, bbox, keypts

class Flip(object):
    ''' Flip image randomly either horizontally or vertically '''
    def __call__(self, data, bbox, keypts):
        if torch.rand(1) < 0.5:
            # horizontal flip
            data = T.hflip(data)
            keypts[0] = 1.0 - keypts[0]
        else:
            # vertical flip
            data = T.vflip(data)
            keypts[1] = 1.0 - keypts[1]

        return data, bbox, keypts

class BrightnessContrast(object):
    """ Adjust brightness and contrast of the image in a fashion of
        OpenCV's convertScaleAbs, where

        newImage = alpha * image + beta

        image: torch.Tensor image (0 ~ 1)
        alpha: multiplicative factor
        beta:  additive factor (0 ~ 255)
    """
    def __init__(self, alpha=(0.5, 2.0), beta=(-25, 25)):
        self.alpha = torch.tensor(alpha).log()
        self.beta  = torch.tensor(beta)/255

    def __call__(self, image, bbox, keypts):
        # Contrast - multiplicative factor
        loga = torch.rand(1) * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        a = loga.exp()

        # Brightness - additive factor
        b = torch.rand(1) * (self.beta[1]  - self.beta[0])  + self.beta[0]

        # Apply
        image = torch.clamp(a*image + b, 0, 1)

        return image, bbox, keypts

class GaussianNoise(object):
    """ Add random Gaussian white noise

        image: torch.Tensor image (0 ~ 1)
        std:   noise standard deviation (0 ~ 255)
    """
    def __init__(self, std=25):
        self.std = std/255

    def __call__(self, image, bbox, keypts):
        noise = torch.randn(image.shape, dtype=torch.float32) * self.std
        image = torch.clamp(image + noise, 0, 1)
        return image, bbox, keypts

class RandomCrop(object):
    ''' Crop the image with random bounding box fully containing
        the satellite foreground
    '''
    def __init__(self, output_shape, is_train=True):
        self.shape    = output_shape
        self.is_train = is_train

    def __call__(self, image, bbox, keypts):
        # bbox: [xmin, xmax, ymin, ymax] (pix)
        # keypts: [2 x Nk] (pix)
        xmin, xmax, ymin, ymax = bbox
        w, h = xmax-xmin, ymax-ymin
        x, y = xmin+w/2.0, ymin+h/2.0

        # Original image shape
        org_w, org_h = image.size

        # Make sure the RoI is SQUARE, as is most input to the CNN
        roi_size = max((w, h))

        # Since the 2D bounding box is supposedly very "tight",
        # Give some extra room in both directions
        if self.is_train:
            # Enlarge tight RoI by random factor within [1, 1.5]
            roi_size  = (1 + 0.5 * torch.rand(1)) * roi_size

            # Shift expanded RoI by random factor as well
            # Factor within range of [-f*roi_size, +f*roi_size]
            fx = 0.2 * (torch.rand(1)*2 - 1) * roi_size
            fy = 0.2 * (torch.rand(1)*2 - 1) * roi_size
        else:
            # For testing, just enlarge by fixed amount
            roi_size = (1 + 0.2) * roi_size
            fx = fy = 0

        # Construct new RoI
        xmin = max(0, int(x - roi_size/2.0 + fx))
        xmax = min(org_w, int(x + roi_size/2.0 + fx))
        ymin = max(0, int(y - roi_size/2.0 + fy))
        ymax = min(org_h, int(y + roi_size/2.0 + fy))

        bbox = torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float32)

        # Adjust keypoints (0 ~ 1)
        keypts = torch.tensor(keypts, dtype=torch.float32)
        keypts[0] = (keypts[0] - xmin) / (xmax - xmin)
        keypts[1] = (keypts[1] - ymin) / (ymax - ymin)

        # Crop and resize
        image = T.resized_crop(image, ymin, xmin, ymax-ymin, xmax-xmin, self.shape)

        return image, bbox, keypts

class ResizeCrop(object):
    ''' Resize and crop image given bounding box'''
    def __init__(self, output_shape):
        self.shape = output_shape

    def __call__(self, image, bbox, keypts):
        # bbox: [xmin, xmax, ymin, ymax] (pix)
        xmin, xmax, ymin, ymax = bbox

        # Original image shape
        org_w, org_h = image.size

        # Make sure bbox is within image frame
        xmin = max(0, int(xmin))
        xmax = min(org_w, int(xmax))
        ymin = max(0, int(ymin))
        ymax = min(org_h, int(ymax))

        # Crop and resize
        image = T.resized_crop(image, ymin, xmin, ymax-ymin, xmax-xmin, self.shape)

        # For SPN, return original bounding box
        bbox  = torch.tensor(bbox, dtype=torch.float32)

        return image, bbox, keypts

class ToTensor(object):
    ''' Same as torchvision.ToTensor(), but passes extra arguments'''
    def __call__(self, image, bbox, keypts):
        # float tensor [0, 1]
        return T.to_tensor(image).type(torch.float32), bbox, keypts

class RandomApply(object):
    ''' Sameas torchvision.RandomApply(), but randomly apply EACH transform
        instead of the whole set of transforms
    '''
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, bbox, keypts):
        for t in self.transforms:
            if torch.rand(1) < self.p:
                image, bbox, keypts = t(image, bbox, keypts)

        return image, bbox, keypts

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox, keypts):
        for t in self.transforms:
            image, bbox, keypts = t(image, bbox, keypts)
        return image, bbox, keypts

def build_transforms(model_name, input_size, p_aug=0.5, is_train=True):
    # First, resize & crop image
    if model_name == 'krn':
        transforms = [RandomCrop(input_size, is_train)]
    elif model_name == 'spn':
        transforms = [ResizeCrop(input_size)]

    # Image to tensor [0, 1] before augmentation
    transforms = transforms + [ToTensor()]

    # Add augmentation if training for KRN, skip if not
    if is_train and model_name == 'krn':
        augmentations = [
            RandomApply(
                [Rotate(), Flip(),
                 BrightnessContrast(alpha=(0.5,2.0), beta=(-25,25)),
                 GaussianNoise(std=25)],
            p=p_aug)
        ]
        transforms = transforms + augmentations

    # Compose and return
    return Compose(transforms)