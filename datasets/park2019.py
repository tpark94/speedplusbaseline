import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF

""" << csvfile >> is a path to a .csv file containing the following:
    Path to image:     imagepath,
    Tight RoI Coord.:  xmin, xmax, ymin, ymax     [pix]
    True pose:         q0, q1, q2, q3, t1, t2, t3 [m]
    Keypoint Coord.:   kx1, ky1, ..., kx11, ky11  [pix]
"""

def rotate(data, x, y, w, h):
    """ Randomly rotate data and label by [90, 180, 270] deg
        data:         torch.Tensor
        (x, y, w, h): normalized bbox coord.
    """
    angle = 90*np.random.randint(1,4) # counter-clockwise

    # Rotate image
    data = TF.rotate(data, angle)

    # Rotate label
    if angle == 90:
        x, y, w, h = y, 1-x, h, w
    elif angle == 180:
        x, y = 1-x, 1-y
    elif angle == 270:
        x, y, w, h = 1-y, x, h, w

    return data, x, y, w, h

def flip(data, x, y):
    """ Flip data and bbox either horizontally or vertically
        data:   torch.Tensor
        (x, y): normalized bbox center coord.
    """
    if np.random.rand() < 0.5:
        # Horizontal flip
        data = TF.hflip(data)
        x = 1-x
    else:
        # Vertical flip
        data = TF.vflip(data)
        y = 1 - y

    return data, x, y

def brightness_contrast(data, alpha=(0.5, 1.5), beta=(-25, 25)):
    """ Adjust brightness and contrast of the image in a fashion of
        OpenCV's convertScaleAbs:

        newImage = alpha * image + beta

        data: torch.Tensor image
        alpha: multiplicative factor
        beta:  additive factor (0 ~ 255)
    """
    a = np.random.rand() * (alpha[1] - alpha[0]) + alpha[0]
    b = np.random.rand() * (beta[1]  - beta[0])  + beta[0]

    # Emulate OpenCV convertScaleAbs
    data = torch.clamp(a*data + b/255, 0, 1)

    return data

def gaussian_noise(data, std=25):
    """ Add random Gaussian white noise

        data: torch.Tensor image
        std:  noise standard deviation (0 ~ 255)
    """
    noise = torch.randn(data.shape) * std/255
    data  = torch.clamp(data + noise, 0, 1)
    return data

def randomCrop(bbox_gt, shape=(1920,1200), split='train'):
    """ Given a ground-truth bounding box (bbox_gt), create a new bounding box
        that is randomly enlarged and shifted from bbox_gt

        bbox_gt: [xmin, xmax, ymin, ymax] in pixel coord. (np.array)
    """
    xmin, xmax, ymin, ymax = bbox_gt
    w, h = xmax-xmin, ymax-ymin
    x, y = xmin+w/2.0, ymin+h/2.0

    # Make sure the RoI is SQUARE, as is most input to the CNN
    roi_size = max((w, h))

    # Since the 2D bounding box is supposedly very "tight",
    # Give some extra room in both directions
    f = 0.2
    if split == 'train':
        # Enlarge tight RoI by random factor within [1, 1.5]
        roi_size  = (1 + 0.5*np.random.rand()) * roi_size

        # Shift expanded RoI by random factor as well
        # Factor within range of [-f*roi_size, +f*roi_size]
        fx = f * (np.random.rand()*2 - 1) * roi_size
        fy = f * (np.random.rand()*2 - 1) * roi_size
    else:
        # For testing, just enlarge by fixed amount
        roi_size = (1 + f) * roi_size
        fx = fy = 0

    # Construct new RoI
    xmin = max(0, int(x - roi_size/2.0 + fx))
    xmax = min(shape[0], int(x + roi_size/2.0 + fx))
    ymin = max(0, int(y - roi_size/2.0 + fy))
    ymax = min(shape[1], int(y + roi_size/2.0 + fy))

    # Return [top, left, height, width]
    return ymin, xmin, ymax-ymin, xmax-xmin

class Dataset_Park2019_ODN(Dataset):
    def __init__(self, root, csvfile, shape=(416,416), split='train', load_labels=True):
        # Read .csv file
        self.root        = root
        self.csv         = pd.read_csv(os.path.join(root, csvfile), header=None)
        self.shape       = tuple(shape)
        self.split       = split
        self.load_labels = load_labels

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        assert index < len(self), 'Index range error'

        #------------ Read Images
        imgpath = os.path.join(self.root, self.csv.iloc[index, 0])
        data    = Image.open(imgpath).convert('RGB')

        #------------ Normalize bounding box labels
        if self.load_labels:
            bbox = np.array(self.csv.iloc[index, 1:5],  dtype=np.float32) # [xmin, xmax, ymin, ymax]
            bbox = bbox / np.array([data.width, data.width, data.height, data.height], dtype=np.float32)
            x, y, w, h = (bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2, bbox[1] - bbox[0], bbox[3] - bbox[2]
        else:
            x, y, w, h = 0, 0, 0, 0

        #------------ Resize & into Tensor
        data = TF.resize(data, self.shape, torchvision.transforms.InterpolationMode.BILINEAR)
        data = TF.to_tensor(data)

        #------------ Transformations
        if self.split == 'train':
            fl = ['rotate', 'flip', 'bc', 'gn']
            p  = np.random.rand(len(fl)) < 0.5

            for i, f in enumerate(fl):
                if f == 'rotate' and p[i]:
                    data, x, y, w, h = rotate(data, x, y, w, h)
                elif f == 'flip' and p[i]:
                    data, x, y = flip(data, x, y)
                elif f == 'bc' and p[i]:
                    data = brightness_contrast(data, alpha=(0.5, 2.0), beta=(-20, 20))
                elif f == 'gn' and p[i]:
                    data = gaussian_noise(data)

        if self.load_labels:
            # Bbox to tensor
            label = torch.tensor([x, y, w, h], dtype=torch.float32)
            return data, label
        else:
            return data

class Dataset_Park2019_KRN(Dataset):
    def __init__(self, root, csvfile, shape=(224,224), split='train', load_labels=True):
        # Read .csv file
        self.root       = root
        self.csv        = pd.read_csv(os.path.join(root, csvfile), header=None)
        self.shape      = tuple(shape)
        self.split      = split
        self.load_labels = load_labels

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        assert index < len(self), 'Index range error'

        #------------ Read Images
        imgpath = os.path.join(self.root, self.csv.iloc[index, 0])
        data    = Image.open(imgpath).convert('RGB')
        bbox    = np.array(self.csv.iloc[index, 1:5], dtype=np.float32)

        #------------ Transformations
        # (1) Random cropping given ground-truth bounding box coord. & resizing
        top, left, height, width = randomCrop(bbox, shape=data.size, split=self.split)
        data = TF.resized_crop(data, top, left, height, width, self.shape)
        data = TF.to_tensor(data)

        if self.load_labels:
            points2D = np.array(self.csv.iloc[index, 12:], dtype=np.float32) # [2N,]
            x = (points2D[[0,2,4,6,8,10,12,14,16,18,20]] - left) / width # x
            y = (points2D[[1,3,5,7,9,11,13,15,17,19,21]] - top) / height # y (Normalized)
        else:
            x, y = 0, 0

        # (2) Augmentation
        if self.split == 'train':
            fl = ['rotate', 'flip', 'bc', 'gn']
            p  = np.random.rand(len(fl)) < 0.5

            for i, f in enumerate(fl):
                if f == 'rotate' and p[i]:
                    data, x, y, _, _ = rotate(data, x, y, 0, 0)
                elif f == 'flip' and p[i]:
                    data, x, y = flip(data, x, y)
                elif f == 'bc' and p[i]:
                    data = brightness_contrast(data, alpha=(0.5, 2.0), beta=(-20, 20))
                elif f == 'gn' and p[i]:
                    data = gaussian_noise(data)

        if self.load_labels:
            points2D = np.reshape(np.transpose(np.vstack((x, y))), (22,))
            label = torch.from_numpy(points2D)
            if self.split == 'train':
                return data, label
            else:
                pose  = np.array(self.csv.iloc[index, 5:12], dtype=np.float32)
                pose  = torch.from_numpy(pose)# [quat + pos]
                roi   = torch.tensor([left, left+width, top, top+height], dtype=torch.float32)
                return data, label, pose, roi
        else:
            return data

