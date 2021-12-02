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

import os.path as osp
import logging
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

""" << csvfile >> is a path to a .csv file containing the following:
    Path to image:     imagepath,
    Tight RoI Coord.:  xmin, xmax, ymin, ymax     [pix]
    True pose:         q0, q1, q2, q3, t1, t2, t3 [-], [m]
    Keypoint Coord.:   kx1, ky1, ..., kx11, ky11  [pix]
"""
class Park2019KRNDataset(Dataset):
    def __init__(self, cfg, transforms=None, is_train=True, is_source=True, load_labels=True):
        self.is_train    = is_train
        self.load_labels = load_labels
        self.root        = osp.join(cfg.dataroot, cfg.dataname)
        self.num_keypts  = cfg.num_keypoints

        if is_train:
            if is_source:
                # Source domain - train
                csvfile = osp.join(self.root, cfg.train_domain,
                        'splits_'+cfg.model_name, cfg.train_csv)
                assert load_labels
            else:
                # Target domain - train for DANN
                # Load test CSV with is_train=True
                csvfile = osp.join(self.root, cfg.test_domain,
                        'splits_'+cfg.model_name, cfg.test_csv)
                assert not load_labels
        else:
            csvfile = osp.join(self.root, cfg.test_domain,
                    'splits_'+cfg.model_name, cfg.test_csv)

        logger.info('{} from {}'.format(
            'Training' if is_train else 'Testing', csvfile
        ))

        # Read CSV file
        self.csv = pd.read_csv(csvfile, header=None)

        # Image transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        assert index < len(self), 'Index range error'

        #------------ Read Images & Bbox
        imgpath = osp.join(self.root, self.csv.iloc[index, 0])
        data    = Image.open(imgpath).convert('RGB')
        bbox    = np.array(self.csv.iloc[index, 1:5], dtype=np.float32)

        #------------ Load keypoints
        if self.is_train and self.load_labels:
            keypts = np.array(self.csv.iloc[index, 12:], dtype=np.float32)  # [22,]
            keypts = np.transpose(np.reshape(keypts, (self.num_keypts, 2))) # [2 x 11]
        else:
            keypts = np.zeros((2, self.num_keypts))

        #------------ Data transform
        if self.transforms is not None:
            data, bbox, keypts = self.transforms(data, bbox, keypts)

        #------------ Return keypoints (train) or pose (test)
        if self.is_train:
            if self.load_labels:
                return data, keypts
            else:
                return data
        else:
            q_gt = np.array(self.csv.iloc[index, 5:9],  dtype=np.float32)
            t_gt = np.array(self.csv.iloc[index, 9:12], dtype=np.float32)
            return data, bbox, torch.from_numpy(q_gt), torch.from_numpy(t_gt)

