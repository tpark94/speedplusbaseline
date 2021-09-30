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
import cv2

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SPNDataset(Dataset):
    def __init__(self, cfg, transforms=None, is_train=True, is_source=True):
        self.is_train      = is_train
        self.root          = cfg.dataroot
        self.num_classes   = cfg.num_classes
        self.num_neighbors = cfg.num_neighbors

        if is_train:
            if is_source:
                # Source domain - train
                csvfile = osp.join(self.root, cfg.train_domain,
                        'splits_'+cfg.model_name, cfg.train_csv)
            else:
                # Target domain - train for DANN
                # Load test CSV with is_train=True
                csvfile = osp.join(self.root, cfg.test_domain,
                        'splits_'+cfg.model_name, cfg.test_csv)
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

        #------------ Read Image & Bbox
        imgpath = osp.join(self.root, self.csv.iloc[index, 0])
        data    = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        data    = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        bbox    = np.array(self.csv.iloc[index, 1:5], dtype=np.float32)

        #------------ Data transform
        if self.transforms is not None:
            data, bbox = self.transforms(data, bbox)

        #------------ Return classes & weights (train) or pose (test)
        if self.is_train:
            attClasses = np.array(self.csv.iloc[index, 12:12+self.num_neighbors], dtype=np.int32)
            attWeights = np.array(self.csv.iloc[index, 12+self.num_neighbors:],   dtype=np.float32)

            # Classes into n-hot vector
            yClasses = np.zeros(self.num_classes, dtype=np.float32)
            yClasses[attClasses] = 1. / self.num_neighbors

            # Weights into n-hot vector as well
            yWeights = np.zeros(self.num_classes, dtype=np.float32)
            yWeights[attClasses] = attWeights

            return data, yClasses, yWeights
        else:
            q_gt = np.array(self.csv.iloc[index, 5:9],  dtype=np.float32)
            t_gt = np.array(self.csv.iloc[index, 9:12], dtype=np.float32)
            return data, bbox, q_gt, t_gt
