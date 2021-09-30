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
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import logging

logger = logging.getLogger(__name__)

def softmax_cross_entropy_with_logits(logits, target, reduction='mean'):
    """ Implementation of tensorflow's function of the same name
        - logits [B x C]
        - target [B x C]
    """
    loss = -torch.sum(target.detach() * F.log_softmax(logits, dim=1), dim=1) # [B,]
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class SpacecraftPoseNet(nn.Module):

    def __init__(self, num_classes, keep_prob=0.5, pretrain=True):
        super(SpacecraftPoseNet, self).__init__()

        self.num_classes  = num_classes
        self.regress_size = self.num_classes
        self.keep_prob    = keep_prob

        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0) # Valid padding
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.norm1 = nn.LocalResponseNorm(2, alpha=2e-5, beta=0.75, k=1.0)

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=0)
        self.norm2 = nn.LocalResponseNorm(2, alpha=2e-5, beta=0.75, k=1.0)

        # 3rd Layer: Conv (w ReLU)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2)
        self.pool5 = nn.MaxPool2d(3, stride=2, padding=0)

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        self.fc6 = nn.Linear(9216, 4096)
        self.dropout6 = nn.Dropout(p=self.keep_prob, inplace=True)

        # 7th Layer: FC (w ReLu) -> Dropout
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout(p=self.keep_prob, inplace=True)

        # 8th Layer: FC (no ReLu), return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = nn.Linear(4096, self.num_classes)

        # 9th Layer: Flatten -> FC (w ReLu) -> Dropout
        self.fc9 = nn.Linear(9216, 4096)
        self.dropout9 = nn.Dropout(p=self.keep_prob, inplace=True)

        # 10th Layer: FC (w ReLu) -> Dropout
        self.fc10 = nn.Linear(4096, 4096)
        self.dropout10 = nn.Dropout(p=self.keep_prob, inplace=True)

        # 11th Layer: FC (no ReLu), return unscaled activations
        self.fc11 = nn.Linear(4096, self.regress_size)

        if pretrain:
            self.load_weights('checkpoints/pretrained/bvlc_alexnet.npy')

    def load_weights(self, weight_path):

        logger.info('   - Loading pretrained weights from {}'.format(
            weight_path
        ))

        # Load weight dict
        weights_dict = np.load(weight_path, allow_pickle=True, encoding='bytes').item()

        # Only load weights for the first 5 layers
        with torch.no_grad():
            for name in weights_dict:
                if name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                    for data in weights_dict[name]:
                        if len(data.shape) == 4:
                            # [H, W, Cin, Cout] -> [Cout, Cin, H, W]
                            data = np.transpose(data, (3,2,0,1))
                            getattr(self, name).weight.copy_(torch.from_numpy(data).float())
                        else:
                            getattr(self, name).bias.copy_(torch.from_numpy(data).float())

    def forward(self, x, y=None):
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = torch.flatten(x, start_dim=1)

        # Attitude classification
        c = self.dropout6(F.relu(self.fc6(x)))
        c = self.dropout7(F.relu(self.fc7(c)))
        c = self.fc8(c)

        # Attitude regression
        r = self.dropout9(F.relu(self.fc9(x)))
        r = self.dropout10(F.relu(self.fc10(r)))
        r = self.fc11(r)

        return c, r

if __name__=='__main__':
    # Test

    x = torch.rand(1, 3, 227, 227)
    model = SpacecraftPoseNet(1000)

    c, r = model(x)
    print(c.shape, r.shape)