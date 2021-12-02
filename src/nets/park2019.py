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
from torchvision import models

class ConvDw(nn.Module):
    """ Serapable convolution module consisting of
        1. Depthwise convolution (3x3)
        2. pointwise convolution (1x1)

        Reference:
        Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
        Tobias Weyand, Marco Andreetto, and Hartwig Adam. MobileNets: Efficient
        convolutional neural neworks for mobile vision applications. CoRR, abs/1704.04861, 2017.

    """
    def __init__(self, inp, oup, stride):
        super(ConvDw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)

class RouterV2(nn.Module):
    def __init__(self, inp, oup, stride=2):
        super(RouterV2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.stride = stride

    def forward(self, x1, x2):
        # prune channel
        x2 = self.conv(x2)
        # reorg
        B, C, H, W = x2.size()
        s = self.stride
        x2 = x2.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x2 = x2.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x2 = x2.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        x2 = x2.view(B, s * s * C, H // s, W // s)
        return torch.cat((x2, x1), dim=1)

class RouterV3(nn.Module):
    def __init__(self, inp, oup, stride=1, mode='bilinear'):
        super(RouterV3, self).__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x1, x2):
        # prune channel
        x1 = self.conv(x1)
        # up
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode=self.mode, align_corners=True)
        return torch.cat((x1, x2), dim=1)

# --------------------------------------------------------------- #
# Keypoint Regression Network (KRN)
class KeypointRegressionNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointRegressionNet, self).__init__()
        self.nK = num_keypoints

        # Base feature extractor (ImageNet-pretrained)
        self.base = models.mobilenet_v2(pretrained=True)
        self.base = nn.ModuleList(list(self.base.features.children())[:-1])

        # Extra layers
        # feature_layer_name = [['DW', 'DW', 13, 'DW']]
        # feature_layer_size = [[1024, 1024, 64, 1024]]
        self.extras = nn.ModuleList([
            ConvDw(320, 1024, stride=1),
            ConvDw(1024, 1024, stride=1),
            RouterV2(96, 64), # self.base[13].depth = 96
            ConvDw(1024 + 64*4, 1024, stride=1)
        ])

        # Head layer(s), in case detection is made at multiple levels
        self.head = nn.ModuleList([nn.Conv2d(1024, 2*num_keypoints, kernel_size=7)])

        # Loss for keypoints
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x, y=None):
        B = x.shape[0]

        # Feed-forward (Base)
        for i, block in enumerate(self.base):
            x = block(x)
            if i == 13: temp = x

        # Rest
        for i, block in enumerate(self.extras):
            x = block(x, temp) if i == 2 else block(x)

        # Detection at the end for pose estimation
        x = self.head[0](x) # B x 2K x 1 x 1

        # Re-organize into (x,y) coord.
        x = x.view(B, 2*self.nK)
        xc = x[:,list(range(0,2*self.nK,2))]
        yc = x[:,list(range(1,2*self.nK,2))]

        if y is not None:
            # TRAINING
            txc = y[:,0]
            tyc = y[:,1]

            # Loss
            loss_x, loss_y = 0.0, 0.0
            for i in range(self.nK):
                loss_x = loss_x + self.loss(xc[:,i], txc[:,i])
                loss_y = loss_y + self.loss(yc[:,i], tyc[:,i])
            loss = loss_x + loss_y

            # Report summary
            sm = {'loss_x': float(loss_x),
                  'loss_y': float(loss_y)}

            return loss, sm
        else:
            # TESTING
            return xc.cpu(), yc.cpu()

