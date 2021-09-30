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

from .park2019 import KeypointRegressionNet
from .spn      import SpacecraftPoseNet

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class RevGrad(nn.Module):
    def __init__(self, cfg):
        super(RevGrad, self).__init__()

        # DANN only applied to KRN
        assert cfg.model_name == 'krn'

        # Net (Feature extractor + pose estimation network)
        self.net = KeypointRegressionNet(cfg.num_keypoints)

        # Forward hook after feature extractor
        self.feature = torch.empty(0)
        def return_output(module, input, output):
            self.feature = output

        # Attach hook
        self.net.base[-1].register_forward_hook(return_output)

        # Domain classifier - follow the MobileNetv2 architecture
        # Output of feature extractor: [B x 320 x 7 x 7]
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(320, 1280, 1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1280, 2, 7, bias=True),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x, y=None, alpha=None):
        if y is not None:
            # Training: {loss, sm} - KRN
            #           {classes, weights} : SPN
            out1 = self.net(x, y)
        else:
            # Test: {xc, yc} or bbox - KRN
            out1 = self.net(x)

        if alpha is not None:
            # Domain adaptation - return domain output
            out2 = GradientReversalFunction.apply(self.feature, alpha)
            out2 = self.domain_classifier(out2)
            return out1, out2.squeeze()
        else:
            return out1