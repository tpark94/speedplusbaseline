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

import logging
import torch

from .park2019 import KeypointRegressionNet
from .spn      import SpacecraftPoseNet
from .revgrad  import RevGrad

from src.utils.utils import num_total_parameters, num_trainable_parameters

logger = logging.getLogger(__name__)

def get_model(cfg):
    assert cfg.model_name=='krn' or cfg.model_name=='spn', \
        'Model name must be either krn or spn'

    if not cfg.dann:
        if cfg.model_name == 'krn':
            model = KeypointRegressionNet(cfg.num_keypoints)
            logger.info('KRN created')
        elif cfg.model_name == 'spn':
            model = SpacecraftPoseNet(cfg.num_classes, pretrain=True)
            logger.info('SPN created')
    else:
        # DANN
        model = RevGrad(cfg.num_keypoints)
        logger.info('RevGrad created with {}'.format(cfg.model_name))

    logger.info('   - Number of total parameters:     {:,}'.format(num_total_parameters(model)))
    logger.info('   - Number of trainable parameters: {:,}'.format(num_trainable_parameters(model)))

    return model

def get_optimizer(cfg, model):
    param = filter(lambda p:p.requires_grad, model.parameters())

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param,
                        lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param,
                        lr=cfg.lr, alpha=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(param,
                        lr=cfg.lr, betas=(cfg.momentum,0.999), weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param,
                        lr=cfg.lr, betas=(cfg.momentum,0.999), weight_decay=cfg.weight_decay)

    logger.info('Optimizer created: {}'.format(cfg.optimizer))

    return optimizer