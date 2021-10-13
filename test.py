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
import os
import os.path as osp
import logging
from scipy.io import loadmat

from cfgs.config import cfg
from src.nets.build     import get_model
from src.datasets.build import make_dataloader
from src.core.inference import valid_krn, valid_spn
from src.utils.utils    import setup_logger, load_tango_3d_keypoints, load_camera_intrinsics

logger = logging.getLogger(__name__)

def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Logger & Output directory
    setup_logger('test')
    if not osp.exists(cfg.logdir): os.makedirs(cfg.logdir)

    # Pose estimation CNN
    model = get_model(cfg)

    # Dataloader
    test_loader = make_dataloader(cfg, is_train=False, is_source=False)

    # Load checkpoint
    if osp.exists(cfg.pretrained):
        model.load_state_dict(torch.load(cfg.pretrained, map_location='cpu'), strict=True)
        logger.info('Model loaded from {}'.format(cfg.pretrained))

    model = model.to(device)

    # PnP-related items
    corners3D = load_tango_3d_keypoints(cfg.keypts_3d_model)
    cameraMatrix, distCoeffs = load_camera_intrinsics(
                osp.join(cfg.dataroot, cfg.dataname, 'camera.json'))
    attClasses = loadmat('src/utils/attitudeClasses.mat')['qClass'] # [Nclasses x 4]
    assert attClasses.shape[0] == cfg.num_classes, 'Number of classes not matching.'

    # Main loop
    eR, eT, speed_raw, speed_mod = eval('valid_'+cfg.model_name)(
        0, cfg, model, test_loader,
        cameraMatrix, distCoeffs, corners3D, None, device, attClasses)

    # # ! Temporary: Write the attitude predictions to a file
    # writefn = osp.join(cfg.logdir, 'attitudes_{}.txt'.format(cfg.test_domain))
    # if cfg.model_name == 'spn':
    #     q_all = eT

    #     with open(writefn, 'w') as f:
    #         for i in range(len(q_all)):
    #             f.write(' '.join(str(item) for item in q_all[i]))
    #             f.write('\n')

if __name__=='__main__':
    main()