import torch
import torchvision.transforms.functional as TF

import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from PIL import Image
from copy import deepcopy

from datasets.park2019      import randomCrop
from utils.utils_eval       import *
from utils.utils            import load_surface_points, logging, report_progress, AverageMeter
from nets.park2019.park2019 import ObjectDetectionNet, KeypointRegressionNet

# from utils.utils_vis import *

def predict(odn_dict, krn_dict, cfg, cameraMatrix, distCoeffs=np.zeros((1,5)), device=torch.device('cpu'), krn_only=False):

    # ---------------- Camera & 3D keypoints
    corners3D = load_surface_points('utils/tangoPoints.mat')

    # ---------------- Load models & weights
    if not krn_only:
        # -> ODN
        objDetectNet = ObjectDetectionNet()
        load_dict    = torch.load(odn_dict, map_location='cpu')
        objDetectNet.load_state_dict(load_dict['weights'], strict=True)
        objDetectNet.to(device)
        objDetectNet.eval()

        # IMPORTANT!! Make a deep copy of the first model, otherwise loading state_dict of
        #             the second model modifies the state_dict of the first model
        objDetectNet = deepcopy(objDetectNet)

    # -> KRN
    kpRegressNet = KeypointRegressionNet(11)
    load_dict    = torch.load(krn_dict, map_location='cpu')
    kpRegressNet.load_state_dict(load_dict['weights'], strict=True)
    kpRegressNet.to(device)
    kpRegressNet.eval()

    # ---------------- Read .CSV containing data info.
    csv = pd.read_csv(os.path.join(cfg.dataroot, cfg.val_csv), header=None)

    # ---------------- Prediction
    elapse_t = AverageMeter('ms')
    errs_iou = AverageMeter('-')
    errs_tra = AverageMeter('m')
    errs_rot = AverageMeter('deg')
    errs_spd = AverageMeter('-')
    errs_add = AverageMeter('-')

    ei = []
    et = []
    er = []
    es = []

    for i in range(len(csv)):

        # Read original image
        imgpath = os.path.join(cfg.dataroot, csv.iloc[i, 0])
        dataOrg = Image.open(imgpath).convert('RGB')
        wOrg, hOrg = dataOrg.width, dataOrg.height

        ts = time.time()

        # ----------------------------
        # (1) Object Detection CNN
        if not krn_only:
            # Processing for object detection (no normalization at this step)
            data = TF.resize(dataOrg, (416,416))
            data = TF.to_tensor(data)
            data = data.unsqueeze(0).to(device)

            with torch.no_grad():
                out = objDetectNet(data)

            # Get highest-confidence detection
            # bbox format here: [xc, yc, width, height] (normalized)
            bbox, _ = detect(out)

            # Box to original image size
            orgImgSize = torch.Tensor([wOrg, hOrg, wOrg, hOrg])
            bbox = bbox[0].cpu() * orgImgSize

            # Box to [xmin, xmax, ymin, ymax] format
            bbox = torch.tensor([bbox[0]-bbox[2]/2.0, bbox[0]+bbox[2]/2.0,
                                bbox[1]-bbox[3]/2.0, bbox[1]+bbox[3]/2.0],
                                dtype=torch.float32)

        # Ground-truth bbox
        bbox_gt = np.array(csv.iloc[i, 1:5], dtype=np.float32) # [xmin, xmax, ymin, ymax]
        bbox_gt = torch.as_tensor(bbox_gt)

        # plot_2D_bbox(F.to_tensor(dataOrg), bbox)

        # ----------------------------
        # (2) Pose Estimation CNN
        # First, crop & resize the image according to predicted bbox
        # Then convert to tensor, normalize according to SPEED stats.
        if krn_only:
            top, left, height, width = randomCrop(bbox_gt, (wOrg, hOrg), 'test')
        else:
            top, left, height, width = randomCrop(bbox, (wOrg, hOrg), 'test')
        data = TF.resized_crop(dataOrg, top, left, height, width, (224,224))
        data = TF.to_tensor(data)
        data = data.unsqueeze(0).to(device)

        # RoI used for cropping [xmin, xmax, ymin, ymax]
        roi  = np.array([left, left+width, top, top+height])

        # plot_2D_bbox(F.to_tensor(dataOrg), roi)

        with torch.no_grad():
            x_pr, y_pr = kpRegressNet(data) # [1 x 11], torch.Tensor (cpu)

        # scatter_keypoints(data[0], x_pr[0], y_pr[0], normalized=True)

        # Post-processing
        R_pr, t_pr = predictPose(x_pr.cpu(), y_pr.cpu(), roi, corners3D, cameraMatrix, distCoeffs)
        q_pr = R.from_matrix(R_pr).as_quat() # [qvec, qcos]

        tf = time.time()

        # ----------------------------
        # (3) Performance evaluation
        # Read ground-truth labels
        q_gt = np.array(csv.iloc[i, 5:9], dtype=np.float32) # [qcos, qvec]
        q_gt = R.from_quat(q_gt[[1,2,3,0]]).as_quat()
        t_gt = np.array(csv.iloc[i, 9:12],   dtype=np.float32)

        # IoU error
        # - bbox input rearranged to [xmin, ymin, xmax, ymax]
        if not krn_only:
            err_iou = bbox_iou(bbox[[0,2,1,3]], bbox_gt[[0,2,1,3]], x1y1x2y2=True).numpy()
            errs_iou.update(err_iou, 1)
            ei.append(err_iou)

        # Translation error
        err_tra = error_translation(t_pr, t_gt)
        errs_tra.update(err_tra, 1)
        et.append(err_tra)

        # Rotation error
        err_rot = error_orientation(q_pr, q_gt)
        errs_rot.update(err_rot, 1)
        er.append(err_rot)

        # SPEED score
        speed   = speed_score(t_pr, q_pr, t_gt, q_gt)
        errs_spd.update(speed, 1)
        es.append(speed)

        elapse_t.update((tf-ts)*1000, 1)

        report_progress(epoch=1, lr=np.NAN, epoch_iter=i+1, epoch_size=len(csv),
                        time=elapse_t, mode='val', IoU=errs_iou, Er=errs_rot, Et=errs_tra, speed=errs_spd)

    logging('IoU:   %.3f [-]' % (errs_iou.avg))
    logging('eT:    %.3f [m]' % (errs_tra.avg))
    logging('eR:    %.3f [deg]' % (errs_rot.avg))
    logging('SPEED: %.3f [-]' % (errs_spd.avg))

    # out_csv = pd.DataFrame([ei, et, er, es])
    # out_csv.to_csv('/media/shared/Jeff/SLAB/Dataset/asc2021/synth_out.csv')

    return errs_iou, errs_tra, errs_rot, errs_spd