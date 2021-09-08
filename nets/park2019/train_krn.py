import torch

import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from nets.park2019.park2019 import KeypointRegressionNet
from datasets.park2019      import Dataset_Park2019_KRN
from utils.utils_eval       import *
from utils.utils            import *

# from utils.utils_vis import *

def train_krn_single_epoch(epoch, lr, model, dataloader, optimizer, device):

    model.train()

    # ---------------- Prediction
    elapse_time = AverageMeter('ms')
    loss_x_rec  = AverageMeter('-')
    loss_y_rec  = AverageMeter('-')

    for idx, (data, target) in enumerate(dataloader):
        B  = data.size(0)
        ts = time.time()

        # x = target[0, [0,2,4,6,8,10,12,14,16,18,20]]
        # y = target[0, [1,3,5,7,9,11,13,15,17,19,21]]
        # scatter_keypoints(data[0], x, y, normalized=True)

        # To device
        data   = data.to(device)
        target = target.to(device)

        # Zero the optimizer gradients before backprop
        optimizer.zero_grad(set_to_none=True)

        # Feed-forward
        loss, sm = model(data, target)

        # Backprop
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update Weights
        optimizer.step()
        tf = time.time()

        # Update
        loss_x_rec.update(sm['loss_x'], B)
        loss_y_rec.update(sm['loss_y'], B)
        elapse_time.update((tf-ts)*1000, B)
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(dataloader),
                        time=elapse_time, loss_x=loss_x_rec, loss_y=loss_y_rec)

def val_krn(epoch, lr, model, dataloader, corners3D, cameraMatrix, distCoeffs, device):

    model.eval()

    # ---------------- Prediction
    elapse_time = AverageMeter('ms')
    errs_t  = AverageMeter('m')
    errs_q  = AverageMeter('deg')
    speeds  = AverageMeter('-')

    for idx, (data, target, pose, roi) in enumerate(dataloader):
        B  = data.size(0)
        ts = time.time()

        # x = target[0, [0,2,4,6,8,10,12,14,16,18,20]]
        # y = target[0, [1,3,5,7,9,11,13,15,17,19,21]]
        # scatter_keypoints(data[0], x, y, normalized=True)

        # To device
        data   = data.to(device)
        target = target.to(device)

        # Feed-forward
        with torch.no_grad():
            x_pr, y_pr = model(data)

        # Pose processing
        R_pr, t_pr = predictPose(x_pr, y_pr, roi[0].numpy(), corners3D, cameraMatrix, distCoeffs)
        q_pr = R.from_matrix(R_pr).as_quat() # [qvec, qcos]

        # Read ground-truth labels
        q_gt = np.array(pose[0,:4], dtype=np.float32) # [qcos, qvec]
        t_gt = np.array(pose[0,4:], dtype=np.float32)

        # Translation error
        err_t = error_translation(t_pr, t_gt)
        errs_t.update(err_t, 1)

        # Rotation error
        q_gt  = R.from_quat(q_gt[[1,2,3,0]]).as_quat()
        err_q = error_orientation(q_pr, q_gt) # [deg]
        errs_q.update(err_q, 1)

        # SPEED score
        speed = speed_score(t_pr, q_pr, t_gt, q_gt)
        speeds.update(speed, 1)

        tf = time.time()

        # Update
        elapse_time.update((tf-ts)*1000, B)
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(dataloader),
                        time=elapse_time, mode='val', eT=errs_t, eR=errs_q, speed=speeds)

def train_krn(cfg, cameraJson, corners3D, device, init_epoch=None):

    with open(cameraJson, 'r') as f:
        cam = json.load(f)
    cameraMatrix = np.array(cam['cameraMatrix'], dtype=np.float32)
    distCoeffs   = np.array(cam['distCoeffs'], dtype=np.float32)

    # Create KRN
    kpRegressNet = KeypointRegressionNet(11)
    logging('KRN created. Number of parameters: {}'.format(num_total_parameters(kpRegressNet)))

    # # Multiple GPUs?
    # if torch.cuda.device_count() > 1:
    #     logging('   - Using {} GPUs'.format(torch.cuda.device_count()))
    #     kpRegressNet = torch.nn.DataParallel(kpRegressNet)

    # KRN Dataloaders
    train_dataset = Dataset_Park2019_KRN(cfg.dataroot, cfg.train_csv, shape=cfg.input_shape_krn, split='train')
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataset   = Dataset_Park2019_KRN(cfg.dataroot, cfg.val_csv, shape=cfg.input_shape_krn, split='test')
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(kpRegressNet.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha)

    # Load weights if specified
    if init_epoch is not None:
        load_checkpoint(init_epoch, cfg, kpRegressNet, optimizer, device)
    else:
        init_epoch = 0

    # Model to CUDA
    kpRegressNet.to(device)

    for epoch in range(init_epoch+1, cfg.max_epochs+1):
        # Current learning rate
        for pg in optimizer.param_groups:
            cur_lr = pg['lr']

        # Train KRN for single epoch
        train_krn_single_epoch(epoch, cur_lr, kpRegressNet, train_loader, optimizer, device)

        # Update LR
        scheduler.step()

        # Save weights
        if (epoch % cfg.save_epoch == 0):
            save_checkpoint(epoch, cfg, kpRegressNet, optimizer)

        # Validate KRN
        if (epoch % cfg.test_epoch == 0):
            val_krn(epoch, cur_lr, kpRegressNet, val_loader, corners3D, cameraMatrix, distCoeffs, device)

    # Save to 'final.pth'
    save_checkpoint(cfg.max_epochs, cfg, kpRegressNet, optimizer, last=True)

