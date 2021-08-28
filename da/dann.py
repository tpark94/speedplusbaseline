import torch
import torch.nn as nn

import os
import numpy as np
import time
import json

from nets.park2019.park2019  import KeypointRegressionNet, ObjectDetectionNet
from datasets.park2019       import Dataset_Park2019_KRN, Dataset_Park2019_ODN
from nets.park2019.train_odn import val_odn
from nets.park2019.train_krn import val_krn

from utils.utils import AverageMeter, report_progress, load_checkpoint, save_checkpoint

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)

    This class is forked from the following repo:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class DANN_Park2019(nn.Module):
    def __init__(self, model):
        super(DANN_Park2019, self).__init__()

        # Net (Feature extractor + pose estimation network)
        if model == 'odn':
            self.net = ObjectDetectionNet()
        elif model == 'krn':
            self.net = KeypointRegressionNet(11)
        else:
            AssertionError('Model must be odn or krn.')

        # Forward hook after feature extractor
        def return_output(module, input, output):
            self.feature = output
        self.net.base[-1].register_forward_hook(return_output)

        # Domain classifier - follow the MobileNetv2 architecture
        # Output of feature extractor: [B x 320 x 7 x 7]
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(320, 1280, 1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1280, 1280, 1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7),
            nn.Conv2d(1280, 1, 1)
        )

    def forward(self, x, y=None, alpha=None):
        if y is not None:
            # Training: {loss, sm}
            out1 = self.net(x, y)
        else:
            # Test: {xc, yc} or bbox
            out1 = self.net(x)

        if alpha is not None:
            # Domain adaptation - return domain output
            out2 = GradientReversalFunction.apply(self.feature, alpha)
            out2 = self.domain_classifier(out2)
            return out1, out2.squeeze()
        else:
            return out1

def train_dann_single_epoch(epoch, lr, model, dataloader_source, dataloader_target, optimizer, max_epochs, device):

    model.train()

    # ---------------- Prediction
    elapse_time   = AverageMeter('ms')
    loss_pose_rec = AverageMeter('-')
    loss_ds_rec   = AverageMeter('-')
    loss_dt_rec   = AverageMeter('-')

    batches   = zip(dataloader_source, dataloader_target)
    n_batches = min(len(dataloader_source), len(dataloader_target))

    for idx, ((source, label), target) in enumerate(batches):
        B  = source.size(0)
        ts = time.time()

        # To device
        source = source.to(device)
        label  = label.to(device)
        target = target.to(device)

        # Zero the optimizer gradients before backprop
        optimizer.zero_grad(set_to_none=True)

        # Domain classifier loss factor
        p = float(idx + epoch * n_batches) / max_epochs / n_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Feed-forward (source)
        loss_source, domain_source_pred = model(source, y=label, alpha=alpha)
        loss_pose_source, sm = loss_source

        # Domain loss (1: source, 0: target)
        loss_domain_source = nn.functional.binary_cross_entropy_with_logits(
                                domain_source_pred, torch.ones(B).to(device)
        )

        # Feed-forward (target)
        _, domain_target_pred = model(target, alpha=alpha)
        loss_domain_target = nn.functional.binary_cross_entropy_with_logits(
                                domain_target_pred, torch.zeros(B).to(device)
        )

        # Backprop
        loss = loss_pose_source + loss_domain_source + loss_domain_target
        loss.backward()

        # Update Weights
        optimizer.step()
        tf = time.time()

        # Update
        loss_pose_rec.update(float(loss_pose_source), B)
        loss_ds_rec.update(float(loss_domain_source), B)
        loss_dt_rec.update(float(loss_domain_target), B)
        elapse_time.update((tf-ts)*1000, B)
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=n_batches,
                        time=elapse_time, loss_source=loss_pose_rec, domain_s=loss_ds_rec, domain_t=loss_dt_rec)

def train_dann_odn(cfg, device, init_epoch=None):

    # Create KRN
    model = DANN_Park2019('odn')

    # KRN Dataloaders
    source_train_dataset = Dataset_Park2019_ODN(cfg.dataroot, cfg.source_train_csv, shape=cfg.input_shape_odn, split='train', load_labels=True)
    source_train_loader  = torch.utils.data.DataLoader(source_train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    target_train_dataset = Dataset_Park2019_ODN(cfg.dataroot, cfg.target_train_csv, shape=cfg.input_shape_odn, split='train', load_labels=False)
    target_train_loader  = torch.utils.data.DataLoader(target_train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    target_val_dataset   = Dataset_Park2019_ODN(cfg.dataroot, cfg.target_val_csv, shape=cfg.input_shape_odn, split='test', load_labels=True)
    target_val_loader    = torch.utils.data.DataLoader(target_val_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Optimizer - AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha)

    # Load weights if specified
    if init_epoch is not None:
        load_checkpoint(init_epoch, cfg, model, optimizer, device)
    else:
        # initialize_conv_weights(model)
        init_epoch = 0

    # Model to CUDA
    model.to(device)

    for epoch in range(init_epoch+1, cfg.max_epochs+1):
        # Current learning rate
        for pg in optimizer.param_groups:
            cur_lr = pg['lr']

        # Train KRN for single epoch
        train_dann_single_epoch(epoch, cur_lr, model, source_train_loader, target_train_loader, optimizer, cfg.max_epochs, device)

        # Update LR
        scheduler.step()

        # Save weights
        if (epoch % cfg.save_epoch == 0):
            save_checkpoint(epoch, cfg, model, optimizer)

        # Validate KRN
        if (epoch % cfg.test_epoch == 0):
            val_odn(epoch, cur_lr, model, target_val_loader, device)

    # Final things to do
    save_checkpoint(cfg.max_epochs, cfg, model, optimizer, last=True)
    val_odn(cfg.max_epochs, 0, model, target_val_loader, device,
            writefn=os.path.join(cfg.projroot, cfg.savedir, 'dann_odn_out.csv'))

def train_dann_krn(cfg, cameraJson, corners3D, device, init_epoch=None):

    with open(cameraJson) as f:
        cam = json.load(f)
    cameraMatrix = np.array(cam['K'], dtype=np.float32)
    distCoeffs   = np.array(cam['dist_coeffs'], dtype=np.float32)

    # Create KRN
    model = DANN_Park2019('krn')

    # KRN Dataloaders
    source_train_dataset = Dataset_Park2019_KRN(cfg.source_train_csv, shape=cfg.input_shape_krn, split='train', load_labels=True)
    source_train_loader  = torch.utils.data.DataLoader(source_train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    target_train_dataset = Dataset_Park2019_KRN(cfg.target_train_csv, shape=cfg.input_shape_krn, split='train', load_labels=False)
    target_train_loader  = torch.utils.data.DataLoader(target_train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    target_val_dataset   = Dataset_Park2019_KRN(cfg.target_val_csv, shape=cfg.input_shape_krn, split='test', load_labels=True)
    target_val_loader    = torch.utils.data.DataLoader(target_val_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Optimizer - AdamW
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha)

    # Load weights if specified
    if init_epoch is not None:
        load_checkpoint(init_epoch, cfg, model, optimizer, device)
    else:
        # initialize_conv_weights(model)
        init_epoch = 0

    # Model to CUDA
    model.to(device)

    for epoch in range(init_epoch+1, cfg.max_epochs+1):
        # Current learning rate
        for pg in optimizer.param_groups:
            cur_lr = pg['lr']

        # Train KRN for single epoch
        train_dann_single_epoch(epoch, cur_lr, model, source_train_loader, target_train_loader, optimizer, cfg.max_epochs, device)

        # Save weights
        if (epoch % cfg.save_epoch == 0):
            save_checkpoint(epoch, cfg, model, optimizer)

        # Validate KRN
        if (epoch % cfg.test_epoch == 0):
            val_krn(epoch, cur_lr, model, target_val_loader, corners3D, cameraMatrix, distCoeffs, device)

        # Update LR
        scheduler.step()

    # Final things to do
    save_checkpoint(cfg.max_epochs, cfg, model, optimizer, last=True)

if __name__=='__main__':
    net = DANN_Park2019('krn')
    x   = torch.rand(1,3,224,224)
    net(x)