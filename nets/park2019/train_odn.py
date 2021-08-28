import torch
import time
import pandas as pd

from nets.park2019.park2019 import ObjectDetectionNet
from datasets.park2019      import Dataset_Park2019_ODN
from utils.utils_eval       import detect, bbox_iou
from utils.utils            import *

from utils.utils_vis import *

def train_odn_single_epoch(epoch, lr, model, dataloader, optimizer, device):

    model.train()

    # ---------------- Prediction
    elapse_time   = AverageMeter('ms')
    loss_giou_rec = AverageMeter('-')
    loss_conf_rec = AverageMeter('-')

    for idx, (data, target) in enumerate(dataloader):
        B  = data.size(0)
        ts = time.time()

        # plot_2D_bbox(data[0], target[0] * 416)

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
        loss_giou_rec.update(sm['loss_giou'], B)
        loss_conf_rec.update(sm['loss_conf'], B)
        elapse_time.update((tf-ts)*1000, B)
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(dataloader),
                        time=elapse_time, loss_giou=loss_giou_rec, loss_conf=loss_conf_rec)

def val_odn(epoch, lr, model, dataloader, device, writefn=None):

    model.eval()

    # ---------------- Prediction
    elapse_time = AverageMeter('ms')
    errs_iou    = AverageMeter('-')
    bbox_results = []

    for idx, (data, target) in enumerate(dataloader):
        B  = data.size(0)
        ts = time.time()

        # plot_2D_bbox(data[0], target[0] * 416)

        # To device
        data = data.to(device)

        # Feed-forward
        with torch.no_grad():
            out = model(data)

        # Get highest-confidence detection
        # bbox format here: [xc, yc, width, height] (normalized)
        bbox, _ = detect(out)

        x, y, w, h = float(bbox[0][0]), float(bbox[0][1]), float(bbox[0][2]), float(bbox[0][3])
        bbox_results.append([(x-w/2)*1920, (x+w/2)*1920, (y-h/2)*1200, (y+h/2)*1200])

        # IoU
        iou = bbox_iou(bbox[0], target[0], x1y1x2y2=False, GIoU=False).numpy()

        tf = time.time()

        # Update
        errs_iou.update(iou.item(), B)
        elapse_time.update((tf-ts)*1000, B)
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(dataloader), mode='val',
                        time=elapse_time, IoU=errs_iou)

    if writefn is not None:
        df = pd.DataFrame(bbox_results)
        df.to_csv(writefn, header=False, index=False)

def train_odn(cfg, device, init_epoch=None):

    # Create ODN
    objDetectNet = ObjectDetectionNet()
    logging('ODN created. Number of parameters: {}'.format(num_total_parameters(objDetectNet)))

    # # Multiple GPUs?
    # if torch.cuda.device_count() > 1:
    #     logging('   - Using {} GPUs'.format(torch.cuda.device_count()))
    #     objDetectNet = torch.nn.DataParallel(objDetectNet)

    # ODN Dataloaders
    train_dataset = Dataset_Park2019_ODN(cfg.dataroot, cfg.train_csv, shape=cfg.input_shape_odn, split='train')
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataset   = Dataset_Park2019_ODN(cfg.dataroot, cfg.val_csv, shape=cfg.input_shape_odn, split='test')
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(objDetectNet.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha)

    # Load weights if specified
    if init_epoch is not None:
        load_checkpoint(init_epoch, cfg, objDetectNet, optimizer, device)
    else:
        init_epoch = 0

    # Model to CUDA
    objDetectNet.to(device)

    for epoch in range(init_epoch+1, cfg.max_epochs+1):
        # Current learning rate
        for pg in optimizer.param_groups:
            cur_lr = pg['lr']

        # Train ODN for single epoch
        train_odn_single_epoch(epoch, cur_lr, objDetectNet, train_loader, optimizer, device)

        # Update LR
        scheduler.step()

        # Save weights
        if (epoch % cfg.save_epoch == 0):
            save_checkpoint(epoch, cfg, objDetectNet, optimizer)

        # Validate ODN
        if (epoch % cfg.test_epoch == 0):
            val_odn(epoch, cur_lr, objDetectNet, val_loader, device)

    # Save to 'final.pth'
    save_checkpoint(cfg.max_epochs, cfg, objDetectNet, optimizer, last=True)