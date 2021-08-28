import torch
import torch.nn.functional as F
import numpy as np
import cv2

# -----------------------------------------------
# For YOLO-style post-processing (object detection)
def get_grids(nG):
    # Offsets
    grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG])
    grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG])
    return grid_x, grid_y

def build_targets_giou(pred, targ, best_ns, nB, nA, nG):
    device = targ.device

    # Placeholders
    obj_mask    = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)
    noobj_mask  = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)
    tconf       = torch.zeros(nB, nA, nG, nG, dtype=torch.float32, device=device)
    loss_giou   = torch.zeros(nB, nA, nG, nG, dtype=torch.float32, device=device)

    # Label as numpy array
    targ = targ.detach()

    for b in range(nB):
        # Count in loss only if best anchor matching
        if best_ns[b] is not None:
            best_n = best_ns[b]

            # In terms of grid cell numbers
            gx, gy, _, _ = targ[b]*nG

            # Cell location of the ground-truth bounding box
            gi, gj = int(gx), int(gy)

            # Mask - confidence
            # Set the correct cell's confidence level to prescribed object_scale
            tconf[b, best_n, gj, gi]      = 1.0
            obj_mask[b, best_n, gj, gi]   = 1
            noobj_mask[b, best_n, gj, gi] = 0

            # Generalized IoU loss
            giou = bbox_iou(pred[b, best_n, gj, gi]*nG,
                        targ[b]*nG, x1y1x2y2=False, GIoU=True)
            loss_giou[b, best_n, gj, gi] = 1.0 - giou

    return obj_mask, noobj_mask, loss_giou, tconf

def giouloss(pred, targ, anchors=None, iou_scale=1, obj_scale=5, noobj_scale=0.1):
    """ Implements Generalized IoU as a loss function

        Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
        Rezatofighi et al.

        pred : list of (nB x nA x nG x nG x 5) tensors, in number of grids
    """
    # Device
    device = targ.device

    # Sizes
    nB  = targ.size(0)
    nGs = [p.size(2) for p in pred]

    # For best anchor location, which defines objectness mask
    best_anchor = []

    # First, find out where the best anchor match is
    # ACROSS all 9 anchors
    for b in range(nB):
        ious = []
        for idx, anchor in enumerate(anchors):
            nA = len(anchor)
            nG = nGs[idx]

            # Ground-truth box
            gwh = torch.squeeze(targ[b, 2:] * nG)

            # Anchor scaled to grids
            anchors_scaled = torch.tensor([ (aw*nG, ah*nG) for aw, ah in anchor ], dtype=torch.float32, device=device)

            # Compare iou's
            ious.append( torch.stack([bbox_wh_iou(anchor_scaled, gwh)
                                    for anchor_scaled in anchors_scaled]) )

        # For each bach, identify best batch and its index
        _, best_n = torch.cat(ious).max(0)
        best_anchor.append(int(best_n))

    loss_giou  = 0
    loss_conf  = 0

    # Using best anchor position for each batch, construct target
    idx = 0
    for p, anchor in zip(pred, anchors):
        # Now, build the target
        nA = len(anchor)
        nG = p.size(2)
        best_n = [a-idx if a >= idx and a < idx + nA else None for a in best_anchor]

        # Build target
        box  = p[..., :4]
        conf = p[..., 4]  # logits
        obj_mask, noobj_mask, loss_giou_i, tconf = build_targets_giou(box, targ, best_n, nB, nA, nG)

        # Confidence
        if torch.any(obj_mask):
            loss_conf_obj   = F.binary_cross_entropy_with_logits(conf[obj_mask], tconf[obj_mask], reduction='mean')
            loss_conf_noobj = F.binary_cross_entropy_with_logits(conf[noobj_mask], tconf[noobj_mask], reduction='mean')

            # Sum up
            loss_giou += iou_scale * torch.mean(loss_giou_i[obj_mask])
            loss_conf += obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj

        # End, update idx
        idx += nA

    loss_total = loss_giou + loss_conf

    return loss_total, loss_giou, loss_conf

def detect(predictions):
    # predictions - torch.Tensor
    #             - nB x sum(nA x nG x nG) x 5 - (conf, box)
    predictions = torch.FloatTensor(predictions.cpu())

    # Break it down
    pred_conf  = torch.sigmoid(predictions[:,:, 4])
    pred_boxes = predictions[:,:,:4]

    best_box =  []
    best_conf = []
    for b in range(predictions.size(0)):
        max_conf = -1
        for i in range(predictions.size(1)):

            # NOTE: Works for SINGLE object
            conf = pred_conf[b,i]
            box  = pred_boxes[b,i]

            if conf > max_conf:
                max_conf = conf
                box_best = box

        best_box.append(box_best)
        best_conf.append(max_conf)

    return best_box, best_conf

# -----------------------------------------------
# For pose estimation
def pnp(points_3D, points_2D, cameraMatrix, distCoeffs=np.zeros((1,5)), rvec=None, tvec=None, useExtrinsicGuess=False):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((5, 1), dtype='float32')

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    points_3D = np.ascontiguousarray(points_3D).reshape((-1,1,3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1,1,2))

    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D,
                               cameraMatrix,
                               distCoeffs, rvec, tvec, useExtrinsicGuess,
                               flags=cv2.SOLVEPNP_EPNP)

    R, _ = cv2.Rodrigues(R_exp)

    return R, np.squeeze(t)

def predictPose(x_pr, y_pr, roi, corners3D, cameraMatrix, distCoeffs=np.zeros((1,5))):
    # x_pr, y_pr  [torch.Tensor]  1 x 11
    # roi         [numpy.ndarray]
    corners2D_pr = torch.cat((x_pr, y_pr), dim=0) # [2 x 11]
    corners2D_pr = corners2D_pr.cpu().t().numpy() # [11 x 2]

    # Apply RoI
    xmin, xmax, ymin, ymax = roi
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * (xmax-xmin) + xmin
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * (ymax-ymin) + ymin

    # Compute [R|t] by pnp
    R_pr, t_pr = pnp(corners3D, corners2D_pr, cameraMatrix, distCoeffs)

    return R_pr, t_pr

# -----------------------------------------------
# IoU computation
def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False):
    # Returns the IoU of box1 to box2.
    # - Both are torch.tensor of length 4

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area + 1e-8

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-8  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou

def bbox_wh_iou(wh1, wh2):
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-8) + w2 * h2 - inter_area
    return inter_area / union_area

# -----------------------------------------------
# Pose Metrics
def error_translation(t_pr, t_gt):
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))

    return np.sqrt(np.sum(np.square(t_gt - t_pr)))

def error_orientation(q_pr, q_gt):
    # q must be [qvec, qcos]
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    qdot = np.dot(q_pr, q_gt)
    return np.rad2deg(2*np.arccos(np.abs(qdot))) # [deg]

def speed_score(t_pr, q_pr, t_gt, q_gt):
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(q_pr, q_gt) # [deg]

    t_gt = np.reshape(t_gt, (3,))
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_r = np.deg2rad(err_q)
    return speed_t + speed_r
