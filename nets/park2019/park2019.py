import torch
import torch.nn as nn
from torchvision import models

from utils.utils_eval import get_grids, giouloss

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
# Object Detection Network (ODN)
class YOLOLayer(nn.Module):
    """ This class post-processes the output.
        Loss is computed over ALL stages and anchors (nA = 9) at the end
    """
    def __init__(self):
        super(YOLOLayer, self).__init__()

    def forward(self, p, anchors, train=True):
        # p : nB x (nA x 5) x nG x nG
        nB, nA, nG = p.size(0), len(anchors), p.size(2)
        assert p.size(2) == p.size(3)

        # Device
        device = p.device

        # Re-order to nB x nA x nG x nG x 5
        p = p.view(nB, nA, -1, nG, nG).permute(0,1,3,4,2).contiguous()

        # Process anchors
        anchors_scaled = torch.tensor([ (aw*nG, ah*nG) for aw, ah in anchors], dtype=torch.float32, device=device)
        anchor_w = anchors_scaled[:,0].view(1, nA, 1, 1)
        anchor_h = anchors_scaled[:,1].view(1, nA, 1, 1)

        # Process the output with sigmoids
        conf = p[..., 0] # Logits
        x    = p[..., 1].sigmoid()
        y    = p[..., 2].sigmoid()
        w    = p[..., 3]
        h    = p[..., 4]

        # Offsets
        grid_x, grid_y = get_grids(nG)

        # Apply offsets
        box = torch.zeros_like(p[..., 1:], dtype=torch.float32, device=device)
        box[..., 0] = x + grid_x.to(device)
        box[..., 1] = y + grid_y.to(device)
        box[..., 2] = torch.exp(w) * anchor_w
        box[..., 3] = torch.exp(h) * anchor_h

        if train:
            # Return [nB, nA, nG, nG, 5 (xywh, conf)]
            return torch.cat( (box.view(nB, nA, nG, nG, 4) / nG,
                               conf.view(nB, nA, nG, nG, 1)), -1)
        else:
            # Return [nB, -1, 5 (xywh, conf)]
            return torch.cat( (box.view(nB, -1, 4) / nG,
                               conf.view(nB, -1, 1)), -1)

class ObjectDetectionNet(nn.Module):
    def __init__(self):
        super(ObjectDetectionNet, self).__init__()
        self.anchors = [[[0.60109,1.2229],[0.64568,0.94241],[0.58851,0.64]],
                        [[0.39969,0.87549],[0.34564,0.6335],[0.38756,0.40009]],
                        [[0.23926,0.47623],[0.18063,0.33938],[0.22621,0.22773]]]

        # Base feature extractor (ImageNet-pretrained)
        self.base = models.mobilenet_v2(pretrained=True)
        self.base = nn.ModuleList(list(self.base.features.children())[:-1])

        # Extra layers
        # feature_layer_name = [['DW','DW','DW'],
        #                       [13,'DW','DW','DW'],
        #                       [6,'DW','DW','DW']]
        # feature_layer_size = [[1024, 1024, 1024],
        #                       [256, 512, 512, 512],
        #                       [128, 256, 256, 256]]
        self.extras = nn.ModuleList([
            ConvDw(320, 1024, stride=1), # self.base[-1].depth,
            ConvDw(1024, 1024, stride=1),
            ConvDw(1024, 1024, stride=1), # End of first detection stage
            RouterV3(1024, 256),
            ConvDw(256 + 96, 512, stride=1), # + self.base[13].depth
            ConvDw(512, 512, stride=1),
            ConvDw(512, 512, stride=1),   # End of second detection stage
            RouterV3(512, 128),
            ConvDw(128 + 32, 256, stride=1), # + self.base[6].depth,
            ConvDw(256, 256, stride=1),
            ConvDw(256, 256, stride=1)    # End of third detection stage
        ])

        # Head layer(s), in case detection is made at multiple levels
        self.head = nn.ModuleList([
            nn.Conv2d(1024, 3*5, 1),
            nn.Conv2d(512,  3*5, 1),
            nn.Conv2d(256,  3*5, 1)
        ])

        # Post-processing layer
        self.post = YOLOLayer()

    def forward(self, x, y=None):
        predetect = []
        output    = []

        # Feed-forward (Base)
        for i, block in enumerate(self.base):
            x = block(x)
            if i == 13: x13 = x
            if i == 6:  x6  = x

        # Rest
        for i, block in enumerate(self.extras[:3]):
            x = block(x)
        predetect.append(x)

        for i, block in enumerate(self.extras[3:7]):
            x = block(x, x13) if i == 0 else block(x)
        predetect.append(x)

        for i, block in enumerate(self.extras[7:]):
            x = block(x, x6) if i == 0 else block(x)
        predetect.append(x)

        # Detection at three levels
        for i in range(3):
            x = self.head[i](predetect[i])
            x = self.post(x, self.anchors[i], train=y is not None)
            output.append(x)

        if y is not None:
            loss, loss_giou, loss_conf = giouloss(output, y, anchors=self.anchors,
                                            iou_scale=1.0, obj_scale=5.0, noobj_scale=0.1)

            sm = {'loss_giou': float(loss_giou),
                  'loss_conf': float(loss_conf)}
            return loss, sm
        else:
            return torch.cat(output, axis=1)

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
            txc = y[:,list(range(0,2*self.nK,2))]
            tyc = y[:,list(range(1,2*self.nK,2))]

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
