from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import patches as patches

import numpy as np

import torchvision.transforms.functional as F

# -----------------------------------------------------------------------
# Helper functions for image visualization / saving.
def imshow(tensor, mode='RGB', savefn=None):
    tensor = tensor.clamp(0.0, 1.0).cpu()
    im     = F.to_pil_image(tensor, mode=mode)
    cmap   = 'gray' if mode == 'L' else None
    plt.imshow(im, cmap=cmap)
    plt.axis('off')
    plt.show()
    if savefn is not None:
        plt.savefig(savefn, bbox_inches = 'tight', pad_inches = 0)

def plot_2D_bbox(tensor, bbox, xywh=True):
    # bbox: [xmin, xmax, ymin, ymax]

    # Processing
    data = F.to_pil_image(tensor)

    if xywh:
        x, y, w, h = bbox
        xmin, xmax, ymin, ymax = x-w/2, x+w/2, y-h/2, y+h/2
    else:
        xmin, xmax, ymin, ymax = bbox

    # figure
    fig = plt.figure()
    plt.imshow(data)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 
                color='g', linewidth=1.5)
    plt.show()

def scatter_keypoints(tensor, x_pr, y_pr, normalized=False):
    # Tensor:     C x H x W
    # Keypoints : Nc x 2
    _, h, w = tensor.shape
    data  = tensor.cpu().numpy().astype(np.float32).transpose((1,2,0)) * 255
    data  = data.astype(np.uint8) # [H x W x C]

    if normalized:
        x_pr = x_pr * w
        y_pr = y_pr * h

    # figure
    fig = plt.figure()
    plt.imshow(Image.fromarray(data))
    plt.scatter(x_pr , y_pr, c='lime', marker='+')
    plt.show()
