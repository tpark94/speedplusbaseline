from __future__ import print_function

import argparse
import os

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from stylePredictor import StylePredictor
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

''' Run below to get SPEED+ synthetic embedding
python src/styleaug/get_embedding_mean_and_covariance.py \
    --data_dir /home/jeffpark/SLAB/Dataset/speedplus/synthetic/images/ \
    --batchsize 16 --input_size (480,320) \
    --checkpoint src/styleaug/checkpoints/checkpoint_stylepredictor.pth
'''

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='/home/jeffpark/SLAB/Dataset/speedplus/synthetic/',type=str,help='Path to style image directory (make sure the images are in a subdirectory of data_dir or else ImageFolder will complain)')
parser.add_argument('--batchsize',default=8,type=int)
parser.add_argument('--input_size',default=(480,320),type=tuple,help='Size to resize images to')
parser.add_argument('--checkpoint',default='src/styleaug/checkpoints/checkpoint_stylepredictor.pth',type=str,help='Path to style predictor checkpoint')
args = parser.parse_args()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = Compose([Resize(args.input_size), ToTensor()])

# create the one and only loader:
print("Creating loaders... ", end=' ')
dataset = ImageFolder(args.data_dir,transform=transform)
loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
print("Done")

# create models:
print("Creating models... ", end=' ')
stylePredictor = StylePredictor()
stylePredictor.to(device)
stylePredictor.eval()
print("Done")

# load checkpoint:
checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
stylePredictor.load_state_dict(checkpoint['state_dict_stylepredictor'])


# =================================== MAIN LOOP ===================================

embeddings = np.zeros((len(dataset),100))
i = 0 # number of images processed so far
for images, _ in tqdm(loader):

    style_im = images.to(device)

    embedding_batch = stylePredictor(style_im)
    embeddings[i:i+embedding_batch.shape[0],:] = embedding_batch.detach().cpu().numpy()
    i += embedding_batch.shape[0]

embeddings = embeddings[:i] # probably there are a few empty rows to chop off here because of drop_last=True

# get mean vector and covariance matrix:
mean = np.mean(embeddings, axis=0)
sigma = np.cov(embeddings, rowvar=False)

# save all embeddings:
checkpoint_dir = 'src/styleaug/checkpoints'
np.save(os.path.join(checkpoint_dir,'embeddings_speedplus.npy'), embeddings)

# save mean and convariance:
np.save(os.path.join(checkpoint_dir,'embedding_mean_speedplus.npy'), mean)
np.save(os.path.join(checkpoint_dir,'embedding_covariance_speedplus.npy'), sigma)