import torch
import torch.nn as nn

from .ghiasi import Ghiasi
import numpy as np
from os.path import join, dirname

''' Cloned on 09/21/2021
    - Only ghiasi portion remains and is copied to GPU to save memory
    - This means only sample from embedding mean (get it separately from SPEED+ training images)
'''
class StyleAugmentor(nn.Module):
    def __init__(self, alpha, device):
        super(StyleAugmentor,self).__init__()
        self.alpha  = alpha
        self.device = device

        # create transformer
        self.ghiasi = Ghiasi()
        self.ghiasi.to(device)

        # load checkpoints:
        checkpoint_ghiasi     = torch.load(join(dirname(__file__),'checkpoints/checkpoint_transformer.pth'))
        checkpoint_embeddings = torch.load(join(dirname(__file__),'checkpoints/checkpoint_embeddings.pth'))

        # load weights for ghiasi
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'], strict=False)

        # load mean imagenet embedding:
        self.imagenet_embedding = torch.from_numpy(np.load(join(dirname(__file__),
                                        'checkpoints/embedding_mean_speedplus.npy')))
        self.imagenet_embedding = self.imagenet_embedding.float().to(device)

        # get mean and covariance of PBN style embeddings:
        self.mean = checkpoint_embeddings['pbn_embedding_mean'].to(device) # 1 x 100
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']

        # compute SVD of covariance matrix:
        u, s, _ = np.linalg.svd(self.cov.numpy())

        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float().to(device) # 100 x 100

    def sample_embedding(self,n):
        # n: number of embeddings to sample
        # returns n x 100 embedding tensor
        embedding = torch.randn(n,100).to(self.device) # n x 100
        embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean # n x 100
        return embedding

    def forward(self,x):
        # augments a batch of images with style randomization
        # x: B x C x H x W image tensor
        # alpha: float in [0,1], controls interpolation between random style and original style

        # style embedding for when alpha=0:
        base = self.imagenet_embedding # SPEED+ embedding, despite naming

        with torch.no_grad():
            # sample a random embedding
            embedding = self.sample_embedding(x.size(0))

            # interpolate style embeddings:
            embedding = self.alpha*embedding + (1-self.alpha)*base

            restyled = self.ghiasi(x,embedding)

        return restyled.detach() # detach prevents the user from accidentally backpropagating errors into stylePredictor or ghiasi while training a downstream model

if __name__=='__main__':
    import cv2
    import torchvision.transforms.functional as TF

    data = cv2.imread('/home/jeffpark/SLAB/Dataset/speedplus/synthetic/images/img000100.jpg', cv2.IMREAD_COLOR)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = cv2.resize(data, (480,320))
    data = TF.to_tensor(data)

    device = torch.device('cuda:0')
    augmentor = StyleAugmentor(device)
    styled = augmentor(data.unsqueeze(0).to(device), alpha=0.5)

    styled = styled.squeeze(0).permute(1,2,0).cpu().numpy()

    cv2.imshow('0', styled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()