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

from .Park2019KRNDataset import Park2019KRNDataset
from .SPNDataset         import SPNDataset
from .transforms         import build_transforms

def build_dataset(cfg, is_train=True, is_source=True):

    transforms = build_transforms(cfg.model_name, cfg.input_shape, is_train=is_train)

    if cfg.model_name == 'krn':
        dataset = Park2019KRNDataset(cfg, transforms, is_train, is_source)
    elif cfg.model_name == 'spn':
        dataset = SPNDataset(cfg, transforms, is_train, is_source)

    return dataset

def make_dataloader(cfg, is_train=True, is_source=True):
    if is_train:
        images_per_gpu = cfg.batch_size
        shuffle = True
        num_workers = cfg.num_workers
    else:
        images_per_gpu = 1
        shuffle = False
        num_workers = 0

    dataset = build_dataset(cfg, is_train, is_source)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=cfg.dann # drop last if DANN
    )

    return data_loader
