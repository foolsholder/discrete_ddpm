import torch
import torchvision
import math
import numpy as np

from matplotlib import pyplot as plt

from default_cifar_config import prepare_upsample, create_default_cifar_config
from downsample_runner import DownsampleGeneration
from diffusion import DiffusionRunner
from ds_diffusion import DSDiffusionRunner

from typing import Optional, Sequence
from copy import copy, deepcopy
from torch.nn import functional as F
from tqdm.auto import trange
from skimage.io import imsave

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=250)
    parser.add_argument('--idx_start', type=int, default=0)
    parser.add_argument('--total_images', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--tau', type=str, required=True)
    return parser.parse_args()


args = parse_args()
configs = [
    create_default_cifar_config(resolution=16),
    create_default_cifar_config(resolution=32)
]

configs[0].sde.typename = 'vp-sde'
configs[0].sde.solver = 'euler'
configs[0].checkpoints_prefix = '16x16'

configs[1].sde.typename = 'ds-sde'
configs[1].sde.solver = 'ds-euler'
configs[1].sde.tau = float(args.tau)
configs[1].checkpoints_prefix = 'big2'

device_str = args.device
configs[0].device = device_str
configs[1].device = device_str

times = [float(args.tau)]

downsample_gen_16x16_ds = DownsampleGeneration(configs, times)

VALID_DATASET_LEN = args.total_images

num_iters = (VALID_DATASET_LEN - 1) // args.n_samples + 1

global_idx = args.idx_start
os.makedirs(args.path, exist_ok=True)

for idx in trange(num_iters):
    images: torch.Tensor = downsample_gen_16x16_ds.sample_images(batch_size=args.n_samples).cpu()
    images = images.permute(0, 2, 3, 1).data.numpy().astype(np.uint8)

    for i in range(len(images)):
        imsave(os.path.join(args.path, f'{global_idx}.png'), images[i])
        global_idx += 1
