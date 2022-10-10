import torch
import torchvision
import math
import numpy as np

from skimage.io import imsave
from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner
from typing import Optional, Sequence
from tqdm.auto import trange

from sys import argv
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=250)
    parser.add_argument('--idx_start', type=int, default=0)
    parser.add_argument('--total_images', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--path', type=str, required=True)
    return parser.parse_args()


args = parse_args()


config = create_default_cifar_config(resolution=32)
config.checkpoints_prefix = ''
config.device = args.device

gen = DiffusionRunner(config, eval=True)

total_images = args.total_images
idx_start = args.idx_start
batch_size = args.n_samples
folder_path = args.path

os.makedirs(folder_path, exist_ok=True)

num_iters = total_images // batch_size

global_idx = idx_start

for idx in trange(num_iters):
    images: torch.Tensor = gen.sample_images(batch_size=batch_size).cpu()
    images = images.permute(0, 2, 3, 1).data.numpy().astype(np.uint8)

    for i in range(len(images)):
         imsave(os.path.join(folder_path, f'{global_idx}.png'), images[i])
         global_idx += 1

