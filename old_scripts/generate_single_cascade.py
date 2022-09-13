import torch
import torchvision
import math
import numpy as np

from skimage.io import imsave
from default_cifar_config import create_default_cifar_config
from upsample_runner import UpsampleGeneration
from diffusion import DiffusionRunner
from typing import Optional, Sequence
from tqdm.auto import trange

from sys import argv
import os


"""
argv[1] - cuda:0, cuda:1 etc
argv[2] - time to switch model
argv[3] - total_images
argv[4] - idx_start
argv[5] - batch_size
argv[6] - folder
"""

configs = create_default_cifar_config(resolution=32)
configs.checkpoints_prefix = 'best'

gen = DiffusionRunner(configs, eval=True)

total_images = int(argv[3])
idx_start = int(argv[4])
batch_size = int(argv[5])
folder_path = argv[6]

os.makedirs(folder_path, exist_ok=True)

num_iters = total_images // batch_size

global_idx = idx_start

for idx in trange(num_iters):
    images: torch.Tensor = gen.sample_images(batch_size=batch_size).cpu()
    images = images.permute(0, 2, 3, 1).data.numpy().astype(np.uint8)

    for i in range(len(images)):
         imsave(os.path.join(folder_path, f'{global_idx}.png'), images[i])
         global_idx += 1
