import torch
from torchvision.datasets import CIFAR10
from skimage.io import imsave
import numpy as np
import os
from tqdm.auto import tqdm


if __name__ == '__main__':
    dataset = CIFAR10(train=True, root='../data', download=True)

    os.makedirs('../cifar_train', exist_ok=True)
    for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
        image = np.array(image, np.uint8)
        imsave(os.path.join('../cifar_train', f'{idx}.png'), image)