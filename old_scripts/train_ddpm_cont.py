from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = create_default_cifar_config(resolution=16)
diffusion = DiffusionRunner(config)

config.checkpoints_prefix = '16x16_loss_div4'
diffusion.train(
    project_name='upsamling_sde',
    experiment_name='vp-sde-16x16_STDiv2'
)
