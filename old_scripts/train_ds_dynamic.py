from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner
from ds_diffusion import DSDiffusionRunner

import os

config = create_default_cifar_config(resolution=32)
config.sde.typename = 'ds-sde-proper'
config.sde.solver = 'ds-euler'
config.sde.tau = 0.5

ds_diff = DSDiffusionRunner(config, eval=False)

config.checkpoints_prefix = 'ds_05'
ds_diff.train(
    project_name='upsamling_sde',
    experiment_name='ds-diff'
)
