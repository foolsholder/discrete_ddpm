from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner


config = create_default_cifar_config(resolution=32)
diffusion = DiffusionRunner(config)

config.checkpoints_prefix = 'improved_ddpm'
diffusion.train(
    project_name='experimental_ddpm',
    experiment_name='ddpm'
)

