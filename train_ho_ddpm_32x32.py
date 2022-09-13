from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner


config = create_default_cifar_config(resolution=32)
diffusion = DiffusionRunner(config)

config.checkpoints_prefix = 'ddpm_dyn_ddpm'
diffusion.train(
    project_name='experimental_ncsn',
    experiment_name='ddpm_dyn_ddpm'
)

