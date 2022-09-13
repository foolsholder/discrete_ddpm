import ml_collections


def create_big_model_config(pp: bool):
    model = ml_collections.ConfigDict()

    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.name = 'ddpm'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    return model


def create_default_cifar_config(resolution: int = 32):
    config = ml_collections.ConfigDict()

    # data
    data = config.data = ml_collections.ConfigDict()
    data.image_size = resolution
    data.num_channels = 3
    data.centered = True
    data.norm_mean = (0.5)
    data.norm_std = (0.5)
    # model

    config.model = create_big_model_config(False)

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.weight_decay = 0

    # training
    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 250_000
    training.checkpoint_freq = 5_000
    training.eval_freq = 5_000
    training.snapshot_freq = 10_000
    training.snapshot_batch_size = 100
    training.batch_size = 128

    training.checkpoints_folder = './ddpm_checkpoints/'
    config.checkpoints_prefix = ''
    config.checkpoint_path = 'where???decide on your own, dude'

    # sde
    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-dyn'
    sde.solver = 'ancestral'
    sde.N = 1000
    sde.beta_min = 1e-4
    sde.beta_max = 2e-2

    config.device = 'cuda:0'
    return config
