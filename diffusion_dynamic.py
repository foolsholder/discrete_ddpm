import torch
import numpy as np
from typing import Dict, Union, List, Optional


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = float(i) / num_diffusion_timesteps
        t2 = (i + 1.) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.FloatTensor(betas)


def get_cosine_betas(num_steps_diff: int):
    return betas_for_alpha_bar(num_steps_diff, lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2)


def get_betas(num_steps_diff: int, noise_scheduler: str):
    if noise_scheduler == 'linear':
        return torch.linspace(1e-4, 2e-2, num_steps_diff).float()
    return get_cosine_betas(num_steps_diff)


def _extract_tensor_with_time(arr_t, t: torch.LongTensor):
    vals = arr_t[t]
    vals = vals[:, None, None, None].to(t.device)
    return vals


class VariancePreserving:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = N = config.sde.N
        #self.beta_min = beta_min = config.sde.beta_min
        #self.beta_max = beta_max = config.sde.beta_max

        self.betas = get_betas(self.N, config.sde.noise_scheduler)

        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.FloatTensor([1.0]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.FloatTensor([0.0])])

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.one_m_alphas_cumprod = 1 - self.alphas_cumprod
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.posterior_mean_coef_x_0 = self.betas * self.sqrt_alphas_cumprod_prev / self.one_m_alphas_cumprod
        self.posterior_mean_coef_x_t = self.sqrt_alphas * (1 - self.alphas_cumprod_prev) * self.sqrt_alphas / self.one_m_alphas_cumprod

        self.posterior_std = torch.sqrt(
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

    @property
    def T(self):
        return 1

    def prior_sampling(self, shape) -> torch.Tensor:
        return torch.randn(*shape)

    def marginal_posterior(self, x_0, x_t, t: torch.LongTensor) -> Dict[str, torch.Tensor]:
        coef_x_0 = _extract_tensor_with_time(self.posterior_mean_coef_x_0, t)
        coef_x_t = _extract_tensor_with_time(self.posterior_mean_coef_x_t, t)

        x_mean = x_0 * coef_x_0 + x_t * coef_x_t

        std = _extract_tensor_with_time(self.posterior_std, t)

        x_new = x_mean + torch.randn_like(x_mean) * std
        return {
            "x": x_new,
            "x_mean": x_mean
        }

    def marginal_forward(self, clean_x, t: torch.LongTensor) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(clean_x)
        coef_x = _extract_tensor_with_time(self.sqrt_alphas_cumprod, t)
        x_t = coef_x * clean_x + noise * self.marginal_std(t)
        return {
            "x_t": x_t,
            "noise": noise
        }

    def marginal_std(self, t: torch.LongTensor) -> torch.Tensor:
        return _extract_tensor_with_time(self.sqrt_1m_alphas_cumprod, t)

    def calc_score(self, model, x_t, t: torch.LongTensor) -> Dict[str, torch.Tensor]:
        labels = t.float() # just technic for training, SDE looks the same
        eps_theta = model(x_t, labels)
        std = self.marginal_std(t)
        score = -eps_theta / std
        return {
            "score": score,
            "eps_theta": eps_theta
        }

    def predict_x0(self, x_t, t: torch.LongTensor, eps_theta=None, model=None):
        std = self.marginal_std(t)
        coef = _extract_tensor_with_time(self.alphas_cumprod, t)
        if eps_theta is None:
            eps_theta = self.calc_score(model, x_t, t)['eps_theta']
        x_0 = (x_t - eps_theta * std) / coef
        return x_0


class AncestralDiffusionSampling:
    def __init__(self, dyn: VariancePreserving, config):
        self.dyn = dyn

    def step(self, model, x_t, t: torch.LongTensor) -> Dict[str, torch.Tensor]:
        pred = self.dyn.calc_score(model, x_t, t)

        x_0 = self.dyn.predict_x0(x_t, t, pred['eps_theta'])

        return self.dyn.marginal_posterior(x_0, x_t, t)


def create_dyn(config):
    possible_sde = {
        "vp-dyn": VariancePreserving
    }
    return possible_sde[config.sde.typename](config)


def create_sampler(config, dyn):
    possible_solver = {
        "ancestral": AncestralDiffusionSampling
    }
    return possible_solver[config.sde.solver](dyn, config)
