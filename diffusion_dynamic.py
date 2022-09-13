import torch
import numpy as np
from typing import Dict, Union, List, Optional


class VariancePreserving:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = N = config.sde.N
        self.beta_min = beta_min = config.sde.beta_min
        self.beta_max = beta_max = config.sde.beta_max

        self.discrete_betas = torch.linspace(beta_min, beta_max, N).to(config.device)
        self.sqrt_betas = torch.sqrt(self.discrete_betas)

        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.one_m_alphas_cumprod = 1 - self.alphas_cumprod
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def prior_sampling(self, shape) -> torch.Tensor:
        return torch.randn(*shape)

    def marginal_posterior(self, x_0, x_t, t) -> Dict[str, torch.Tensor]:
        timestep = (t * (self.N - 1) + 1e-8).long()
        assert timestep.min().item() >= 1

        coef_x_0 = self.sqrt_alphas_cumprod[timestep - 1] * \
                   self.discrete_betas[timestep]

        coef_x_t = self.sqrt_alphas[timestep] * \
                   self.one_m_alphas_cumprod[timestep - 1]

        coef_x_0 = coef_x_0[:, None, None, None]
        coef_x_t = coef_x_t[:, None, None, None]
        #print(x_t.shape, coef_x_t.shape, flush=True)
        x_mean = (x_0 * coef_x_0 + x_t * coef_x_t) / self.one_m_alphas_cumprod[timestep][:, None, None, None]

        x_new = x_mean + torch.randn_like(x_mean) * self.sqrt_betas[timestep][:, None, None, None]
        return {
            "x": x_new,
            "x_mean": x_mean
        }

    def marginal_forward(self, clean_x, t) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(clean_x)
        timestep = (t * (self.N - 1) + 1e-8).long()
        coef_x = self.sqrt_alphas_cumprod[timestep][:, None, None, None]
        x_t = coef_x * clean_x + noise * self.marginal_std(t)
        return {
            "x_t": x_t,
            "noise": noise
        }

    def marginal_std(self, t) -> torch.Tensor:
        timestep = (t * (self.N - 1) + 1e-8).long()
        std = self.sqrt_1m_alphas_cumprod[timestep]
        return std[:, None, None, None]

    def calc_score(self, model, x_t, t) -> Dict[str, torch.Tensor]:
        labels = t * 999 # just technic for training, SDE looks the same
        eps_theta = model(x_t, labels)
        std = self.marginal_std(t)
        score = -eps_theta / std
        return {
            "score": score,
            "eps_theta": eps_theta
        }

    def predict_x0(self, x_t, t, eps_theta=None, model=None):
        timestep = (t * (self.N - 1) + 1e-8).long()
        std = self.marginal_std(t)
        coef = self.sqrt_alphas_cumprod[timestep][:, None, None, None]
        if eps_theta is None:
            eps_theta = self.calc_score(model, x_t, t)['eps_theta']
        x_0 = (x_t - eps_theta * std) / coef
        return x_0


class AncestralDiffusionSampling:
    def __init__(self, dyn: VariancePreserving, config):
        self.dyn = dyn

    def step(self, model, x_t, t) -> Dict[str, torch.Tensor]:
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
