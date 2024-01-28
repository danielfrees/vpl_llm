
import torch
import torch.nn as nn
import math

class PreFusionEncoder(nn.Module):
    def __init__(self, embed_dim, latent_dim):
        super(PreFusionEncoder, self).__init__()
        self.FC_mean = nn.Linear(embed_dim, latent_dim)
        self.FC_var = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = x.bfloat16()
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        return mean, log_var
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(2*embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.2),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.2),
        )
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self._model(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self._model(x)


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder, llm, latent_dim, learned_prior=False):
        super(VAEModel, self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        self.llm = llm
        self.latent_dim = latent_dim
        self.prior_mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.prior_log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, fused, e0, e1):
        # encoder_input = torch.cat([e0, e1], dim=1).reshape(e0.shape[0], -1)
        mean, log_var = self.Encoder(fused)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        x0 = torch.cat([e0, z], dim=1)
        x1 = torch.cat([e1, z], dim=1)
        r0 = self.Decoder(x0)
        r1 = self.Decoder(x1)
        # p = torch.sigmoid(r0 - r1)
        return None, r0, r1, mean, log_var


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return