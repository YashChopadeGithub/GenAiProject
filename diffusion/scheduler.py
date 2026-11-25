"""
DDPM/DDIM Noise Scheduler for Document Synthesis.

Implements:
  1. Forward process  q(x_t | x_0): add noise to clean image at timestep t
  2. Reverse process  p(x_{t-1} | x_t): predict clean image from noisy input
  3. DDIM sampling: accelerated deterministic sampling (250 steps instead of 1000)
  4. Cosine noise schedule (Improved DDPM - more stable than linear)
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple


class DDPMScheduler:
    """
    DDPM Noise Scheduler with optional DDIM accelerated sampling.

    Noise Schedule Options:
      - "linear": Original DDPM (Ho et al. 2020) - simple linear beta schedule
      - "cosine": Improved DDPM (Nichol & Dhariwal 2021) - better for fine details

    The cosine schedule results in slower noise addition at low timesteps,
    which is especially beneficial for document images with sharp text.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = device

        # ── Build Beta Schedule ────────────────────────────────────────────
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        betas = betas.clamp(0.0, 0.9999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # ── Pre-compute useful quantities ──────────────────────────────────
        self.betas                    = betas.to(device)
        self.alphas                   = alphas.to(device)
        self.alphas_cumprod           = alphas_cumprod.to(device)
        self.alphas_cumprod_prev      = alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod      = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.log_one_minus_alphas_cumprod  = torch.log(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod     = torch.sqrt(1.0 / alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod   = torch.sqrt(1.0 / alphas_cumprod - 1).to(device)

        # Posterior distribution q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance         = posterior_variance.clamp(1e-20).to(device)
        self.posterior_log_variance     = torch.log(posterior_variance.clamp(1e-20)).to(device)
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(device)
        self.posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).to(device)

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.
        Results in smoother noise addition curve, better preserving fine document details.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.9999)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting."""
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(shape) - 1)))

    # ── Forward Process (Add Noise) ────────────────────────────────────────────

    def add_noise(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_start: (B, C, H, W) clean image in [-1, 1]
            t:       (B,) integer timesteps
            noise:   Optional pre-sampled noise (B, C, H, W)

        Returns:
            x_noisy: Noisy image at timestep t
            noise:   The noise that was added (needed for loss computation)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar  = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus  = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_noisy = sqrt_alpha_bar * x_start + sqrt_one_minus * noise
        return x_noisy, noise

    # ── Reverse Process (Denoise One Step) ────────────────────────────────────

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """Estimate x_0 from predicted noise at timestep t."""
        sqrt_recip    = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1  = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise_pred

    @torch.no_grad()
    def ddpm_step(
        self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one DDPM reverse step: x_t -> x_{t-1}.

        p(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma^2)

        Args:
            x_t:        (B, C, H, W) noisy image at time t
            t:          (B,) current timestep
            noise_pred: (B, C, H, W) predicted noise from U-Net

        Returns:
            x_{t-1}: (B, C, H, W) slightly denoised image
        """
        # Compute x_0 estimate
        x0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        x0_pred = x0_pred.clamp(-1, 1)

        # Compute posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean  = coef1 * x0_pred + coef2 * x_t

        # Add noise (but not at the last step t=0)
        noise = torch.randn_like(x_t)
        var   = self._extract(self.posterior_variance, t, x_t.shape)
        mask  = (t > 0).float().view(-1, 1, 1, 1)
        return mean + mask * torch.sqrt(var) * noise

    # ── DDIM Sampling (Accelerated, Fewer Steps) ───────────────────────────────

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Perform one DDIM step (deterministic when eta=0).
        Allows sampling with much fewer steps (e.g., 250 vs 1000).

        Args:
            x_t:        (B, C, H, W) noisy image at time t
            t:          current scalar timestep
            t_prev:     previous scalar timestep (smaller)
            noise_pred: (B, C, H, W) predicted noise from U-Net
            eta:        stochasticity [0 = deterministic DDIM, 1 = stochastic DDPM]

        Returns:
            x_{t_prev}: (B, C, H, W)
        """
        alpha_bar      = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        sqrt_alpha_bar      = alpha_bar.sqrt()
        sqrt_alpha_bar_prev = alpha_bar_prev.sqrt()
        sqrt_one_minus      = (1 - alpha_bar).sqrt()

        # Predicted x_0
        x0_pred = (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha_bar
        x0_pred = x0_pred.clamp(-1, 1)

        # Direction pointing to x_t
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()
        dir_xt = (1 - alpha_bar_prev - sigma ** 2).sqrt() * noise_pred

        noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
        return sqrt_alpha_bar_prev * x0_pred + dir_xt + noise

    # ── Full DDIM Sampling Loop (inference) ───────────────────────────────────

    @torch.no_grad()
    def sample_ddim(
        self,
        model,
        shape: Tuple,
        condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        sampling_timesteps: int = 250,
        eta: float = 0.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Full DDIM reverse process from pure Gaussian noise to clean image.

        Args:
            model:              DiffusionUNet instance
            shape:              (B, C, H, W) desired output shape
            condition:          (B, 3, H, W) layout condition map
            cfg_scale:          Classifier-Free Guidance scale (1.0 = disabled)
            sampling_timesteps: How many denoising steps (less = faster)
            eta:                DDIM eta (0=deterministic, 1=stochastic)
            device:             torch device string

        Returns:
            x_0: (B, C, H, W) final generated image in [-1, 1]
        """
        model.eval()
        dev_str = str(device)
        is_cuda = "cuda" in dev_str

        # Build DDIM timestep sequence
        times = np.linspace(0, self.timesteps - 1, sampling_timesteps + 1, dtype=int)
        times = list(reversed(times))  # from T down to 0

        x = torch.randn(shape, device=device)

        with torch.amp.autocast('cuda', enabled=is_cuda):
            for i in range(len(times) - 1):
                t_curr = times[i]
                t_prev = times[i + 1]

                t_tensor = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)

                # Conditional prediction
                noise_pred_cond = model(x, t_tensor, condition)

                # Classifier-Free Guidance
                noise_pred = noise_pred_cond
                if cfg_scale > 1.0 and condition is not None:
                    noise_pred_uncond = model(x, t_tensor, condition, cfg_drop=True)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                if i == len(times) - 2:
                    # Final step: directly predict clean x0 without adding directional noise back
                    alpha_bar = self.alphas_cumprod[t_curr]
                    sqrt_alpha_bar = alpha_bar.sqrt()
                    sqrt_one_minus = (1.0 - alpha_bar).sqrt()
                    x = (x - sqrt_one_minus * noise_pred) / sqrt_alpha_bar
                else:
                    x = self.ddim_step(x, t_curr, t_prev, noise_pred, eta=eta)

        return x.clamp(-1, 1)

