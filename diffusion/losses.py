"""
Composite Loss Formulation for DDPM Document Diffusion Model.

Combines 4 complementary loss objectives:
  1. MSE Loss (L_MSE):       Standard DDPM L2 noise residual loss ||epsilon - epsilon_theta||^2
  2. L1 Loss (L_L1):         L1 noise loss ||epsilon - epsilon_theta||_1 for crisp character edges
  3. Variational Lower Bound (L_VLB): Analytical KL divergence between posterior q(x_{t-1}|x_t,x_0)
                             and parameterized reverse transition p_theta(x_{t-1}|x_t)
  4. Perceptual Loss (L_perc): Deep VGG-16 feature distance between predicted clean image x_0_hat and ground truth x_0

Total Loss:
  L_total = lambda_mse * L_MSE + lambda_l1 * L_L1 + lambda_vlb * L_VLB + lambda_perc * L_perc
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from diffusion import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Perceptual Feature Loss (VGG-16) with Structural Fallback
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionPerceptualLoss(nn.Module):
    """
    Perceptual loss comparing predicted clean document image x_0_hat with target x_0.
    Uses frozen VGG-16 feature representations (relu1_2, relu2_2, relu3_3).
    If offline or weights cannot be loaded, falls back to multi-scale Laplacian/Sobel edge loss.
    """
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or cfg.DEVICE
        self.use_vgg = False

        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            features = vgg.features
            self.slice1 = nn.Sequential(*list(features.children())[:4]).to(self.device)   # relu1_2
            self.slice2 = nn.Sequential(*list(features.children())[4:9]).to(self.device)  # relu2_2
            self.slice3 = nn.Sequential(*list(features.children())[9:16]).to(self.device) # relu3_3

            for p in self.parameters():
                p.requires_grad = False

            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
            self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))
            self.use_vgg = True
        except Exception as e:
            # Fallback: structural Sobel edge loss if VGG weights cannot be fetched
            self.use_vgg = False

    def _vgg_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _sobel_fallback(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Lightweight spatial gradient comparison when VGG is offline."""
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=pred.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=pred.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        gx_pred, gx_tgt = F.conv2d(pred, kx, padding=1, groups=3), F.conv2d(target, kx, padding=1, groups=3)
        gy_pred, gy_tgt = F.conv2d(pred, ky, padding=1, groups=3), F.conv2d(target, ky, padding=1, groups=3)

        return F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt) + F.l1_loss(pred, target)

    def forward(self, pred_x0: torch.Tensor, target_x0: torch.Tensor) -> torch.Tensor:
        """
        Input tensors are expected in [0, 1] range.
        """
        if not self.use_vgg:
            return self._sobel_fallback(pred_x0, target_x0)

        # Downsample to 128x128 to drastically reduce VGG backward activation memory on 4GB VRAM
        if pred_x0.shape[-1] > 128:
            pred_x0 = F.interpolate(pred_x0, size=(128, 128), mode="bilinear", align_corners=False)
            target_x0 = F.interpolate(target_x0, size=(128, 128), mode="bilinear", align_corners=False)

        # Normalize to ImageNet distribution
        p_norm = self._vgg_norm(pred_x0)
        t_norm = self._vgg_norm(target_x0)

        f1_p, f1_t = self.slice1(p_norm), self.slice1(t_norm)
        f2_p, f2_t = self.slice2(f1_p),   self.slice2(f1_t)
        f3_p, f3_t = self.slice3(f2_p),   self.slice3(f2_t)

        loss = F.l1_loss(f1_p, f1_t) + F.l1_loss(f2_p, f2_t) + F.l1_loss(f3_p, f3_t)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 2. Variational Lower Bound (VLB) KL Divergence Calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_variational_lower_bound(
    scheduler,
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Variational Lower Bound (VLB) loss term via analytical KL divergence.

    At timestep t:
      True posterior:      q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_tilde_t(x_t, x_0), beta_tilde_t * I)
      Predicted posterior: p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), beta_tilde_t * I)

    KL Divergence simplifies to:
      D_KL(q || p) = 1 / (2 * beta_tilde_t) * ||mu_tilde_t - mu_theta||^2
    """
    # 1. True posterior mean: mu_tilde_t = coef1 * x_0 + coef2 * x_t
    coef1 = scheduler.posterior_mean_coef1[t].view(-1, 1, 1, 1)
    coef2 = scheduler.posterior_mean_coef2[t].view(-1, 1, 1, 1)
    true_mean = coef1 * x_0 + coef2 * x_t

    # 2. Model predicted posterior mean from noise prediction
    # mu_theta = 1 / sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_cumprod_t)) * epsilon_theta)
    beta_t = scheduler.betas[t].view(-1, 1, 1, 1)
    alpha_t = scheduler.alphas[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod) * noise_pred)

    # 3. Posterior variance beta_tilde_t
    variance = scheduler.posterior_variance[t].view(-1, 1, 1, 1)

    # 4. KL divergence for Gaussian distributions (in-place mean squared error)
    kl_div = 0.5 * F.mse_loss(model_mean, true_mean) / variance.mean().clamp(min=1e-8)

    return kl_div


# ─────────────────────────────────────────────────────────────────────────────
# 3. Composite Multi-Task Diffusion Loss
# ─────────────────────────────────────────────────────────────────────────────

class CompositeDiffusionLoss(nn.Module):
    """
    Multi-objective composite loss evaluator:
      - MSE Loss:         Noise vector regression
      - L1 Loss:          Fine character edge alignment
      - VLB Loss:         Variational Lower Bound KL divergence
      - Perceptual Loss:  VGG deep feature consistency on reconstructed clean image x_0_hat
    """
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_l1: float = 0.5,
        lambda_vlb: float = 0.05,
        lambda_perc: float = 0.10,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or cfg.DEVICE
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_vlb = lambda_vlb
        self.lambda_perc = lambda_perc

        self.perceptual_fn = DiffusionPerceptualLoss(device=self.device)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        scheduler,
    ) -> dict:
        """
        Args:
          noise_pred:  Predicted noise epsilon_theta (B, 3, H, W)
          noise_true:  True Gaussian noise epsilon (B, 3, H, W)
          x_0:         Clean target image in [-1, 1]
          x_t:         Noisy image at timestep t in [-1, 1]
          t:           Timestep tensor (B,)
          scheduler:   DDPMScheduler instance

        Returns:
          dict with 'total_loss', 'mse_loss', 'l1_loss', 'vlb_loss', 'perc_loss'
        """
        # 1. MSE Loss
        loss_mse = F.mse_loss(noise_pred, noise_true)

        # 2. L1 Loss
        loss_l1 = F.l1_loss(noise_pred, noise_true)

        # 3. Variational Lower Bound (VLB)
        loss_vlb = compute_variational_lower_bound(
            scheduler=scheduler,
            x_0=x_0,
            x_t=x_t,
            noise_pred=noise_pred,
            t=t,
        )

        # 4. Perceptual Loss on predicted x_0_hat (downsampled to 128x128 for 4GB VRAM)
        sqrt_alpha_bar = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar.clamp(min=1e-6)

        if x_0_pred.shape[-1] > 128:
            x_0_pred_small = F.interpolate(x_0_pred, size=(128, 128), mode="bilinear", align_corners=False)
            x_0_small      = F.interpolate(x_0, size=(128, 128), mode="bilinear", align_corners=False)
        else:
            x_0_pred_small = x_0_pred
            x_0_small      = x_0

        x_0_pred_vis = (x_0_pred_small * 0.5 + 0.5).clamp(0, 1)
        x_0_vis      = (x_0_small * 0.5 + 0.5).clamp(0, 1)

        loss_perc = self.perceptual_fn(x_0_pred_vis, x_0_vis)

        # Composite total loss
        total_loss = (
            self.lambda_mse  * loss_mse +
            self.lambda_l1   * loss_l1 +
            self.lambda_vlb  * loss_vlb +
            self.lambda_perc * loss_perc
        )

        return {
            "total_loss": total_loss,
            "mse_loss":   loss_mse,
            "l1_loss":    loss_l1,
            "vlb_loss":   loss_vlb,
            "perc_loss":  loss_perc,
        }
