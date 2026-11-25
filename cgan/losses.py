"""
Loss functions for Conditional GAN (cGAN) Document Synthesis:
  1. Adversarial Loss (LSGAN / BCE)
  2. L1 Pixel Distance
  3. Perceptual VGG Feature Loss (Texture & Letterform Realism)
  4. Sobel Edge Loss (Sharp Typography Boundaries)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

from cgan.config import (
    LAMBDA_GAN,
    LAMBDA_L1,
    LAMBDA_PERCEPTUAL,
    LAMBDA_EDGE,
    DEVICE
)


class VGGPerceptualLoss(nn.Module):
    """VGG-16 perceptual feature loss for realistic document texture and typography."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        # Extract features up to relu2_2 (layer 8) and relu3_3 (layer 15)
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(9)]).eval()
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(9, 16)]).eval()

        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.slice1[0].weight.device != pred.device:
            self.to(pred.device)
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        p_f1 = self.slice1(pred_norm)
        t_f1 = self.slice1(target_norm)
        p_f2 = self.slice2(p_f1)
        t_f2 = self.slice2(t_f1)

        loss = nn.functional.l1_loss(p_f1, t_f1) + nn.functional.l1_loss(p_f2, t_f2)
        return loss


class SobelEdgeLoss(nn.Module):
    """Sobel edge loss to enforce sharp letter boundaries and prevent blurred characters."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sobel_x.device != pred.device:
            self.to(pred.device)

        # Convert to grayscale via luminance weights
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        gx_p = nn.functional.conv2d(pred_gray, self.sobel_x, padding=1)
        gy_p = nn.functional.conv2d(pred_gray, self.sobel_y, padding=1)
        gx_t = nn.functional.conv2d(target_gray, self.sobel_x, padding=1)
        gy_t = nn.functional.conv2d(target_gray, self.sobel_y, padding=1)

        edge_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-6)
        edge_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-6)

        return nn.functional.l1_loss(edge_p, edge_t)


class cGANLoss(nn.Module):
    """
    Combined Composite Loss for Conditional GAN:
    L_total = λ_GAN * L_adv + λ_L1 * L_L1 + λ_VGG * L_perceptual + λ_edge * L_edge
    """
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device if device is not None else DEVICE
        # Use LSGAN MSE loss for stable gradient flow
        self.criterion_gan = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        self.edge_loss = SobelEdgeLoss().to(self.device)

    def get_discriminator_loss(
        self,
        d_pred_real: torch.Tensor,
        d_pred_fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute LSGAN loss for discriminator."""
        real_labels = torch.ones_like(d_pred_real)
        fake_labels = torch.zeros_like(d_pred_fake)

        loss_real = self.criterion_gan(d_pred_real, real_labels)
        loss_fake = self.criterion_gan(d_pred_fake, fake_labels)

        return 0.5 * (loss_real + loss_fake)

    def get_generator_loss(
        self,
        d_pred_fake: torch.Tensor,
        fake_imgs: torch.Tensor,
        real_imgs: torch.Tensor
    ) -> dict:
        """Compute composite generator loss (Adversarial + L1 + Perceptual + Edge)."""
        real_labels = torch.ones_like(d_pred_fake)

        loss_adv = self.criterion_gan(d_pred_fake, real_labels)
        loss_l1 = self.criterion_l1(fake_imgs, real_imgs)
        loss_vgg = self.perceptual_loss(fake_imgs, real_imgs)
        loss_edge = self.edge_loss(fake_imgs, real_imgs)

        total_loss = (
            LAMBDA_GAN * loss_adv +
            LAMBDA_L1 * loss_l1 +
            LAMBDA_PERCEPTUAL * loss_vgg +
            LAMBDA_EDGE * loss_edge
        )

        return {
            "total_loss": total_loss,
            "adv_loss": loss_adv,
            "l1_loss": loss_l1,
            "vgg_loss": loss_vgg,
            "edge_loss": loss_edge
        }
