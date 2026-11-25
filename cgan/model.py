"""
Conditional GAN (cGAN) Architecture for Document Synthesis.
Includes:
  1. DocumentGenerator: Layout-conditioned U-Net generator with skip connections and latent style injection.
  2. PatchDiscriminator: 70x70 PatchGAN discriminator evaluating local document texture, font realism, and alignment.
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvDownBlock(nn.Module):
    """Downsampling conv block: Conv4x4(stride=2) -> InstanceNorm -> LeakyReLU"""
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvUpBlock(nn.Module):
    """Upsampling conv block: ConvTranspose4x4(stride=2) -> InstanceNorm -> Dropout(optional) -> ReLU"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.block(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


class DocumentcGANGenerator(nn.Module):
    """
    U-Net Conditional Generator:
    Takes a conditioned document layout mask (B, 3, 512, 512) and an optional
    latent noise vector (B, latent_dim) to synthesize realistic document scans.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Latent projection to bottleneck (2x2 spatial)
        self.latent_fc = nn.Linear(latent_dim, 256 * 2 * 2)

        # ─── Encoder (512x512 down to 2x2) ──────────────────────────
        self.enc1 = ConvDownBlock(in_channels, 64, normalize=False)  # 256x256
        self.enc2 = ConvDownBlock(64, 128)                           # 128x128
        self.enc3 = ConvDownBlock(128, 256)                          # 64x64
        self.enc4 = ConvDownBlock(256, 512)                          # 32x32
        self.enc5 = ConvDownBlock(512, 512)                          # 16x16
        self.enc6 = ConvDownBlock(512, 512)                          # 8x8
        self.enc7 = ConvDownBlock(512, 512)                          # 4x4
        self.enc8 = ConvDownBlock(512, 512, normalize=False)         # 2x2 (bottleneck)

        # ─── Decoder with Skip Connections ──────────────────────────
        # Bottleneck takes enc8 + latent (512 + 256 = 768)
        self.dec1 = ConvUpBlock(512 + 256, 512, dropout=0.5)         # 4x4
        self.dec2 = ConvUpBlock(512 + 512, 512, dropout=0.5)         # 8x8 (skip enc7)
        self.dec3 = ConvUpBlock(512 + 512, 512, dropout=0.5)         # 16x16 (skip enc6)
        self.dec4 = ConvUpBlock(512 + 512, 512, dropout=0.0)         # 32x32 (skip enc5)
        self.dec5 = ConvUpBlock(512 + 512, 256, dropout=0.0)         # 64x64 (skip enc4)
        self.dec6 = ConvUpBlock(256 + 256, 128, dropout=0.0)         # 128x128 (skip enc3)
        self.dec7 = ConvUpBlock(128 + 128, 64, dropout=0.0)          # 256x256 (skip e2)
        
        # Final output layer (256x256 -> 512x512)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Normalizes output image to [0, 1]
        )

    def forward(self, condition: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = condition.size(0)
        if z is None:
            z = torch.randn(B, self.latent_dim, device=condition.device)

        # Encode condition
        e1 = self.enc1(condition)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Project and concatenate latent style vector at bottleneck
        z_feat = self.latent_fc(z).view(B, 256, 2, 2)
        bottleneck = torch.cat([e8, z_feat], dim=1)

        # Decode with skip connections
        d1 = self.dec1(bottleneck, e7)
        d2 = self.dec2(d1, e6)
        d3 = self.dec3(d2, e5)
        d4 = self.dec4(d3, e4)
        d5 = self.dec5(d4, e3)
        d6 = self.dec6(d5, e2)
        d7 = self.dec7(d6, e1)
        out = self.final(d7)

        return out


class DocumentcGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator:
    Evaluates joint pairs of (Image, Condition) to judge whether local document patches
    look authentically photographed/scanned or synthetically fabricated.
    """
    def __init__(self, in_channels: int = 6):
        super().__init__()

        def disc_block(in_f: int, out_f: int, stride: int = 2, normalize: bool = True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(in_channels, 64, stride=2, normalize=False),   # 256x256
            *disc_block(64, 128, stride=2),                            # 128x128
            *disc_block(128, 256, stride=2),                           # 64x64
            *disc_block(256, 512, stride=1),                           # 63x63
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)      # 62x62 patch logits
        )

    def forward(self, img: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Concatenate image and condition along channel dimension
        pair = torch.cat([img, condition], dim=1)
        return self.model(pair)
