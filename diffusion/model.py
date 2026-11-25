"""
Conditioned U-Net Noise Predictor for DDPM Document Synthesis.

A clean, explicit U-Net implementation with:
  - Sinusoidal time embeddings injected into every residual block via FiLM
  - Layout condition map (RGB) concatenated to noisy image at input (6 channels)
  - Classifier-Free Guidance (CFG) support via condition dropout during training
  - Self-attention at low resolutions for global layout coherence
  - EMA (Exponential Moving Average) wrapper for stable high-quality sampling

Input:  noisy_image (B, 3, H, W) + condition (B, 3, H, W)  [concatenated = 6ch]
Output: predicted_noise (B, 3, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint


# ==============================================================================
# 1. Sinusoidal Time Embedding
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Transforms scalar timestep t into a high-dimensional sinusoidal embedding,
    then projects through two linear layers to time_dim*4.
    """
    def __init__(self, base_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.proj = nn.Sequential(
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.base_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)  # (B, base_dim*4)


# ==============================================================================
# 2. Residual Block with Time Embedding (FiLM conditioning)
# ==============================================================================

class ResBlock(nn.Module):
    """
    ResNet block conditioned on time embedding via FiLM (scale + shift).
    GroupNorm + SiLU for training stability.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)  # scale + shift
        )
        self.norm2   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        sc = self.time_proj(t_emb)[:, :, None, None]
        scale, shift = sc.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ==============================================================================
# 3. Self-Attention Block
# ==============================================================================

class AttentionBlock(nn.Module):
    """Multi-head spatial self-attention for global coherence at low resolutions."""
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


# ==============================================================================
# 4. Downsample / Upsample
# ==============================================================================

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ==============================================================================
# 5. U-Net Noise Predictor (Explicit Architecture)
# ==============================================================================

class DiffusionUNet(nn.Module):
    """
    Explicit conditioned U-Net for DDPM document noise prediction.
    Architecture tuned for 512x512 with 4GB VRAM (model_channels=64).

    Encoder:  512->256->128->64->32  (4 downsampling levels)
    Bottleneck: 32x32 with attention
    Decoder:  32->64->128->256->512  (4 upsampling levels with skip connections)
    """
    def __init__(
        self,
        img_channels:  int = 3,
        cond_channels: int = 3,
        model_channels: int = 64,
        channel_mult:  Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (32, 16, 8),
        dropout: float = 0.1,
        time_emb_dim: int = 512,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.img_channels  = img_channels
        self.cond_channels = cond_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing
        in_ch = img_channels + cond_channels  # 6

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(model_channels)
        time_dim = model_channels * 4

        # Determine channel counts per level
        channels = [model_channels * m for m in channel_mult]  # e.g. [64,128,256,512]

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # ── Encoder ───────────────────────────────────────────────────────
        # We build explicit lists of blocks per level for clean forward pass
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_ch = channels[0]
        cur_res = 512

        for i, out_ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(cur_ch, out_ch, time_dim, dropout))
                if cur_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_ch))
                else:
                    level_blocks.append(nn.Identity())
                cur_ch = out_ch
            self.enc_blocks.append(level_blocks)

            if i < len(channels) - 1:
                self.downsamples.append(Downsample(cur_ch))
                cur_res //= 2
            else:
                self.downsamples.append(nn.Identity())  # no downsample at last level

        # ── Bottleneck ────────────────────────────────────────────────────
        bn_ch = channels[-1]
        self.mid1 = ResBlock(bn_ch, bn_ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(bn_ch)
        self.mid2 = ResBlock(bn_ch, bn_ch, time_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────
        self.dec_blocks = nn.ModuleList()
        self.upsamples  = nn.ModuleList()

        # Encoder outputs for skip connections (in reverse order):
        #   Level 3 output: channels[3]
        #   Level 2 output: channels[2]
        #   Level 1 output: channels[1]
        #   Level 0 output: channels[0]
        rev_channels = list(reversed(channels))  # [512,256,128,64]

        cur_ch = channels[-1]  # start from bottleneck

        for i, out_ch in enumerate(rev_channels):
            skip_ch = rev_channels[i]  # skip from corresponding encoder level
            level_blocks = nn.ModuleList()

            # First res-block takes concat of cur + skip
            for j in range(num_res_blocks + 1):
                in_res = (cur_ch + skip_ch) if j == 0 else out_ch
                level_blocks.append(ResBlock(in_res, out_ch, time_dim, dropout))
                if cur_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_ch))
                else:
                    level_blocks.append(nn.Identity())
                cur_ch = out_ch

            self.dec_blocks.append(level_blocks)

            if i < len(rev_channels) - 1:
                self.upsamples.append(Upsample(cur_ch))
                cur_res *= 2
            else:
                self.upsamples.append(nn.Identity())

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, cur_ch), cur_ch),
            nn.SiLU(),
            nn.Conv2d(cur_ch, img_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        cfg_drop: bool = False,
    ) -> torch.Tensor:
        B = x.shape[0]

        # Build condition input
        if condition is None or cfg_drop:
            cond = torch.zeros(B, self.cond_channels, x.shape[2], x.shape[3], device=x.device)
        else:
            cond = condition

        h = torch.cat([x, cond], dim=1)  # (B, 6, H, W)

        # Time embedding
        t_emb = self.time_emb(t)  # (B, time_dim)

        # Input projection
        h = self.input_proj(h)

        def _apply_resblock(block: nn.Module, feat: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
            if self.training and self.use_gradient_checkpointing and feat.requires_grad:
                return checkpoint(block, feat, emb, use_reentrant=False)
            return block(feat, emb)

        # ── Encoder ──────────────────────────────────────────────────────
        skips = []
        for level_blocks, down in zip(self.enc_blocks, self.downsamples):
            for j, block in enumerate(level_blocks):
                if isinstance(block, ResBlock):
                    h = _apply_resblock(block, h, t_emb)
                else:
                    h = block(h)
            skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)

        # ── Bottleneck ────────────────────────────────────────────────────
        h = _apply_resblock(self.mid1, h, t_emb)
        h = self.mid_attn(h)
        h = _apply_resblock(self.mid2, h, t_emb)

        # ── Decoder ──────────────────────────────────────────────────────
        for level_blocks, up in zip(self.dec_blocks, self.upsamples):
            skip = skips.pop()
            first = True
            for j, block in enumerate(level_blocks):
                if isinstance(block, ResBlock):
                    if first:
                        h = torch.cat([h, skip], dim=1)
                        first = False
                    h = _apply_resblock(block, h, t_emb)
                else:
                    h = block(h)
            if not isinstance(up, nn.Identity):
                h = up(h)

        return self.output_proj(h)


# ==============================================================================
# 6. EMA Wrapper
# ==============================================================================

class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy that updates slowly for stable sample quality.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.detach().clone()
                else:
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict):
        raw_shadow = state.get("shadow", {})
        self.shadow = {k: v.detach().clone() for k, v in raw_shadow.items()}
        self.decay  = state.get("decay", self.decay)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    model = DiffusionUNet(model_channels=32, channel_mult=(1,2,4), num_res_blocks=1,
                          attention_resolutions=(8,), dropout=0.0, time_emb_dim=128)
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(1, 3, 64, 64)
    t = torch.tensor([500])
    c = torch.randn(1, 3, 64, 64)
    out = model(x, t, c)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print("PASSED!")
