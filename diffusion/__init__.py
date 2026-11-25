"""
Diffusion Model module for SROIE Document Synthesis.

Components:
  - config.py:     Hyperparameter configuration (DDPM schedule, U-Net arch, training)
  - model.py:      DiffusionUNet (conditioned noise predictor) + EMA wrapper
  - scheduler.py:  DDPMScheduler (forward noise process + DDIM reverse sampling)
  - dataset.py:    DiffusionDocumentDataset ([-1,1] normalized with layout conditions)
  - train.py:      Full DDPM training loop with AMP, EMA, CFG, DDIM sample grids
  - synthesizer.py: Inference script for generating document images
"""

from diffusion.model import DiffusionUNet, EMA
from diffusion.scheduler import DDPMScheduler
from diffusion.dataset import DiffusionDocumentDataset

__all__ = ["DiffusionUNet", "EMA", "DDPMScheduler", "DiffusionDocumentDataset"]
