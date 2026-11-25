"""
Conditional GAN (cGAN) package for generative document synthesis.
"""

from cgan.config import (
    DEVICE,
    IMG_SIZE,
    GEN_CKPT_PATH,
    DISC_CKPT_PATH
)
from cgan.model import DocumentcGANGenerator, DocumentcGANDiscriminator
from cgan.synthesizer import DocumentSynthesizer

__all__ = [
    "DocumentcGANGenerator",
    "DocumentcGANDiscriminator",
    "DocumentSynthesizer",
    "DEVICE",
    "IMG_SIZE",
    "GEN_CKPT_PATH",
    "DISC_CKPT_PATH"
]
