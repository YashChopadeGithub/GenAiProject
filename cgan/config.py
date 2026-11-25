"""
Centralized configuration for the Conditional GAN (cGAN) document synthesis pipeline.
Defines hyperparameters for generator, discriminator, dataset, and loss functions.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "img")
BOX_DIR = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "box")
KEY_DIR = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "key")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Image & Condition Dimensions ───────────────────────────────────────────
IMG_SIZE = 512            # Resolution 512x512
IMG_CHANNELS = 3          # Output RGB document
COND_CHANNELS = 3         # Semantic condition map (RGB layout mask: Headers, Items, Total, Date)
LATENT_DIM = 128          # Noise vector size injected into generator

# ─── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 4            # Safe batch size for 4GB-8GB VRAM
NUM_WORKERS = 0           # Avoid Windows IPC multiprocessing overhead
LEARNING_RATE_G = 2e-4    # Adam optimizer learning rate for Generator
LEARNING_RATE_D = 2e-4    # Adam optimizer learning rate for Discriminator
BETA1 = 0.5               # Adam beta1 (Pix2Pix standard)
BETA2 = 0.999             # Adam beta2
EPOCHS = 30
VAL_RATIO = 0.2
RANDOM_SEED = 42

# ─── Loss Weights ────────────────────────────────────────────────────────────
LAMBDA_GAN = 1.0          # Adversarial loss weight
LAMBDA_L1 = 100.0         # Pix2Pix L1 pixel reconstruction loss weight
LAMBDA_PERCEPTUAL = 5.0   # VGG feature loss for crisp letterforms
LAMBDA_EDGE = 10.0        # Edge loss to enforce sharp typography

# ─── Checkpoint Names ────────────────────────────────────────────────────────
GEN_CKPT_NAME = "best_cgan_generator.pth"
DISC_CKPT_NAME = "best_cgan_discriminator.pth"
GEN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, GEN_CKPT_NAME)
DISC_CKPT_PATH = os.path.join(CHECKPOINT_DIR, DISC_CKPT_NAME)

LATEST_GEN_CKPT_NAME = "latest_cgan_generator.pth"
LATEST_DISC_CKPT_NAME = "latest_cgan_discriminator.pth"
LATEST_GEN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, LATEST_GEN_CKPT_NAME)
LATEST_DISC_CKPT_PATH = os.path.join(CHECKPOINT_DIR, LATEST_DISC_CKPT_NAME)
