"""
Centralized configuration for the Diffusion Model (DDPM) document synthesis pipeline.
Implements a Denoising Diffusion Probabilistic Model conditioned on SROIE receipt layouts.
Tuned specifically for NVIDIA RTX 2050 (4GB VRAM) - safe batch sizes and image resolutions.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "img")
BOX_DIR     = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "box")
KEY_DIR     = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "key")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "outputs", "diffusion")
SAMPLE_DIR     = os.path.join(OUTPUT_DIR, "samples")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Settings ---
IMG_SIZE       = 512
IMG_CHANNELS   = 3
COND_CHANNELS  = 3

# --- Diffusion Schedule ---
TIMESTEPS          = 1000
BETA_START         = 1e-4
BETA_END           = 0.02
BETA_SCHEDULE      = "cosine"
SAMPLING_TIMESTEPS = 250
DDIM_ETA           = 0.0

# --- U-Net Architecture ---
MODEL_CHANNELS          = 64
CHANNEL_MULT            = (1, 2, 4, 8)
NUM_RES_BLOCKS          = 2
ATTENTION_RESOLUTIONS   = (32, 16, 8)
DROPOUT                 = 0.1
TIME_EMB_DIM            = 512

# --- Training Hyperparameters ---
BATCH_SIZE          = 1
MAX_STEPS_PER_EPOCH = 100   # Intermediate mode: 100 steps/epoch (4,000 steps total for 40 epochs)

NUM_WORKERS    = 0
LEARNING_RATE  = 2e-4
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
EPOCHS         = 40
VAL_RATIO      = 0.2
RANDOM_SEED    = 42

# --- EMA ---
USE_EMA    = True
EMA_DECAY  = 0.9999

# --- Multi-Objective Loss Weights ---
LOSS_TYPE         = "composite"
LAMBDA_MSE        = 1.0     # L2 Noise Prediction Residual
LAMBDA_L1         = 0.5     # L1 Noise Regularization for crisp edges
LAMBDA_VLB        = 0.05    # Variational Lower Bound (NLL KL Divergence)
LAMBDA_PERCEPTUAL = 0.10    # VGG-16 Deep Feature Perceptual Loss


# --- Classifier-Free Guidance ---
CFG_SCALE     = 3.0
CFG_DROP_PROB = 0.10

# --- Logging & Checkpointing ---
LOG_INTERVAL    = 10
SAVE_GRID_EVERY = 5
CKPT_NAME       = "best_diffusion_unet.pth"
CKPT_PATH       = os.path.join(CHECKPOINT_DIR, CKPT_NAME)
LATEST_CKPT_NAME = "latest_diffusion_unet.pth"
LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, LATEST_CKPT_NAME)
EMA_CKPT_NAME   = "best_diffusion_unet_ema.pth"
EMA_CKPT_PATH   = os.path.join(CHECKPOINT_DIR, EMA_CKPT_NAME)
