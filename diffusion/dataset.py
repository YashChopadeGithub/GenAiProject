"""
Dataset pipeline for Diffusion Model (DDPM) document synthesis.
Reuses the semantic layout condition map generation from cGAN dataset.
Outputs: (real_tensor_normalized, condition_tensor) in [-1, 1] range.

NOTE: Diffusion models require images normalized to [-1, 1] (not [0, 1]).
This is different from the cGAN and autoencoder pipelines that use [0, 1].
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from diffusion.config import DATA_DIR, BOX_DIR, KEY_DIR, IMG_SIZE
from autoencoder.dataset import ResizeWithPad
from cgan.dataset import create_layout_condition_map


class DiffusionDocumentDataset(Dataset):
    """
    SROIE receipt dataset for DDPM training.

    Returns:
        x_0:       (3, H, W) clean real image normalized to [-1, 1]
        condition: (3, H, W) semantic layout map normalized to [0, 1]
        fname:     filename string

    The [-1, 1] normalization is required for diffusion models because:
      - Pure noise x_T ~ N(0, I) has mean=0, range roughly [-3, 3]
      - Matching the signal range avoids training instability
    """
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        box_dir:  str = BOX_DIR,
        key_dir:  str = KEY_DIR,
        img_size: int = IMG_SIZE,
    ):
        self.data_dir = data_dir
        self.box_dir  = box_dir
        self.key_dir  = key_dir
        self.img_size = img_size

        self.resize_pad = ResizeWithPad(img_size)

        # [0, 1] -> [-1, 1] transform for real images (DDPM standard)
        self.to_tensor_11 = T.Compose([
            T.ToTensor(),                       # [0, 255] -> [0, 1]
            T.Normalize(mean=[0.5, 0.5, 0.5],  # [0, 1] -> [-1, 1]
                        std=[0.5, 0.5, 0.5])
        ])

        # [0, 1] for condition map (used as input, not noise target)
        self.to_tensor_01 = T.ToTensor()

        # In-memory condition map cache
        self._cond_cache: dict = {}

        # Find all image files
        self.filenames = []
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.filenames.append(f)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname  = self.filenames[idx]
        bname  = os.path.splitext(fname)[0]
        fpath  = os.path.join(self.data_dir, fname)

        # Load and resize real receipt image
        real_img  = Image.open(fpath).convert("RGB")
        padded    = self.resize_pad(real_img)

        # Normalize to [-1, 1] for DDPM noise matching
        x_0 = self.to_tensor_11(padded)

        # Build or load cached condition map
        if idx in self._cond_cache:
            cond_tensor = self._cond_cache[idx].float() / 255.0
        else:
            box_path = os.path.join(self.box_dir, f"{bname}.csv") if self.box_dir else None
            key_path = os.path.join(self.key_dir, f"{bname}.json") if self.key_dir else None

            cond_img     = create_layout_condition_map(real_img, box_path, key_path)
            padded_cond  = self.resize_pad(cond_img)
            cond_tensor  = self.to_tensor_01(padded_cond)

            # Cache as uint8 to save RAM
            self._cond_cache[idx] = (cond_tensor * 255.0).to(torch.uint8)

        return x_0, cond_tensor, fname
