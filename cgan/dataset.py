"""
Dataset pipeline for Conditional GAN (cGAN) Document Synthesis.
Generates semantic layout condition maps paired with ground-truth document scans.
"""

import os
import re
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as T
import numpy as np

from cgan.config import DATA_DIR, BOX_DIR, KEY_DIR, IMG_SIZE
from autoencoder.dataset import ResizeWithPad


def create_layout_condition_map(image: Image.Image, box_path: str = None, key_path: str = None) -> Image.Image:
    """
    Construct a color-coded semantic layout condition map from bounding boxes and entities.
    Colors represent structural document roles:
      - Blue: Merchant / Company Header
      - Green: Address / Entity Location
      - Purple: Date / Invoice Reference
      - Red: Total / Financial Breakdown
      - Dark Grey: Itemized Lines
    """
    w, h = image.size
    # Start with a crisp light canvas
    cond_img = Image.new("RGB", (w, h), (248, 250, 252))
    draw = ImageDraw.Draw(cond_img)

    boxes_found = False

    # 1. Parse Key Entities if available
    entity_texts = {}
    if key_path and os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8", errors="ignore") as f:
                entity_texts = json.load(f)
        except Exception:
            pass

    company_val = str(entity_texts.get("company", "")).strip().lower()
    address_val = str(entity_texts.get("address", "")).strip().lower()
    date_val = str(entity_texts.get("date", "")).strip().lower()
    total_val = str(entity_texts.get("total", "")).strip().lower()

    # 2. Parse box file
    if box_path and os.path.exists(box_path):
        try:
            with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split(",", 8)
                    if len(parts) < 9:
                        continue
                    try:
                        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])
                    except ValueError:
                        continue
                    text = parts[8].strip()
                    if not text:
                        continue

                    boxes_found = True
                    min_x = max(0, min(x1, x2, x3, x4))
                    max_x = min(w, max(x1, x2, x3, x4))
                    min_y = max(0, min(y1, y2, y3, y4))
                    max_y = min(h, max(y1, y2, y3, y4))

                    low_t = text.lower()
                    # Assign semantic color based on entity classification
                    if company_val and (low_t in company_val or company_val in low_t):
                        fill_color = (14, 116, 144)      # Cyan / Corporate Blue
                    elif total_val and (low_t in total_val or "total" in low_t):
                        fill_color = (220, 38, 38)       # Red / Financial
                    elif date_val and (low_t in date_val or re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', low_t)):
                        fill_color = (124, 58, 237)      # Purple / Date
                    elif address_val and (low_t in address_val):
                        fill_color = (22, 163, 74)       # Green / Address
                    else:
                        fill_color = (71, 85, 105)       # Slate / Item text

                    # Draw filled semantic box with subtle outline
                    draw.rectangle([min_x, min_y, max_x, max_y], fill=fill_color, outline=(30, 41, 59), width=1)
        except Exception:
            pass

    # 3. Fallback: If no box file exists, generate structural condition map via thresholding
    if not boxes_found:
        gray = image.convert("L")
        np_img = np.array(gray)
        # Threshold dark text on light background
        mask = (np_img < 130).astype(np.uint8) * 255
        
        # Colorize layout features
        cond_rgb = np.full((h, w, 3), 248, dtype=np.uint8)
        cond_rgb[mask > 0] = [51, 65, 85]
        
        # Accent top regions as headers
        top_h = int(h * 0.25)
        cond_rgb[:top_h, :][mask[:top_h, :] > 0] = [14, 116, 144]
        
        # Accent bottom regions as totals
        bot_h = int(h * 0.75)
        cond_rgb[bot_h:, :][mask[bot_h:, :] > 0] = [220, 38, 38]
        
        cond_img = Image.fromarray(cond_rgb)

    return cond_img


class cGANDocumentDataset(Dataset):
    """
    Loads document images paired with their semantic layout condition maps.
    Outputs: (real_tensor, condition_tensor, filename)
    Includes in-memory layout condition map caching to eliminate CPU parsing bottlenecks.
    """
    def __init__(self, data_dir: str = DATA_DIR, box_dir: str = BOX_DIR, key_dir: str = KEY_DIR, img_size: int = IMG_SIZE):
        self.data_dir = data_dir
        self.box_dir = box_dir
        self.key_dir = key_dir
        self.img_size = img_size

        self.resize_pad = ResizeWithPad(img_size)
        self.to_tensor = T.ToTensor()
        self._cond_cache = {}  # In-memory condition map cache (stored as compact uint8 to save RAM)

        # Find all valid image files
        self.filenames = []
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.filenames.append(f)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        bname = os.path.splitext(fname)[0]
        fpath = os.path.join(self.data_dir, fname)

        # Load real image
        real_img = Image.open(fpath).convert("RGB")
        padded_real = self.resize_pad(real_img)
        real_tensor = self.to_tensor(padded_real)

        # Fetch or compute cached condition layout map
        if idx in self._cond_cache:
            cond_tensor = self._cond_cache[idx].float() / 255.0
        else:
            box_path = os.path.join(self.box_dir, f"{bname}.csv") if self.box_dir else None
            key_path = os.path.join(self.key_dir, f"{bname}.json") if self.key_dir else None

            # Build condition layout map
            cond_img = create_layout_condition_map(real_img, box_path, key_path)
            padded_cond = self.resize_pad(cond_img)
            cond_tensor = self.to_tensor(padded_cond)

            # Cache compact uint8 tensor (takes < 490MB for entire dataset of ~626 images)
            self._cond_cache[idx] = (cond_tensor * 255.0).to(torch.uint8)

        return real_tensor, cond_tensor, fname
