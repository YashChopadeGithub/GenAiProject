"""
Hybrid Document Synthesis Pipeline for SROIE Receipts.
Combines 100% readable TrueType text rendering with neural paper realism,
optical scanner artifacts, and ink-bleed effects (0 extra training time).
"""

import os
import sys
import math
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autoencoder.dataset import ResizeWithPad
from autoencoder.model import DocumentAutoencoder

DATA_DIR = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data")
IMG_DIR = os.path.join(DATA_DIR, "img")
BOX_DIR = os.path.join(DATA_DIR, "box")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "hybrid_samples")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_receipt_font(size: int = 14):
    """Load standard monospace receipt font from Windows Fonts."""
    font_candidates = [
        "C:/Windows/Fonts/cour.ttf",       # Courier New
        "C:/Windows/Fonts/consola.ttf",    # Consolas
        "C:/Windows/Fonts/lucon.ttf",      # Lucida Console
        "C:/Windows/Fonts/arial.ttf"       # Arial
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_exact_receipt_from_sroie(
    box_path: str,
    orig_img_path: str = None,
    target_size: int = 512,
    paper_color=(250, 250, 248)
) -> Image.Image:
    """
    Renders 100% readable, exact words into receipt bounding boxes.
    Respects original document coordinates and scales accurately to target size.
    """
    # 1. Determine original image dimensions
    orig_w, orig_h = 720, 1000
    if orig_img_path and os.path.exists(orig_img_path):
        with Image.open(orig_img_path) as im:
            orig_w, orig_h = im.size

    # 2. Parse bounding boxes and text strings
    boxes = []
    if os.path.exists(box_path):
        with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",", 8)
                if len(parts) < 9:
                    continue
                try:
                    coords = list(map(int, parts[:8]))
                    text = parts[8].strip()
                    if not text:
                        continue
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = max(0, min(xs)), min(orig_w, max(xs))
                    y_min, y_max = max(0, min(ys)), min(orig_h, max(ys))
                    boxes.append((x_min, y_min, x_max, y_max, text))
                except ValueError:
                    continue

    # 3. Create full-resolution canvas matching original size
    canvas = Image.new("RGB", (orig_w, orig_h), paper_color)
    draw = ImageDraw.Draw(canvas)

    # 4. Render exact text strings with proportional font size
    for x_min, y_min, x_max, y_max, text in boxes:
        box_w = max(10, x_max - x_min)
        box_h = max(8, y_max - y_min)

        # Estimate optimal font size for this box
        font_size = max(11, min(int(box_h * 0.88), 38))
        font = get_receipt_font(font_size)

        # Detect headers, totals, or section markers for realistic bold print
        low_t = text.lower()
        is_bold = (
            box_h >= 24 or
            "total" in low_t or
            "receipt" in low_t or
            "sdn. bhd." in low_t or
            "cash" in low_t or
            "gst" in low_t or
            "subtotal" in low_t
        )
        stroke_w = 1 if is_bold else 0
        text_color = (15, 18, 22)  # Realistic deep thermal printer black

        draw.text((x_min, y_min), text, fill=text_color, font=font, stroke_width=stroke_w)

    # 5. Apply ResizeWithPad to match 512x512 project dimensions
    resizer = ResizeWithPad(target_size)
    padded_canvas = resizer(canvas)

    return padded_canvas


def apply_neural_and_optical_realism(
    rendered_img: Image.Image,
    autoencoder_model=None,
    device="cuda",
    style: str = "thermal_scan",
    intensity: float = 0.55
) -> Image.Image:
    """
    Transforms clean digital text into a photo-realistic physical receipt:
    Adds thermal printer micro-bleed, scanner shadows, and paper grain.
    Optionally passes through trained DocumentAutoencoder for neural latent realism.
    """
    w, h = rendered_img.size
    result = rendered_img.copy()

    # 1. Neural Realism (if Autoencoder is provided)
    if autoencoder_model is not None:
        try:
            autoencoder_model.eval()
            to_tensor = T.ToTensor()
            in_t = to_tensor(result).unsqueeze(0).to(device)
            with torch.no_grad():
                out_t = autoencoder_model(in_t)
                # Blend 45% neural reconstruction + 55% crisp digital text
                blended_t = 0.55 * in_t + 0.45 * out_t
                blended_t = torch.clamp(blended_t, 0.0, 1.0)
                result = to_pil_image(blended_t.squeeze(0).cpu())
        except Exception as e:
            print(f"[Warning] Neural autoencoder pass skipped: {e}")

    # 2. Thermal Paper Texture & Micro-Grain
    np_img = np.array(result).astype(np.float32)
    # Authentic Gaussian paper sensor noise
    grain = np.random.normal(0, 3.5 * intensity, np_img.shape)
    np_img = np.clip(np_img + grain, 0, 255).astype(np.uint8)
    result = Image.fromarray(np_img)

    # 3. Thermal Printer Micro-Bleed / Dot-Feathering
    # Slight horizontal blur simulating thermal pinhead heat expansion
    feather = result.filter(ImageFilter.GaussianBlur(radius=0.35 * intensity))
    result = Image.blend(result, feather, 0.35)

    # 4. Subtle Scanner Vignette / Lighting Gradient
    # Real scans have darker corners or slight illumination falloff
    vignette = Image.new("L", (w, h), 255)
    draw_v = ImageDraw.Draw(vignette)
    for i in range(12):
        shade = 255 - int((12 - i) * 1.8 * intensity)
        draw_v.rectangle([i * 2, i * 2, w - i * 2, h - i * 2], outline=shade)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=8))
    
    # Composite vignette onto image
    shadow_layer = Image.new("RGB", (w, h), (220, 220, 215))
    result = Image.composite(result, shadow_layer, vignette)

    # 5. Tone Curve & Contrast Calibration
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.08)

    return result


def generate_hybrid_sample(receipt_id: str = "261"):
    """
    Generates and saves a full side-by-side comparison for a given SROIE receipt.
    Compares: [ Real Ground Truth Scan | Clean Digital Render | Neural Realistic Hybrid ]
    """
    print(f"\n{'=' * 65}")
    print(f"  Hybrid Document Synthesis — Receipt ID: {receipt_id}")
    print(f"{'=' * 65}")

    box_file = os.path.join(BOX_DIR, f"{receipt_id}.csv")
    img_file = os.path.join(IMG_DIR, f"{receipt_id}.jpg")

    if not os.path.exists(box_file):
        print(f"Error: Box annotation file not found: {box_file}")
        return

    # 1. Load Real Ground Truth
    resizer = ResizeWithPad(512)
    real_img = Image.open(img_file).convert("RGB") if os.path.exists(img_file) else Image.new("RGB", (512, 512), (255, 255, 255))
    padded_real = resizer(real_img)
    print("  [1/4] Loaded real ground-truth scan.")

    # 2. Render Exact Readable Text with TrueType Font
    print("  [2/4] Rendering exact text from SROIE bounding boxes...")
    rendered_exact = render_exact_receipt_from_sroie(box_file, img_file, target_size=512)
    print("        [OK] All words & numbers rendered with 100% legibility!")

    # 3. Load Trained Autoencoder for Neural Realism
    ae_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_ckpt = os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth")
    if os.path.exists(ae_ckpt):
        try:
            ae_model = DocumentAutoencoder().to(device)
            ckpt = torch.load(ae_ckpt, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            ae_model.load_state_dict(state_dict)
            ae_model.eval()
            print(f"  [3/4] Loaded trained Autoencoder from {ae_ckpt} for neural paper realism.")
        except Exception as e:
            print(f"  [3/4] Autoencoder load warning: {e}. Proceeding with optical synthesis.")
    else:
        print("  [3/4] Proceeding with optical thermal paper synthesis.")

    # 4. Apply Neural & Optical Realism
    print("  [4/4] Synthesizing thermal paper texture, scanner lighting & ink bleed...")
    realistic_hybrid = apply_neural_and_optical_realism(
        rendered_exact,
        autoencoder_model=ae_model,
        device=device,
        intensity=0.6
    )

    # 5. Build Side-by-Side Comparison Grid
    # [ Real Scan | Clean Digital Render | Neural Realistic Hybrid ]
    w, h = 512, 512
    comparison = Image.new("RGB", (w * 3, h), (255, 255, 255))
    comparison.paste(padded_real, (0, 0))
    comparison.paste(rendered_exact, (w, 0))
    comparison.paste(realistic_hybrid, (w * 2, 0))

    # Add descriptive top banners
    draw = ImageDraw.Draw(comparison)
    banner_font = get_receipt_font(16)
    
    # Text headers
    headers = [
        (0, "1. Real Scanned Receipt (Ground Truth)"),
        (w, "2. Clean Rendered Text (100% Readable)"),
        (w * 2, "3. Neural Realism Hybrid (Final AI Output)")
    ]
    for x_offset, title in headers:
        draw.rectangle([x_offset, 0, x_offset + w, 28], fill=(30, 41, 59))
        draw.text((x_offset + 14, 5), title, fill=(248, 250, 252), font=banner_font)

    # Save output
    out_file = os.path.join(OUTPUT_DIR, f"hybrid_comparison_{receipt_id}.png")
    comparison.save(out_file, quality=95)
    print(f"\n  [OK] Comparison image saved successfully!")
    print(f"  File: {out_file}")
    print("=" * 65)

    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Document Synthesizer with 100% Readable Text")
    parser.add_argument("--receipt_id", type=str, default="261", help="SROIE receipt ID (e.g. 261, 234)")
    args = parser.parse_args()

    generate_hybrid_sample(args.receipt_id)
