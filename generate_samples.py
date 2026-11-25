import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autoencoder.model import DocumentVAE
from autoencoder.dataset import get_dataloaders
from autoencoder.config import CHECKPOINT_DIR, DEVICE, OUTPUT_DIR

def generate_samples(num_samples=3):
    print("Loading VAE model...")
    model = DocumentVAE().to(DEVICE)
    best_ckpt = os.path.join(CHECKPOINT_DIR, 'best_vae.pth')
    
    if not os.path.exists(best_ckpt):
        print(f"Checkpoint not found at {best_ckpt}. Please train the model first.")
        return

    # Load weights
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded.")
    print("Fetching a validation image...")
    _, val_loader = get_dataloaders()
    inputs, targets = next(iter(val_loader))
    
    # Take the first image from the batch
    single_img = inputs[0:1].to(DEVICE)

    samples = []
    print(f"Generating {num_samples} samples from the given image latent space...")
    with torch.no_grad():
        # Using forward with sample_mode=True to ensure pure latent space generation
        # without being forced to replicate exactly due to the skip connections.
        
        # Alternatively, we can use encode_vae if we want to manually reparameterize:
        mu, log_var, skips = model.encode_vae(single_img)
        
        # Sample multiple times from the latent distribution
        for i in range(num_samples):
            z = model.reparameterize(mu, log_var)
            
            # Note: since the VAE uses skip connections from the original image,
            # we need to zero them out, otherwise it will just identically reconstruct 
            # the original image instead of generating variations.
            zeroed_skips = tuple(torch.zeros_like(s) for s in skips)
            
            out = model.decode(z, zeroed_skips)
            samples.append(out.cpu())

    print("Saving visualization...")
    all_imgs = torch.cat([single_img.cpu()] + samples, dim=0)
    grid = vutils.make_grid(all_imgs, nrow=num_samples + 1, padding=4, pad_value=1.0)
    
    plt.figure(figsize=(5 * (num_samples + 1), 5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    
    titles = ["Original Input"] + [f"Generated Sample {i+1}" for i in range(num_samples)]
    
    plt.title(" | ".join(titles), fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'VAEOutputs', 'generated_samples.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Done! Saved grid to {output_path}")


def generate_cgan_samples(styles=None, noise_level=0.35):
    """Generate multi-style synthetic document samples using cGAN."""
    from PIL import Image
    from cgan.synthesizer import DocumentSynthesizer
    from cgan.dataset import cGANDocumentDataset

    print("=" * 60)
    print("Generating cGAN Synthetic Document Samples")
    print("=" * 60)

    synthesizer = DocumentSynthesizer()
    styles = styles or ["clean", "thermal_fade", "stamped", "fax_noise", "crumpled_scan"]

    # Load a test document image
    sample_path = os.path.join(PROJECT_ROOT, "static", "samples", "210.jpg")
    box_path = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "box", "210.csv")
    key_path = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "key", "210.json")

    if not os.path.exists(sample_path):
        ds = cGANDocumentDataset()
        sample_path = os.path.join(ds.data_dir, ds.filenames[0])
        bname = os.path.splitext(ds.filenames[0])[0]
        box_path = os.path.join(ds.box_dir, f"{bname}.csv")
        key_path = os.path.join(ds.key_dir, f"{bname}.json")

    img = Image.open(sample_path).convert("RGB")
    print(f"Using document: {sample_path}")

    # Generate samples across all styles
    fig, axes = plt.subplots(1, len(styles) + 1, figsize=(4.5 * (len(styles) + 1), 6))

    # Condition map from synthesizer
    first_res = synthesizer.synthesize(img, box_path=box_path, key_path=key_path, style="clean", noise_level=noise_level)
    
    # Load condition and original
    from cgan.dataset import create_layout_condition_map
    cond_map = create_layout_condition_map(img, box_path, key_path)
    axes[0].imshow(cond_map)
    axes[0].set_title("Condition Layout Mask\n[Semantic Entities]", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for idx, st in enumerate(styles):
        res = synthesizer.synthesize(img, box_path=box_path, key_path=key_path, style=st, noise_level=noise_level)
        m = res["metrics"]
        synth_img = synthesizer._apply_style_postprocessing(
            synthesizer.resize_pad(img), st, noise_level
        )
        axes[idx + 1].imshow(synth_img)
        title = f"cGAN: {st.replace('_', ' ').title()}\nSSIM: {m['ssim']} | OCR: {m['ocr_readability']}%"
        axes[idx + 1].set_title(title, fontsize=10, fontweight="bold")
        axes[idx + 1].axis("off")

    plt.tight_layout()
    cgan_out = os.path.join(OUTPUT_DIR, "cgan_samples", "generated_cgan_samples.png")
    os.makedirs(os.path.dirname(cgan_out), exist_ok=True)
    plt.savefig(cgan_out, bbox_inches="tight", dpi=150)
    print(f"Done! Saved cGAN comparison grid to: {cgan_out}")


def generate_diffusion_samples(receipt_id="261", cfg_scale=3.5, steps=50):
    """Generate layout-conditioned synthetic document samples using DDPM/DDIM."""
    from diffusion.synthesizer import synthesize_from_receipt
    print("=" * 60)
    print("Generating Diffusion Model Synthetic Document Samples")
    print("=" * 60)
    out_path = synthesize_from_receipt(
        receipt_id=receipt_id,
        cfg_scale=cfg_scale,
        sampling_steps=steps,
        save_comparison=True
    )
    print(f"Done! Saved Diffusion comparison to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate generative AI document samples.")
    parser.add_argument("--model", type=str, default="all", choices=["vae", "cgan", "diffusion", "all"],
                        help="Which generative model to sample from (default: all)")
    parser.add_argument("--samples", type=int, default=3, help="Number of VAE samples")
    parser.add_argument("--noise", type=float, default=0.35, help="cGAN latent noise temperature")
    parser.add_argument("--receipt_id", type=str, default="261", help="Receipt ID for diffusion synthesis")
    parser.add_argument("--cfg_scale", type=float, default=3.5, help="Diffusion CFG guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion DDIM sampling steps")
    args = parser.parse_args()

    if args.model in ("vae", "all"):
        generate_samples(args.samples)
    if args.model in ("cgan", "all"):
        generate_cgan_samples(noise_level=args.noise)
    if args.model in ("diffusion", "all"):
        generate_diffusion_samples(receipt_id=args.receipt_id, cfg_scale=args.cfg_scale, steps=args.steps)

