"""
Training script for Conditional GAN (cGAN) Document Synthesis.
Optimizes generator and discriminator with adversarial + L1 + perceptual losses.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from cgan.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE_G,
    LEARNING_RATE_D,
    BETA1,
    BETA2,
    EPOCHS,
    GEN_CKPT_PATH,
    DISC_CKPT_PATH,
    LATEST_GEN_CKPT_PATH,
    LATEST_DISC_CKPT_PATH,
    OUTPUT_DIR,
    LATENT_DIM
)
from cgan.model import DocumentcGANGenerator, DocumentcGANDiscriminator
from cgan.dataset import cGANDocumentDataset
from cgan.losses import cGANLoss


def train_cgan(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    print("=" * 60)
    print("Starting Conditional GAN (cGAN) Document Synthesis Training")
    print(f"Device: {DEVICE} | Epochs: {epochs} | Batch Size: {batch_size}")
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[GPU Acceleration] cuDNN benchmark & TensorFloat-32 (TF32) activated")
        print("[GPU Acceleration] Automatic Mixed Precision (AMP FP16) activated")
    print("=" * 60)

    # 1. Dataset & DataLoader
    dataset = cGANDocumentDataset()
    if len(dataset) == 0:
        print("Error: No document images found in dataset. Check data path.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # 2. Models
    generator = DocumentcGANGenerator().to(DEVICE)
    discriminator = DocumentcGANDiscriminator().to(DEVICE)

    # 3. Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    opt_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

    # 4. Mixed Precision Scalers (AMP)
    use_amp = (DEVICE.type == "cuda")
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5. Composite Loss
    criterion = cGANLoss(device=DEVICE)

    best_g_loss = float("inf")
    start_epoch = 1
    status_file = os.path.join(OUTPUT_DIR, "cgan_training_status.json")

    # Check if a previous checkpoint exists to warm-start / resume
    if os.path.exists(GEN_CKPT_PATH):
        try:
            ckpt = torch.load(GEN_CKPT_PATH, map_location=DEVICE, weights_only=False)
            generator.load_state_dict(ckpt["model_state_dict"])
            best_g_loss = ckpt.get("loss", float("inf"))
            print(f"[Resume] Warm-started generator from best checkpoint (Prior Best Loss: {best_g_loss:.4f})")
            if os.path.exists(DISC_CKPT_PATH):
                d_ckpt = torch.load(DISC_CKPT_PATH, map_location=DEVICE, weights_only=False)
                discriminator.load_state_dict(d_ckpt["model_state_dict"])
        except Exception as e:
            print(f"[Warning] Could not resume checkpoint: {e}")

    try:
        for epoch in range(start_epoch, epochs + 1):
            generator.train()
            discriminator.train()

            total_d_loss = 0.0
            total_g_loss = 0.0
            total_l1_loss = 0.0

            for step, (real_imgs, cond_maps, _) in enumerate(dataloader):
                real_imgs = real_imgs.to(DEVICE, non_blocking=True)
                cond_maps = cond_maps.to(DEVICE, non_blocking=True)
                B = real_imgs.size(0)

                # Sample latent noise
                z = torch.randn(B, LATENT_DIM, device=DEVICE)

                # ─── 1. Forward Generator Once with Mixed Precision ─────
                with torch.amp.autocast("cuda", enabled=use_amp):
                    fake_imgs = generator(cond_maps, z)

                # ─── 2. Train Discriminator ─────────────────────────────
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    d_pred_real = discriminator(real_imgs, cond_maps)
                    d_pred_fake = discriminator(fake_imgs.detach(), cond_maps)
                    d_loss = criterion.get_discriminator_loss(d_pred_real, d_pred_fake)

                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

                # ─── 3. Train Generator (reusing fake_imgs) ──────────────
                opt_g.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    d_pred_fake_for_g = discriminator(fake_imgs, cond_maps)
                    g_loss_dict = criterion.get_generator_loss(d_pred_fake_for_g, fake_imgs, real_imgs)
                    g_loss = g_loss_dict["total_loss"]

                scaler_g.scale(g_loss).backward()
                scaler_g.step(opt_g)
                scaler_g.update()

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                total_l1_loss += g_loss_dict["l1_loss"].item()

            avg_d_loss = total_d_loss / max(len(dataloader), 1)
            avg_g_loss = total_g_loss / max(len(dataloader), 1)
            avg_l1 = total_l1_loss / max(len(dataloader), 1)

            print(f"Epoch [{epoch}/{epochs}] - Loss D: {avg_d_loss:.4f} | Loss G: {avg_g_loss:.4f} | L1: {avg_l1:.4f}")

            # Update training status
            status = {
                "epoch": epoch,
                "total_epochs": epochs,
                "loss_d": round(avg_d_loss, 4),
                "loss_g": round(avg_g_loss, 4),
                "loss_l1": round(avg_l1, 4),
                "best_loss": round(best_g_loss, 4),
                "status": "training" if epoch < epochs else "completed"
            }
            with open(status_file, "w") as f:
                json.dump(status, f, indent=2)

            # Always save latest checkpoints for safe resumption
            torch.save({
                "epoch": epoch,
                "model_state_dict": generator.state_dict(),
                "loss": avg_g_loss
            }, LATEST_GEN_CKPT_PATH)
            torch.save({
                "epoch": epoch,
                "model_state_dict": discriminator.state_dict(),
                "loss": avg_d_loss
            }, LATEST_DISC_CKPT_PATH)

            # Save best generator checkpoint
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "loss": best_g_loss
                }, GEN_CKPT_PATH)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": discriminator.state_dict(),
                    "loss": avg_d_loss
                }, DISC_CKPT_PATH)
                print(f"  [Checkpoint] New best model saved (Loss: {best_g_loss:.4f})")

        print("cGAN training completed successfully!")
        print(f"Checkpoints saved to: {GEN_CKPT_PATH}")

    except (KeyboardInterrupt, Exception) as e:
        print("\n" + "=" * 65)
        if isinstance(e, KeyboardInterrupt):
            print("[!] Training interrupted by user (KeyboardInterrupt / Stopped).")
        else:
            print(f"[!] Training interrupted due to exception: {e}")
        print(f"Restoring best generator weights from: {GEN_CKPT_PATH} ...")
        if os.path.exists(GEN_CKPT_PATH):
            ckpt = torch.load(GEN_CKPT_PATH, map_location=DEVICE, weights_only=False)
            generator.load_state_dict(ckpt["model_state_dict"])
            print(f"[OK] Generator restored to best checkpoint (Epoch {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', 0):.4f})")
        if os.path.exists(DISC_CKPT_PATH):
            d_ckpt = torch.load(DISC_CKPT_PATH, map_location=DEVICE, weights_only=False)
            discriminator.load_state_dict(d_ckpt["model_state_dict"])
            print(f"[OK] Discriminator restored to best checkpoint.")
        print("Model weights safely restored to best saved state.")
        print("=" * 65)
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    curr_status = json.load(f)
                curr_status["status"] = "interrupted_weights_restored"
                with open(status_file, "w") as f:
                    json.dump(curr_status, f, indent=2)
            except Exception:
                pass
        generator.eval()

    return generator, discriminator


if __name__ == "__main__":
    train_cgan(epochs=EPOCHS, batch_size=BATCH_SIZE)
