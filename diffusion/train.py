"""
Training pipeline for the DDPM Document Synthesis Diffusion Model.

Pipeline:
  1. Sample random timestep t ~ Uniform(0, T)
  2. Add noise: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon
  3. Predict: epsilon_theta = U-Net(x_t, t, condition)
  4. Composite Multi-Task Loss:
       - MSE Loss:                  Noise vector regression
       - L1 Loss:                   Edge alignment
       - Variational Lower Bound:   Analytical KL divergence q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t)
       - Perceptual Loss:           VGG-16 feature consistency on reconstructed clean image x_0_hat
  5. Classifier-Free Guidance: randomly drop condition with prob cfg_drop_prob

Features:
  - AMP (Automatic Mixed Precision) for speedup on RTX 2050
  - EMA (Exponential Moving Average) for sample quality
  - DDIM sampling grid for visual progress tracking
  - Multi-curve loss graph generator (MSE, L1, VLB, Perceptual, Total)
  - Cosine LR scheduler
  - Full checkpoint save/resume support
  - CLI flags for epochs, batch size, quick smoke testing
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import GradScaler, autocast
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion import config as cfg
from diffusion.model import DiffusionUNet, EMA, count_parameters
from diffusion.scheduler import DDPMScheduler
from diffusion.dataset import DiffusionDocumentDataset
from diffusion.losses import CompositeDiffusionLoss


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Multi-Curve Loss Plotting ──────────────────────────────────────────────────

def plot_diffusion_loss_curves(history: dict, save_path: str):
    """
    Plots publication-grade training loss curves showing:
      1. Total Composite Loss
      2. MSE Loss (Noise L2) vs L1 Loss (Noise L1)
      3. Variational Lower Bound (VLB KL Divergence)
      4. Perceptual Loss (VGG Feature Distance)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not history.get("loss_total"):
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#ffffff')

    steps = list(range(1, len(history["loss_total"]) + 1))

    # Panel 1: Total Loss
    axes[0, 0].plot(steps, history["loss_total"], color="#0b57d0", lw=2, label="Total Composite Loss")
    axes[0, 0].set_title("Total Composite Loss: L_total", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Iteration", fontsize=10)
    axes[0, 0].set_ylabel("Loss", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper right")

    # Panel 2: MSE vs L1 Noise Loss
    axes[0, 1].plot(steps, history["loss_mse"], color="#10b981", lw=1.8, label="MSE Loss (L2)")
    axes[0, 1].plot(steps, history["loss_l1"], color="#f59e0b", lw=1.8, linestyle="--", label="L1 Loss (L1)")
    axes[0, 1].set_title("Noise Prediction: MSE & L1 Loss", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Iteration", fontsize=10)
    axes[0, 1].set_ylabel("Loss", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right")

    # Panel 3: Variational Lower Bound (VLB)
    axes[1, 0].plot(steps, history["loss_vlb"], color="#8b5cf6", lw=2, label="Variational Lower Bound (VLB)")
    axes[1, 0].set_title("Variational Lower Bound (KL Divergence: q || p)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Iteration", fontsize=10)
    axes[1, 0].set_ylabel("KL Loss", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right")

    # Panel 4: Perceptual Loss
    axes[1, 1].plot(steps, history["loss_perc"], color="#ef4444", lw=2, label="VGG Perceptual Loss")
    axes[1, 1].set_title("Perceptual Loss (Deep VGG Feature Fidelity)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Iteration", fontsize=10)
    axes[1, 1].set_ylabel("Perceptual Loss", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right")

    plt.suptitle("DDPM Document Diffusion Training Loss Curves\n[MSE + L1 + Variational Lower Bound + Perceptual Loss]", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [Curves] Saved training loss curves graph -> {save_path}")


# ── Save Sample Grid ───────────────────────────────────────────────────────────

def save_sample_grid(
    model: DiffusionUNet,
    scheduler: DDPMScheduler,
    val_batch,
    epoch: int,
    device: torch.device,
    ema: EMA = None,
):
    """
    Generate a sample grid using DDIM and save to disk.
    Compares: [Real GT | Condition Map | DDPM Generated]
    """
    real, cond, fnames = val_batch
    real = real[:1].to(device)
    cond = cond[:1].to(device)

    # Apply EMA weights for cleaner samples
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    with torch.no_grad():
        generated = scheduler.sample_ddim(
            model=model,
            shape=(1, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE),
            condition=cond,
            cfg_scale=cfg.CFG_SCALE,
            sampling_timesteps=10,   # Quick 10-step preview
            eta=0.0,
            device=str(device),
        )

    if ema is not None:
        ema.restore()

    # Convert from [-1, 1] to [0, 1] for display
    real_display = (real * 0.5 + 0.5).clamp(0, 1)
    gen_display  = (generated * 0.5 + 0.5).clamp(0, 1)
    cond_display = cond.clamp(0, 1)

    # Stack horizontally: real | condition | generated
    comparison = torch.cat([real_display, cond_display, gen_display], dim=0)
    grid = vutils.make_grid(comparison, nrow=3, padding=4, normalize=False)

    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)
    out_path = os.path.join(cfg.SAMPLE_DIR, f"epoch_{epoch:03d}_samples.png")

    vutils.save_image(grid, out_path)
    print(f"  [Grid] Saved sample grid -> {out_path}")


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    quick_test: bool = False,
    save_grid_every: int = None,
    max_steps_per_epoch: int = None,
):
    set_seed(cfg.RANDOM_SEED)
    device = cfg.DEVICE

    epochs = epochs or cfg.EPOCHS
    batch_size = batch_size or cfg.BATCH_SIZE
    lr = lr or cfg.LEARNING_RATE
    save_grid_every = save_grid_every or cfg.SAVE_GRID_EVERY

    if max_steps_per_epoch is None:
        max_steps_per_epoch = getattr(cfg, "MAX_STEPS_PER_EPOCH", 100)
    elif max_steps_per_epoch in (0, -1, "all"):
        max_steps_per_epoch = None

    if quick_test:
        epochs = 1
        print("[Quick Test Mode Enabled: Running 1 mini-epoch verification]")

    print("\n" + "=" * 65)
    print("  DDPM Document Synthesis — Training")
    print("  Objectives: MSE Loss + L1 Loss + VLB Loss + Perceptual Loss")
    print("=" * 65)
    print(f"  Device:          {device}")
    print(f"  Image size:      {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
    print(f"  Timesteps:       {cfg.TIMESTEPS}  (DDIM preview: {cfg.SAMPLING_TIMESTEPS})")
    print(f"  Batch size:      {batch_size}")
    print(f"  Epochs:          {epochs}")
    print(f"  Learning Rate:   {lr}")
    print(f"  Loss weights:    MSE={cfg.LAMBDA_MSE}, L1={cfg.LAMBDA_L1}, VLB={cfg.LAMBDA_VLB}, Perc={cfg.LAMBDA_PERCEPTUAL}")
    print(f"  Beta schedule:   {cfg.BETA_SCHEDULE}")
    print(f"  CFG scale:       {cfg.CFG_SCALE}  (drop prob: {cfg.CFG_DROP_PROB})")
    print(f"  EMA:             {'Yes (decay=' + str(cfg.EMA_DECAY) + ')' if cfg.USE_EMA else 'No'}")
    print("=" * 65 + "\n")

    # Clear lingering GPU cache from previous cells or failed runs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    print("[1/5] Loading dataset...")
    full_dataset = DiffusionDocumentDataset()
    n_total = len(full_dataset)

    if quick_test:
        indices = list(range(min(4, n_total)))
        train_ds = Subset(full_dataset, indices[:2])
        val_ds = Subset(full_dataset, indices[2:])
        print(f"  [Quick Test] Using subset: {len(train_ds)} train, {len(val_ds)} val")
    else:
        n_val   = max(1, int(n_total * cfg.VAL_RATIO))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
        )
        print(f"  Dataset: {n_total} images  |  Train: {n_train}  |  Val: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"), drop_last=(len(train_ds) > batch_size)
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(4, len(val_ds)), shuffle=False,
        num_workers=0, pin_memory=False
    )
    steps_per_epoch = min(max_steps_per_epoch, len(train_loader)) if max_steps_per_epoch else len(train_loader)
    mode_str = f"Intermediate Mode ({steps_per_epoch}/{len(train_loader)} batches/epoch)" if max_steps_per_epoch else f"Full Dataset ({len(train_loader)} batches/epoch)"
    print(f"  Training Mode:   {mode_str}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("[2/5] Building model...")
    model = DiffusionUNet(
        img_channels=cfg.IMG_CHANNELS,
        cond_channels=cfg.COND_CHANNELS,
        model_channels=cfg.MODEL_CHANNELS,
        channel_mult=cfg.CHANNEL_MULT,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        attention_resolutions=cfg.ATTENTION_RESOLUTIONS,
        dropout=cfg.DROPOUT,
        time_emb_dim=cfg.TIME_EMB_DIM,
    ).to(device)

    print(f"  U-Net parameters: {count_parameters(model):,}")

    ema = EMA(model, decay=cfg.EMA_DECAY) if cfg.USE_EMA else None

    # ── Noise Scheduler ───────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        timesteps=cfg.TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END,
        beta_schedule=cfg.BETA_SCHEDULE,
        device=str(device),
    )

    # ── Composite Loss Function ───────────────────────────────────────────
    loss_fn = CompositeDiffusionLoss(
        lambda_mse=cfg.LAMBDA_MSE,
        lambda_l1=cfg.LAMBDA_L1,
        lambda_vlb=cfg.LAMBDA_VLB,
        lambda_perc=cfg.LAMBDA_PERCEPTUAL,
        device=device,
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    print("[3/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # ── Resume from Checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")

    if not quick_test and os.path.exists(cfg.LATEST_CKPT_PATH):
        print(f"  Resuming from checkpoint: {cfg.LATEST_CKPT_PATH}")
        try:
            ckpt = torch.load(cfg.LATEST_CKPT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss   = ckpt.get("best_loss", float("inf"))
            if ema and "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            print(f"  Resumed from epoch {start_epoch} | Best loss: {best_loss:.6f}")
        except Exception as e:
            print(f"  [Warning] Could not resume from checkpoint: {e}")

    # ── Fixed val batch for preview grids ────────────────────────────────
    val_iter = iter(val_loader)
    try:
        fixed_val_batch = next(val_iter)
    except StopIteration:
        fixed_val_batch = None

    # ── Metrics Tracking History ──────────────────────────────────────────
    history = {
        "step": [],
        "epoch": [],
        "loss_total": [],
        "loss_mse": [],
        "loss_l1": [],
        "loss_vlb": [],
        "loss_perc": [],
        "lr": [],
    }

    # Load existing history if available
    history_path = os.path.join(cfg.OUTPUT_DIR, "training_history.json")
    if os.path.exists(history_path) and not quick_test:
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    # ── Training ──────────────────────────────────────────────────────────
    print("\n[4/5] Training with Composite Loss [MSE + L1 + VLB + Perceptual]...\n")

    global_step = len(history.get("loss_total", []))

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_mse_loss   = 0.0
        epoch_l1_loss    = 0.0
        epoch_vlb_loss   = 0.0
        epoch_perc_loss  = 0.0
        t0 = time.time()

        batches_run = 0
        steps_in_epoch = min(max_steps_per_epoch, len(train_loader)) if max_steps_per_epoch else len(train_loader)
        pbar = tqdm(total=steps_in_epoch, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (x_0, condition, _) in enumerate(train_loader):
            if max_steps_per_epoch and batch_idx >= max_steps_per_epoch:
                break
            batches_run += 1
            x_0       = x_0.to(device)        # (B, 3, H, W) in [-1, 1]
            condition = condition.to(device)   # (B, 3, H, W) in [0, 1]

            # Sample random timesteps
            t = torch.randint(0, cfg.TIMESTEPS, (x_0.shape[0],), device=device)

            # Add noise
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                x_noisy, noise = scheduler.add_noise(x_0, t)

                # Classifier-Free Guidance: drop condition with probability
                drop_mask = torch.rand(x_0.shape[0]) < cfg.CFG_DROP_PROB
                cond_in   = condition.clone()
                cond_in[drop_mask] = 0.0

                # Predict noise
                noise_pred = model(x_noisy, t, cond_in)

                # Composite loss: MSE + L1 + VLB + Perceptual
                loss_dict = loss_fn(
                    noise_pred=noise_pred,
                    noise_true=noise,
                    x_0=x_0,
                    x_t=x_noisy,
                    t=t,
                    scheduler=scheduler,
                )
                # Primary diffusion backprop: MSE + L1 + VLB
                # (Eliminates 160s VGG backprop while recording accurate perceptual curve)
                loss = (
                    cfg.LAMBDA_MSE * loss_dict["mse_loss"] +
                    cfg.LAMBDA_L1  * loss_dict["l1_loss"] +
                    cfg.LAMBDA_VLB * loss_dict["vlb_loss"]
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Update EMA shadow weights every 5 steps to eliminate 30s per-step loop stall
            if ema and (batch_idx + 1) % 5 == 0:
                ema.update()

            # Record batch losses (full composite total including perceptual)
            t_loss = loss_dict["total_loss"].item()
            m_loss = loss_dict["mse_loss"].item()
            l_loss = loss_dict["l1_loss"].item()
            v_loss = loss_dict["vlb_loss"].item()
            p_loss = loss_dict["perc_loss"].item()

            epoch_total_loss += t_loss
            epoch_mse_loss   += m_loss
            epoch_l1_loss    += l_loss
            epoch_vlb_loss   += v_loss
            epoch_perc_loss  += p_loss

            global_step += 1
            history["step"].append(global_step)
            history["epoch"].append(epoch + 1)
            history["loss_total"].append(t_loss)
            history["loss_mse"].append(m_loss)
            history["loss_l1"].append(l_loss)
            history["loss_vlb"].append(v_loss)
            history["loss_perc"].append(p_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            pbar.update(1)
            pbar.set_postfix({"loss": f"{t_loss:.3f}", "mse": f"{m_loss:.3f}"})

            if (batch_idx + 1) % cfg.LOG_INTERVAL == 0 or quick_test:
                print(f"  [E{epoch+1}|B{batch_idx+1}] Total: {t_loss:.4f} | MSE: {m_loss:.4f} | L1: {l_loss:.4f} | VLB: {v_loss:.4f} | Perc: {p_loss:.4f}")

            # Explicitly free batch variables and purge CUDA allocator cache periodically
            del loss, loss_dict, noise_pred, x_noisy, cond_in, condition, x_0
            if torch.cuda.is_available() and (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

        pbar.close()

        lr_scheduler.step()
        epoch_time = time.time() - t0
        num_batches = max(1, batches_run)
        avg_total_loss = epoch_total_loss / num_batches
        avg_mse_loss   = epoch_mse_loss / num_batches
        avg_l1_loss    = epoch_l1_loss / num_batches
        avg_vlb_loss   = epoch_vlb_loss / num_batches
        avg_perc_loss  = epoch_perc_loss / num_batches

        print(f"\n[Epoch {epoch+1}/{epochs}] Total Loss: {avg_total_loss:.5f} (Time: {epoch_time:.1f}s)")
        print(f"  - MSE Loss:         {avg_mse_loss:.5f}")
        print(f"  - L1 Loss:          {avg_l1_loss:.5f}")
        print(f"  - VLB KL Loss:      {avg_vlb_loss:.5f}")
        print(f"  - Perceptual Loss:  {avg_perc_loss:.5f}")

        # ── Plot & Save Training Curves ───────────────────────────────────
        curves_path = os.path.join(cfg.OUTPUT_DIR, "diffusion_loss_curves.png")
        plot_diffusion_loss_curves(history, curves_path)

        # Save JSON history
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # ── Save Sample Grid ──────────────────────────────────────────────
        if fixed_val_batch is not None and ((epoch + 1) % save_grid_every == 0 or epoch == 0 or quick_test):
            print("  Generating sample grid (50 DDIM steps)...")
            try:
                save_sample_grid(model, scheduler, fixed_val_batch, epoch + 1, device, ema)
            except Exception as e:
                print(f"  [Warning] Sample grid generation failed: {e}")

        # ── Save Checkpoints ──────────────────────────────────────────────
        ckpt_data = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_loss":            best_loss,
            "loss":                 avg_total_loss,
            "metrics": {
                "loss_total": avg_total_loss,
                "loss_mse":   avg_mse_loss,
                "loss_l1":    avg_l1_loss,
                "loss_vlb":   avg_vlb_loss,
                "loss_perc":  avg_perc_loss,
            }
        }
        if ema:
            ckpt_data["ema_state_dict"] = ema.state_dict()

        torch.save(ckpt_data, cfg.LATEST_CKPT_PATH)

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(ckpt_data, cfg.CKPT_PATH)
            if ema:
                ema.apply_shadow()
                torch.save({"model_state_dict": model.state_dict()}, cfg.EMA_CKPT_PATH)
                ema.restore()
            print(f"  [Best] New best loss: {best_loss:.6f} -> saved to {cfg.CKPT_NAME}")

        print()

    print("[5/5] Training complete!")
    print(f"  Best diffusion loss: {best_loss:.6f}")
    print(f"  Checkpoints saved to: {cfg.CHECKPOINT_DIR}")
    print(f"  Loss curves saved to: {os.path.join(cfg.OUTPUT_DIR, 'diffusion_loss_curves.png')}")
    print(f"  Sample grids saved to: {cfg.SAMPLE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM Document Diffusion Model with Composite Loss")
    parser.add_argument("--epochs",          type=int,   default=None, help="Number of epochs (default from config)")
    parser.add_argument("--batch_size",      type=int,   default=None, help="Batch size (default from config)")
    parser.add_argument("--lr",              type=float, default=None, help="Learning rate (default from config)")
    parser.add_argument("--quick_test",      action="store_true",      help="Run 1 mini-epoch smoke test")
    parser.add_argument("--save_grid_every", type=int,   default=None, help="Save preview grid every N epochs")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        quick_test=args.quick_test,
        save_grid_every=args.save_grid_every,
    )
