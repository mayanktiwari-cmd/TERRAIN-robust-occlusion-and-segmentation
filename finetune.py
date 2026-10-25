"""
TERRAIN fine-tuning entrypoint (fixes the stuck-at-0.185 segmentation model).

Why this is different from segmentation/train.py:
  - Fine-tunes from the existing checkpoint (decoder/head weights kept).
  - Reloads the Swin encoder from timm ImageNet weights (resets a degraded backbone).
  - Freezes the encoder for the first FREEZE_ENCODER_EPOCHS epochs, then unfreezes it
    with a LOW encoder LR (two AdamW parameter groups, ENCODER_LR / DECODER_LR).
  - OneCycleLR (max_lr = DECODER_LR, pct_start = WARMUP_FRACTION) instead of the old
    warmup->cosine that peaked too early and collapsed by epoch 5.
  - pos_weight (config.POS_WEIGHT) on the Focal/BCE term + sparse-tile oversampling
    (dataset.SAMPLE_WEIGHTED_BY_ROAD) so the model stops predicting "road everywhere".
  - LISS-IV now trains too: every sensor uses a per-tile 80/20 split, so val covers
    all three sensors and the model has seen every resolution.
  - AMP (FP16) + gradient accumulation (GRAD_ACCUM) + early stopping (patience 8).

Public API:
    run_finetune(...) -> (model, history)   # callable from the Kaggle notebook
    python finetune.py                      # CLI
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from segmentation.model import SwinUNet  # noqa: E402
from segmentation.loss import TERRAIN_Loss  # noqa: E402
from segmentation.dataset import MultiSensorDataset, make_sensor_balanced_batches  # noqa: E402


# ────────────────────────────────────────────────────────────────
# Weight loading
# ────────────────────────────────────────────────────────────────
def load_checkpoint_into(model, path, device="cpu"):
    """Load a state_dict (tolerates DataParallel 'module.' prefix), strict=False."""
    if not (path and os.path.isfile(path)):
        print(f"[finetune] init ckpt NOT FOUND: {path} (continuing from pretrained encoder)")
        return 0
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[finetune] loaded init ckpt {path} | missing {len(missing)} "
          f"| unexpected {len(unexpected)}")
    return len(sd)


def reload_encoder_from_timm(model):
    """Reset model.encoder to timm ImageNet weights (keeps decoder from the ckpt).

    timm is asked for the same architecture config the encoder was built with
    (img_size=256) but a 3-channel stem, so the relative position embedding is
    interpolated correctly and the pretrained stem is genuinely 3-channel.
    The first conv is 4-channel on our model; the 3-channel ImageNet stem is copied
    into channels 0:3 and channel 3 is seeded with the channel mean.
    """
    import timm
    try:
        pret = timm.create_model(
            config.ENCODER.removeprefix("tu-"),
            pretrained=True,
            img_size=config.IMG_SIZE,
            in_chans=3,
        )
    except Exception:
        print("[finetune] timm download failed (offline/SSL) — falling back to "
              "random encoder init for the reload")
        pret = timm.create_model(
            config.ENCODER.removeprefix("tu-"),
            pretrained=False,
            img_size=config.IMG_SIZE,
            in_chans=3,
        )
    psd = pret.state_dict()
    en = model.encoder
    loaded = skipped = 0
    with torch.no_grad():
        for name, p in en.state_dict().items():
            if name in psd and psd[name].shape == p.shape:
                p.copy_(psd[name].to(p.dtype))
                loaded += 1
            else:
                skipped += 1
        pe = getattr(en, "patch_embed", None)
        if pe is not None and hasattr(pe, "proj"):
            pw = pret.patch_embed.proj.weight
            w = pe.proj.weight
            if pw.shape[1] == 3 and w.shape[1] == config.IN_CHANNELS:
                w[:, :3] = pw.to(w.dtype)
                w[:, 3:] = pw.mean(dim=1, keepdim=True).to(w.dtype)
                print("[finetune] stem: 3-channel ImageNet conv -> 4 channels")
    print(f"[finetune] encoder reloaded from timm: {loaded} layers copied, {skipped} skipped")


def set_encoder_requires_grad(model, value):
    for p in model.encoder.parameters():
        p.requires_grad = value
    print(f"[finetune] encoder trainable = {value}")


# ────────────────────────────────────────────────────────────────
# Train / validate
# ────────────────────────────────────────────────────────────────
def _load_batch(dataset, indices, device):
    images, targets, metas = [], [], []
    for i in indices:
        img, mask, meta = dataset[i]
        images.append(img.unsqueeze(0))
        targets.append(mask.unsqueeze(0))
        metas.append(meta)
    return (torch.cat(images, 0).to(device),
            torch.cat(targets, 0).to(device), metas)


def train_one_epoch(model, dataset, batches, criterion, optimizer, scheduler,
                    device, scaler, grad_accum, grad_clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, n_iters = 0.0, 0
    for step, indices in enumerate(batches):
        images, targets, metas = _load_batch(dataset, indices, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets) / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item() * grad_accum
        n_iters += 1
        if (step + 1) % grad_accum == 0:
            if grad_clip and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
    if n_iters % grad_accum != 0:
        if grad_clip and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    return total_loss / max(1, n_iters)


def validate(model, dataset, batches, device, threshold=None):
    threshold = config.INFER_THRESHOLD if threshold is None else threshold
    model.eval()
    tp = fp = fn = 0.0
    with torch.no_grad():
        for indices in batches:
            images, targets, metas = _load_batch(dataset, indices, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            pred = (torch.sigmoid(logits) > threshold).float()
            t = targets
            tp += (pred * t).sum().item()
            fp += (pred * (1 - t)).sum().item()
            fn += ((1 - pred) * t).sum().item()
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec}


# ────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────
def run_finetune(model=None, train_ds=None, val_ds=None,
                 epochs=None, batch_size=None, device=None,
                 save_path=None, init_ckpt=None, patience=None,
                 freeze_epochs=None, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    save_path = save_path or config.FINETUNE_SAVE_PATH
    init_ckpt = init_ckpt or config.FINETUNE_INIT_CKPT
    patience = patience if patience is not None else config.FINETUNE_PATIENCE
    freeze_epochs = freeze_epochs if freeze_epochs is not None \
        else config.FREEZE_ENCODER_EPOCHS
    grad_accum = config.GRAD_ACCUM
    grad_clip = config.GRAD_CLIP_NORM

    print(f"[finetune] device={device} epochs={epochs} batch={batch_size} "
          f"grad_accum={grad_accum} (eff batch {batch_size * grad_accum}) "
          f"freeze_encoder={freeze_epochs} patience={patience}")

    # Build the model: checkpoint (decoder) -> timm encoder reload -> freeze.
    if model is None:
        model = SwinUNet(condition_on_gsd=False)
    model.to(device)
    load_checkpoint_into(model, init_ckpt, device)
    reload_encoder_from_timm(model)

    # Datasets + balanced batches (per-sensor 80/20 split, all sensors in both).
    if train_ds is None:
        train_ds = MultiSensorDataset(split="train")
    if val_ds is None:
        val_ds = MultiSensorDataset(split="val")
    train_batches = make_sensor_balanced_batches(train_ds, batch_size)
    val_batches = make_sensor_balanced_batches(val_ds, batch_size)
    steps_per_epoch = max(1, math.ceil(len(train_batches) / grad_accum))
    print(f"[finetune] train tiles={len(train_ds)} val tiles={len(val_ds)} "
          f"batches={len(train_batches)} optimizer steps/epoch={steps_per_epoch}")

    # Two parameter groups + OneCycleLR (per-group max_lr).
    enc_params = list(model.encoder.parameters())
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": config.ENCODER_LR},
        {"params": dec_params, "lr": config.DECODER_LR},
    ], weight_decay=1e-4)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.ENCODER_LR, config.DECODER_LR],
        total_steps=total_steps,
        pct_start=config.WARMUP_FRACTION,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Criterion must live on the same device as the model (its buffers — e.g.
    # FocalLoss's pos_weight — would otherwise stay on CPU and crash BCE).
    criterion = TERRAIN_Loss(pos_weight=config.POS_WEIGHT).to(device)
    set_encoder_requires_grad(model, freeze_epochs <= 0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_iou, best_epoch, bad_epochs, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        # Unfreeze the encoder after the freeze window (epoch index >= freeze_epochs).
        if freeze_epochs > 0 and epoch > freeze_epochs:
            set_encoder_requires_grad(model, True)

        train_loss = train_one_epoch(
            model, train_ds, train_batches, criterion, optimizer, scheduler,
            device, scaler, grad_accum, grad_clip)
        v = validate(model, val_ds, val_batches, device)
        enc_lr = scheduler.get_last_lr()[0]
        dec_lr = scheduler.get_last_lr()[1]
        entry = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 5),
            "val_iou": round(v["iou"], 5),
            "val_dice": round(v["dice"], 5),
            "val_precision": round(v["precision"], 5),
            "val_recall": round(v["recall"], 5),
            "encoder_lr": enc_lr,
            "decoder_lr": dec_lr,
        }
        history.append(entry)
        print(f"[E{epoch:02d}] enc_lr={enc_lr:.1e} dec_lr={dec_lr:.1e} "
              f"train_loss={train_loss:.4f} | val IoU={v['iou']:.4f} "
              f"Dice={v['dice']:.4f} P={v['precision']:.4f} R={v['recall']:.4f}")

        if v["iou"] > best_iou:
            best_iou, best_epoch, bad_epochs = v["iou"], epoch, 0
            torch.save(model.state_dict(), save_path)
            with open(os.path.splitext(save_path)[0] + "_best.json", "w") as f:
                json.dump({"epoch": epoch, "val_iou": best_iou, **v}, f, indent=2)
            print(f"          saved best -> {save_path} (IoU={best_iou:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[finetune] early stop at epoch {epoch} (no val IoU gain for "
                      f"{patience} epochs; best={best_iou:.4f} @ E{best_epoch})")
                break

    with open(os.path.join(os.path.dirname(save_path) or ".",
                           "history_finetune.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[finetune] done. best val IoU={best_iou:.4f} @ E{best_epoch} -> {save_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TERRAIN fine-tune (Kaggle).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--freeze", type=int, default=None)
    args = parser.parse_args()
    run_finetune(epochs=args.epochs, batch_size=args.batch, device=(
        torch.device(args.device) if args.device else None),
        save_path=args.save, init_ckpt=args.init, patience=args.patience,
        freeze_epochs=args.freeze)
