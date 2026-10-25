"""
Training loop for the multi-sensor road segmentation model.

Orchestration:
    train_model(...)  -> builds train/val Datasets + sensor-balanced batches,
                         runs epochs, evaluates per-sensor IoU, saves best ckpt.
    train_epoch / validate_epoch -> single-pass helpers over a list of index-batches.

Features (all driven by config):
    - AMP: FP16 autocast + GradScaler (config.AMP) for ~2x speed / 2x memory headroom.
    - Gradient accumulation (config.GRAD_ACCUM) + global clip (config.GRAD_CLIP_NORM).
    - Warmup -> Cosine LR schedule (config.WARMUP_EPOCHS / MIN_LR / LR).
    - Per-loss-component logging (dice/focal/boundary/connectivity) in history.

Uses the sensor-balanced batch generator from dataset.py so Cartosat-3 (47k tiles)
does not drown LISS-IV / Sentinel-2.
"""

import json
import math
import os

import numpy as np
import torch

import config
from segmentation.dataset import MultiSensorDataset, make_sensor_balanced_batches
from segmentation.model import SwinUNet
from segmentation.loss import TERRAIN_Loss


def make_batches(dataset, batch_size=None, n_batches=None):
    return make_sensor_balanced_batches(dataset, batch_size or config.BATCH_SIZE, n_batches)


def _load_batch(dataset, indices, device):
    """Collate a list of dataset indices into (images, masks, metas)."""
    images, targets, metas = [], [], []
    for i in indices:
        img, mask, meta = dataset[i]
        images.append(img.unsqueeze(0))
        targets.append(mask.unsqueeze(0))   # mask is (1,H,W) => add batch dim
        metas.append(meta)
    images = torch.cat(images, 0).to(device)
    targets = torch.cat(targets, 0).to(device)
    return images, targets, metas


def _forward(model, images, metas, device, condition_on_gsd):
    gsd = torch.tensor([m[1] for m in metas], dtype=torch.float32, device=device)
    return model(images, gsd)


def _step_optimizer(optimizer, model, scaler, grad_clip, use_amp):
    """Clip, step, update scaler, zero grads (with AMP unscale)."""
    if grad_clip:
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, dataset, batches, criterion, optimizer, device,
                condition_on_gsd=False, scaler=None, grad_accum=None, grad_clip=None,
                log_components=False, components_every=None):
    """One training epoch. Returns dict of mean loss (+ per-component means when logged)."""
    model.train()
    total_loss, n_iters = 0.0, 0
    grad_accum = grad_accum if grad_accum is not None else config.GRAD_ACCUM
    grad_clip = config.GRAD_CLIP_NORM if grad_clip is None else grad_clip
    use_amp = scaler is not None
    n_batches = len(batches)
    components_every = components_every or max(1, n_batches // 10)
    comps = {"dice": 0.0, "focal": 0.0, "skeleton": 0.0, "connectivity": 0.0}
    n_comps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, indices in enumerate(batches):
        images, targets, metas = _load_batch(dataset, indices, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = _forward(model, images, metas, device, condition_on_gsd)
            loss = criterion(logits, targets) / grad_accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item() * grad_accum
        n_iters += 1

        if log_components and (step + 1) % components_every == 0:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                c = criterion.components(logits.detach(), targets)
            for k in comps:
                comps[k] += c[k]
            n_comps += 1

        if (step + 1) % grad_accum == 0:
            _step_optimizer(optimizer, model, scaler, grad_clip, use_amp)
    if n_iters % grad_accum != 0:
        _step_optimizer(optimizer, model, scaler, grad_clip, use_amp)

    stats = {"loss": total_loss / max(1, n_iters)}
    if log_components and n_comps > 0:
        for k in comps:
            stats[k] = comps[k] / n_comps
    return stats


def validate_epoch(model, dataset, batches, device, condition_on_gsd=False, amp=None):
    """Return (overall_mean_iou, per_sensor_iou_dict)."""
    amp = config.AMP if amp is None else amp
    model.eval()
    inter = {sid: 0.0 for sid in dataset.sensor_ids.values()}
    union = {sid: 0.0 for sid in dataset.sensor_ids.values()}
    with torch.no_grad():
        for indices in batches:
            images, targets, metas = _load_batch(dataset, indices, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = _forward(model, images, metas, device, condition_on_gsd)
            pred = (torch.sigmoid(logits) > 0.5).float()
            for k, m in enumerate(metas):
                sid = m[0]
                p, t = pred[k], targets[k]
                inter[sid] += (p * t).float().sum().item()
                union[sid] += (p.float().sum() + t.float().sum() - (p * t).sum()).item()
    per_sensor = {sid: (inter[sid] + 1e-6) / (union[sid] + 1e-6) for sid in inter}
    overall = float(np.mean(list(per_sensor.values()))) if per_sensor else 0.0
    return overall, per_sensor


def _get_lr(epoch_0based, total_epochs):
    """Linear warmup then cosine decay from LR down to MIN_LR."""
    warm = config.WARMUP_EPOCHS
    if epoch_0based < warm:
        return config.LR * (epoch_0based + 1) / max(1, warm)
    t = min(1.0, max(0.0, (epoch_0based - warm) / max(1, total_epochs - warm)))
    return config.MIN_LR + 0.5 * (config.LR - config.MIN_LR) * (1 + math.cos(math.pi * t))


def train_model(model=None, train_ds=None, val_ds=None, epochs=None, lr=None,
                device=None, batch_size=None, save_path=None,
                condition_on_gsd=None, seed=0, amp=None, grad_accum=None, grad_clip=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.EPOCHS
    lr = lr or config.LR
    save_path = save_path or config.MODEL_SAVE_PATH
    condition_on_gsd = condition_on_gsd if condition_on_gsd is not None else config.CONDITION_ON_GSD
    amp = config.AMP if amp is None else amp

    if model is None:
        model = SwinUNet(condition_on_gsd=condition_on_gsd)
    model.to(device)

    if train_ds is None or val_ds is None:
        train_ds = MultiSensorDataset(split="train")
        val_ds = MultiSensorDataset(split="val")
    train_batches = make_batches(train_ds, batch_size)
    val_batches = make_batches(val_ds, batch_size)

    criterion = TERRAIN_Loss()
    # fix 5: two param groups — low LR for the frozen-ish encoder, high LR for decoder/head.
    enc_params = list(model.encoder.parameters()) + list(model.patch_embed.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": config.ENCODER_LR},
            {"params": dec_params, "lr": lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_iou, history = -1.0, []
    for epoch in range(1, epochs + 1):
        for pg in optimizer.param_groups:
            pg["lr"] = _get_lr(epoch - 1, epochs)
        tr = train_epoch(model, train_ds, train_batches, criterion, optimizer, device,
                         condition_on_gsd, scaler=scaler, grad_accum=grad_accum,
                         grad_clip=grad_clip, log_components=True)
        val_iou, per_sensor = validate_epoch(model, val_ds, val_batches, device,
                                             condition_on_gsd, amp=amp)
        entry = {"epoch": epoch, "train_loss": tr["loss"], "val_iou": val_iou,
                 "per_sensor": per_sensor}
        entry.update({f"train_{k}": round(v, 5) for k, v in tr.items() if k != "loss"})
        history.append(entry)
        sensor_str = " ".join(f"{sid}:{v:.3f}" for sid, v in per_sensor.items())
        comp_str = " ".join(f"{k[6:]}:{v:.3f}" for k, v in entry.items() if k.startswith("train_"))
        print(f"[E{epoch:02d}] lr={optimizer.param_groups[0]['lr']:.2e} "
              f"train_loss={tr['loss']:.4f} ({comp_str}) | val_meanIoU={val_iou:.4f} "
              f"| per_sensor {{{sensor_str}}}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path)
            sidecar = os.path.splitext(save_path)[0] + "_best.json"
            with open(sidecar, "w") as f:
                json.dump({"epoch": epoch, "val_iou": best_iou, "per_sensor": per_sensor},
                          f, indent=2)
            print(f"          saved best -> {save_path} (meanIoU={best_iou:.4f})")
    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU overfit check")
    parser.add_argument("--save", default=config.MODEL_SAVE_PATH)
    args = parser.parse_args()

    if args.smoke:
        # minimal overfit test on a random subset
        train_ds = MultiSensorDataset(split="train")
        val_ds = MultiSensorDataset(split="val")
        rng = np.random.default_rng(0)
        t_inds = rng.choice(len(train_ds), size=min(160, len(train_ds)), replace=False)
        v_inds = rng.choice(len(val_ds), size=min(80, len(val_ds)), replace=False)
        bsz = args.batch or 4
        t_batches = [t_inds[i:i + bsz].tolist() for i in range(0, len(t_inds), bsz)]
        v_batches = [v_inds[i:i + bsz].tolist() for i in range(0, len(v_inds), bsz)]
        from segmentation.train import train_epoch, validate_epoch, SwinUNet, TERRAIN_Loss
        from segmentation.train import train_model
        # wrap: monkeypatch batches via partial by rebuilding small train/val objects
        model = SwinUNet()
        device = torch.device("cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = TERRAIN_Loss()
        for ep in range(1, 4):
            tr = train_epoch(model, train_ds, t_batches, criterion, optimizer, device)
            vi, ps = validate_epoch(model, val_ds, v_batches, device)
            print(f"[smoke E{ep}] train_loss={tr['loss']:.4f} val_meanIoU={vi:.4f}")
        print("SMOKE OK")
    else:
        train_model(epochs=args.epochs, batch_size=args.batch, save_path=args.save)
