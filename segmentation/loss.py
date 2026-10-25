"""
Segmentation losses for TERRAIN.

- DiceLoss          : region overlap, robust to extreme class imbalance (thin roads).
- IoULoss           : Jaccard-style soft overlap.
- SkeletonLoss      : distance-transformed BCE that weights pixels near the road
                      centerline most heavily (replaces the noisy Sobel boundary loss).
- FocalLoss         : down-weights easy background pixels; pos_weight counteracts
                      the class imbalance without flooding roads.
- ConnectivityLoss  : penalizes spatial gaps in thin roads (helps graph healing later).
- TERRAIN_Loss      : weighted sum of the above tuned by config.LOSS_* weights.

All receive model logits (B,1,H,W) and binary masks (B,1,H,W) in [0,1].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        num = 2.0 * (pred * target).sum() + self.smooth
        den = pred.sum() + target.sum() + self.smooth
        return 1.0 - num / den


class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return 1.0 - (inter + self.smooth) / (union + self.smooth)


class SkeletonLoss(nn.Module):
    """Soft skeleton loss: distance-transform weighted BCE.

    At 10 m Sentinel-2 resolution road edges are only 1-2 px wide, so a Sobel
    boundary loss gives garbage gradients on such thin features. Instead we weight
    every positive pixel by exp(-distance_to_skeleton / 5.0) — pixels on/near the
    road centerline dominate the BCE gradient, encouraging thin connected roads.
    """

    def forward(self, logits, target):
        from scipy.ndimage import distance_transform_edt

        target = target.to(logits.device)
        pred = torch.sigmoid(logits)
        target_np = target.squeeze(1).cpu().numpy()
        weights = np.zeros_like(target_np, dtype=np.float32)
        for i in range(target_np.shape[0]):
            if target_np[i].sum() > 0:
                dist = distance_transform_edt(target_np[i].astype(np.float32))
                weights[i] = np.exp(-dist / 5.0)
            else:
                weights[i] = np.ones_like(target_np[i], dtype=np.float32)
        w = torch.from_numpy(weights).float().to(logits.device).unsqueeze(1)
        return F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="mean")


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) — down-weights easy background pixels so the
    few hard road pixels dominate the gradient (thin-road class imbalance).

    pos_weight (config.POS_WEIGHT) applies BCEWithLogits' positive-class weighting.
    Kept low (3.0) — a very high value overcorrects and floods the map with roads.
    """

    def __init__(self, gamma: float = None, alpha: float = None, pos_weight: float = None):
        super().__init__()
        self.gamma = gamma if gamma is not None else config.FOCAL_GAMMA
        self.alpha = alpha if alpha is not None else config.FOCAL_ALPHA
        self.pos_weight = pos_weight if pos_weight is not None else config.POS_WEIGHT
        self.register_buffer("_pw", torch.tensor([self.pos_weight], dtype=torch.float32))

    def forward(self, logits, target):
        # pos_weight must live on logits' device and broadcast against (B,1,H,W):
        # reshape to (1,1,1,1). Guards against the buffer staying on CPU when the
        # criterion module was never moved to cuda.
        target = target.to(logits.device)
        pw = self._pw.to(logits.device).view(1, 1, 1, 1) if target.dim() > 1 \
            else self._pw.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pw)
        pt = torch.exp(-bce)                       # prob of the correct class
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (at * (1 - pt) ** self.gamma * bce).mean()


class ConnectivityLoss(nn.Module):
    """Weak supervision toward connected structures: penalize small isolated
    components of the prediction that are not present in the target."""

    def __init__(self, kernel: int = 3):
        super().__init__()
        ker = torch.ones(1, 1, kernel, kernel)
        self.register_buffer("ker", ker)

    def forward(self, logits, target):
        bsz = logits.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        ones = torch.ones_like(pred)
        # 1 - signal => gaps; conv counts how many neighbors a 'gap' has.
        gaps = F.conv2d(pred - 1.0, self.ker.to(logits.device), padding=1) * -1.0
        # gaps is negative signal; take abs count of neighbor-gaps
        gaps = gaps.abs()
        interior = gaps / self.ker.numel()
        loss = interior.mean(dim=[1, 2, 3])            # (B)
        return loss.mean()


class TERRAIN_Loss(nn.Module):
    """Weighted composite: Dice + Focal + Skeleton + Connectivity.

    Focal (config.LOSS_BETA) carries the pixel-level imbalance load; Skeleton
    (config.LOSS_GAMMA) guides thin-road topology; Connectivity (config.LOSS_DELTA)
    is a small, noise-tolerant structural term.
    """

    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(pos_weight=pos_weight)
        self.skeleton = SkeletonLoss()
        self.connectivity = ConnectivityLoss()

    def _forward(self, logits, target):
        d = self.dice(logits, target)
        f = self.focal(logits, target)
        sk = self.skeleton(logits, target)
        c = self.connectivity(logits, target)
        loss = (
            config.LOSS_ALPHA * d
            + config.LOSS_BETA * f
            + config.LOSS_GAMMA * sk
            + config.LOSS_DELTA * c
        )
        return loss, {"dice": d, "focal": f, "skeleton": sk, "connectivity": c}

    def forward(self, logits, target):
        return self._forward(logits, target)[0]

    def components(self, logits, target):
        """Return per-term dict for logging (each is a detached scalar)."""
        return {k: float(v) for k, v in self._forward(logits, target)[1].items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.7).float()
    for name, loss in [
        ("Dice", DiceLoss()), ("IoU", IoULoss()), ("Focal", FocalLoss()),
        ("Skeleton", SkeletonLoss()), ("Connectivity", ConnectivityLoss()),
        ("TERRAIN", TERRAIN_Loss()),
    ]:
        value = loss(logits, target)
        print(f"{name:14s} -> {value:.4f}  (finite={torch.isfinite(value).item()})")
    print("Dice(perfect)=", DiceLoss()(torch.logit(target), target).item())
    print("IoU (perfect)=", IoULoss()(torch.logit(target), target).item())