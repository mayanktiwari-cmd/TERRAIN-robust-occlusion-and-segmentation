"""
Evaluation metrics for the multi-sensor road segmentation model.

A deterministic, augmentation-free pass over a MultiSensorDataset. Reports
per-sensor IoU / Dice / precision / recall, an optimal-threshold sweep (IoU is
NOT fixed at 0.5), and a sparse-vs-dense (road-fraction) robustness breakdown so
the "crash on sparse-road tiles" failure mode is a tracked number.

Note: `dataset` here is a MultiSensorDataset (built with augment=False) — it uses
__getitem__/__len__ + .sensor_ids + .items. The old DataLoader-based stubs were
never implemented; this is the replacement API.
"""

import os

import numpy as np
import torch

import config
from segmentation.dataset import MultiSensorDataset  # noqa: F401  (documents the expected type)


def _load_road_fracs():
    try:
        with open(config.ROAD_FRACS_PATH, "r") as f:
            import json
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _confusion_sweep(model, dataset, device, thresholds):
    """Run the model once; accumulate per-sensor TP/FP/FN at every threshold."""
    model.eval()
    ths = np.asarray(thresholds)
    acc = {sid: np.zeros((len(ths), 3)) for sid in dataset.sensor_ids.values()}
    with torch.no_grad():
        for i in range(len(dataset)):
            img, mask, meta = dataset[i]               # augment=False => deterministic
            sid = meta[0]
            logits = model(img.unsqueeze(0).to(device))[0, 0]
            prob = torch.sigmoid(logits)
            t = mask[0].to(device)
            for j, th in enumerate(ths):
                p = (prob > th).float()
                acc[sid][j, 0] += (p * t).sum().item()        # tp
                acc[sid][j, 1] += (p * (1 - t)).sum().item()  # fp
                acc[sid][j, 2] += ((1 - p) * t).sum().item()  # fn
    return acc


def _metrics(tp, fp, fn):
    tp, fp, fn = max(tp, 0.0), max(fp, 0.0), max(fn, 0.0)
    den = tp + fp + fn + 1e-6
    return {
        "iou": tp / den,
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-6),
        "precision": tp / (tp + fp + 1e-6),
        "recall": tp / (tp + fn + 1e-6),
    }


def compute_ious(model, dataset, device,
                 thresholds=None, name="val"):
    """Per-sensor + pooled metrics across a threshold sweep.

    Returns dict with:
      thresholds, best_threshold (argmax of mean per-sensor IoU),
      per_sensor: {sensor_id: {threshold: metrics}},
      overall:    {threshold: pooled metrics across all sensors},
      best:       metrics at best_threshold (pooled).
    """
    thresholds = list(thresholds) if thresholds is not None \
        else [round(t, 2) for t in np.arange(0.30, 0.80, 0.05)]
    acc = _confusion_sweep(model, dataset, device, thresholds)

    per_sensor = {}
    for sid, counts in acc.items():
        per_sensor[sid] = {}
        for j, th in enumerate(thresholds):
            per_sensor[sid][th] = _metrics(*counts[j])
    # pooled overall at each threshold
    overall = {}
    for j, th in enumerate(thresholds):
        tot = np.sum([acc[sid][j] for sid in acc], axis=0)
        overall[th] = _metrics(*tot)

    means = {th: float(np.mean([per_sensor[sid][th]["iou"] for sid in per_sensor]))
             for th in thresholds}
    best_t = max(means, key=means.get)
    print(f"[eval {name}] pooled IoU by threshold: " +
          " ".join(f"{th}:{overall[th]['iou']:.3f}" for th in thresholds) +
          f" | best t={best_t} (mean per-sensor IoU {means[best_t]:.3f}, "
          f"pooled IoU {overall[best_t]['iou']:.3f})")
    for sid in per_sensor:
        m = per_sensor[sid][best_t]
        print(f"  sensor {sid} @ {best_t}: IoU={m['iou']:.3f} Dice={m['dice']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f}")
    return {
        "name": name,
        "thresholds": thresholds,
        "best_threshold": best_t,
        "per_sensor": per_sensor,
        "overall": overall,
        "best": overall[best_t],
    }


def sparse_vs_dense_ious(model, dataset, device, threshold=None,
                         split_at=0.05, road_fracs=None):
    """IoU on low-road (<split_at) vs normal (>=split_at) tiles.

    A big sparse<->dense gap is the signature of overfitting to road-dense tiles.
    Returns dict {'sparse': metrics, 'dense': metrics} or None if no fracs map.
    """
    fracs = road_fracs or _load_road_fracs()
    if not fracs:
        return None
    threshold = config.INFER_THRESHOLD if threshold is None else threshold
    groups = {"sparse": [0.0, 0.0, 0.0], "dense": [0.0, 0.0, 0.0]}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            img, mask, meta = dataset[i]
            name = f"{dataset.items[i][0]}/{os.path.basename(dataset.items[i][2])}"
            frac = fracs.get(name)
            if frac is None:
                continue
            key = "sparse" if frac < split_at else "dense"
            prob = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0]
            p = (prob > threshold).float()
            t = mask[0].to(device)
            groups[key][0] += (p * t).sum().item()
            groups[key][1] += (p * (1 - t)).sum().item()
            groups[key][2] += ((1 - p) * t).sum().item()
    out = {k: _metrics(*v) for k, v in groups.items()}
    print(f"[eval] sparse(<{split_at:.2f} road) IoU={out['sparse']['iou']:.3f} | "
          f"dense(>=) IoU={out['dense']['iou']:.3f}")
    return out


def evaluate_model(model, dataset, device, name="val", do_sparse=True):
    """Full evaluation: threshold sweep + per-sensor + sparse/dense."""
    res = compute_ious(model, dataset, device, name=name)
    res["sparse_vs_dense"] = sparse_vs_dense_ious(model, dataset, device,
                                                  road_fracs=_load_road_fracs())
    return res


def occlusion_recall(model, loader, device, occlude_fraction=0.2):
    raise NotImplementedError("occlusion robustness eval is not part of the current scope")
