"""
train_production.py
---------------------
The real training loop, meant to run on Colab GPU against the HDF5 caches
built by real_data.py. Differences from a prototype loop that matter at
production scale:

  - Resumable: saves full state (model, optimizer, scheduler, epoch, best
    metric) every epoch, so a disconnected Colab session doesn't cost you
    compute units to redo.
  - Mixed precision (torch.cuda.amp) for real GPU throughput.
  - Gradient clipping (transformers are prone to occasional loss spikes).
  - Model selection by LOW-VAF RECALL, not overall accuracy/F1 -- picking
    the checkpoint with the best overall F1 would happily pick a model
    that's great on easy high-VAF calls and bad on the exact regime
    (Objective 2) this whole project is about.
  - Separate train/val/test HDF5 caches expected as input (build these with
    real_data.py once per split, e.g. by chromosome: train=chr1-18,
    val=chr19-20, test=chr21-22, so there's no position overlap.)

Usage (Colab, GPU runtime):
    python3 train_production.py \
        --train_h5 cache/train.h5 --val_h5 cache/val.h5 \
        --arch fusion --loss vaf_aware --epochs 30 \
        --ckpt_dir /content/drive/MyDrive/explainsomatic_ckpts/run1
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from models import ExplainSomaticModel, CNNOnlyModel, TransformerOnlyModel
from losses import VAFAwareLoss, PlainBCELoss
from real_data import CachedSomaticDataset


def build_model(arch, transformer_layers=6):
    if arch == "fusion":
        return ExplainSomaticModel(transformer_layers=transformer_layers)
    elif arch == "cnn_only":
        return CNNOnlyModel()
    elif arch == "transformer_only":
        return TransformerOnlyModel(transformer_layers=transformer_layers)
    raise ValueError(arch)


@torch.no_grad()
def collect_predictions(model, dl, device):
    """Runs the model once and returns raw logits/labels/vaf -- separated
    out so we can both (a) evaluate at a fixed threshold during training,
    and (b) sweep thresholds for calibration afterward, without a second
    forward pass over the data."""
    model.eval()
    all_logits, all_labels, all_vaf = [], [], []
    for pileup, ctx, label, vaf in dl:
        pileup, ctx = pileup.to(device), ctx.to(device)
        logit = model(pileup, ctx)
        all_logits.append(logit.cpu())
        all_labels.append(label)
        all_vaf.append(vaf)
    return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_vaf)


def metrics_at_threshold(logits, labels, vaf, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()

    def prf(mask):
        p, l = preds[mask], labels[mask]
        tp = ((p == 1) & (l == 1)).sum().item()
        fp = ((p == 1) & (l == 0)).sum().item()
        fn = ((p == 0) & (l == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
        return {"precision": precision, "recall": recall, "f1": f1, "n": int(mask.sum())}

    overall = prf(torch.ones_like(labels, dtype=torch.bool))
    low_vaf_mask = (labels == 1) & (vaf < 0.05)
    high_vaf_mask = (labels == 1) & (vaf >= 0.05)
    low_vaf_recall = preds[low_vaf_mask].mean().item() if low_vaf_mask.sum() > 0 else float("nan")
    high_vaf_recall = preds[high_vaf_mask].mean().item() if high_vaf_mask.sum() > 0 else float("nan")

    return {"overall": overall, "low_vaf_recall": low_vaf_recall, "high_vaf_recall": high_vaf_recall,
            "n_low_vaf": int(low_vaf_mask.sum()), "n_high_vaf": int(high_vaf_mask.sum()),
            "threshold": threshold}


def find_best_threshold(logits, labels, vaf, candidates=None):
    """Sweeps classification thresholds and returns the one maximizing
    overall F1 on the given (validation) set. The model was trained on
    artificially balanced batches (via --balanced_sampling) but is scored
    against the TRUE class distribution -- a fixed 0.5 cutoff is essentially
    arbitrary here and, empirically, made the model trigger-happy (high
    recall, very low precision). This replaces "assume 0.5" with "measure
    what actually works on held-out data"."""
    if candidates is None:
        probs = torch.sigmoid(logits)
        candidates = sorted(set(probs.tolist())) or [0.5]
        # thin out if there are a huge number of unique probabilities
        if len(candidates) > 500:
            step = len(candidates) // 500
            candidates = candidates[::step]

    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        m = metrics_at_threshold(logits, labels, vaf, threshold=t)
        f1 = m["overall"]["f1"]
        if f1 == f1 and f1 > best_f1:  # NaN-safe
            best_f1, best_t = f1, t
    return best_t, best_f1


@torch.no_grad()
def evaluate(model, dl, device, threshold=0.5):
    logits, labels, vaf = collect_predictions(model, dl, device)
    return metrics_at_threshold(logits, labels, vaf, threshold=threshold)


def save_checkpoint(path, model, opt, sched, epoch, best_metric):
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": sched.state_dict() if sched else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5", required=True)
    ap.add_argument("--val_h5", required=True)
    ap.add_argument("--arch", choices=["fusion", "cnn_only", "transformer_only"], default="fusion")
    ap.add_argument("--loss", choices=["vaf_aware", "plain_bce"], default="vaf_aware")
    ap.add_argument("--transformer_layers", type=int, default=6, help="full spec is 6; only lower this for a quick sanity run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--balanced_sampling", action="store_true", default=True,
                     help="Oversample positives during training so they actually appear in "
                          "batches (with <0.3%% positive rate, ~86%% of batches would otherwise "
                          "contain zero positives). Validation/test are never resampled -- they "
                          "stay at the true class distribution so metrics remain meaningful.")
    ap.add_argument("--no_balanced_sampling", dest="balanced_sampling", action="store_false")
    ap.add_argument("--target_positive_fraction", type=float, default=0.15,
                     help="What fraction of each training batch should be positives, on average, "
                          "under balanced sampling. 0.5 (the old default) made the model see "
                          "positives and negatives equally often, which is far from the true "
                          "0.23%% rate and made it trigger-happy (high recall, terrible precision). "
                          "0.15 is a calmer middle ground -- enough positive signal to learn from "
                          "without training on a wildly unrealistic distribution.")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = CachedSomaticDataset(args.train_h5)
    val_ds = CachedSomaticDataset(args.val_h5)

    if args.balanced_sampling:
        if hasattr(train_ds, "label"):
            labels = train_ds.label
        else:
            with __import__("h5py").File(args.train_h5, "r") as _h5:
                labels = _h5["label"][:]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        print(f"train set: {n_pos} positives, {n_neg} negatives "
              f"({100*n_pos/(n_pos+n_neg):.3f}% positive rate)")
        # weight each example inversely to its class frequency, so a
        # weighted random draw sees positives and negatives roughly equally
        # often instead of positives showing up in ~14% of batches by chance
        weight_pos = args.target_positive_fraction / max(n_pos, 1)
        weight_neg = (1.0 - args.target_positive_fraction) / max(n_neg, 1)
        sample_weights = torch.where(torch.from_numpy(labels) == 1,
                                      torch.full_like(torch.from_numpy(labels), weight_pos),
                                      torch.full_like(torch.from_numpy(labels), weight_neg))
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args.arch, args.transformer_layers).to(device)
    loss_fn = VAFAwareLoss() if args.loss == "vaf_aware" else PlainBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_low_vaf_recall = -1.0
    ckpt_path = os.path.join(args.ckpt_dir, "last.pt")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    log_path = os.path.join(args.ckpt_dir, "log.jsonl")

    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["opt_state"])
        if state["sched_state"]:
            sched.load_state_dict(state["sched_state"])
        start_epoch = state["epoch"] + 1
        best_low_vaf_recall = state["best_metric"]
        print(f"resumed from epoch {start_epoch}, best_low_vaf_recall so far = {best_low_vaf_recall:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for pileup, ctx, label, vaf in train_dl:
            pileup, ctx, label, vaf = [x.to(device) for x in (pileup, ctx, label, vaf)]
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logit = model(pileup, ctx)
                loss = loss_fn(logit, label, vaf)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1
        sched.step()

        val_metrics = evaluate(model, val_dl, device)
        epoch_time = time.time() - t0
        avg_loss = running_loss / max(n_batches, 1)

        record = {"epoch": epoch, "train_loss": avg_loss, "epoch_time_sec": epoch_time,
                   "lr": sched.get_last_lr()[0], "val": val_metrics}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
              f"val_f1={val_metrics['overall']['f1']:.3f}  "
              f"low_vaf_recall={val_metrics['low_vaf_recall']:.3f} (n={val_metrics['n_low_vaf']})  "
              f"high_vaf_recall={val_metrics['high_vaf_recall']:.3f} (n={val_metrics['n_high_vaf']})  "
              f"[{epoch_time:.1f}s]")

        save_checkpoint(ckpt_path, model, opt, sched, epoch, best_low_vaf_recall)
        # Prefer low-VAF recall as the selection metric whenever it's actually
        # defined (n_low_vaf > 0) -- that's the real Objective 2 metric. But
        # right now, on chr21, val has ZERO low-VAF positives, so that metric
        # is permanently NaN here; falling back to overall F1 in that case is
        # what makes save-best actually produce a checkpoint at all instead of
        # silently never saving one. Once a chromosome with low-VAF val
        # examples is added, this automatically switches back to the real metric.
        if val_metrics["n_low_vaf"] > 0:
            current_score = val_metrics["low_vaf_recall"]
        else:
            current_score = val_metrics["overall"]["f1"]
        if current_score == current_score and current_score > best_low_vaf_recall:  # NaN-safe
            best_low_vaf_recall = current_score
            save_checkpoint(best_path, model, opt, sched, epoch, best_low_vaf_recall)
            fallback_note = "" if val_metrics["n_low_vaf"] > 0 else " (fallback: no low-VAF val examples yet, selected by overall F1)"
            print(f"  -> new best (score={best_low_vaf_recall:.4f}){fallback_note}, saved to {best_path}")

    print("training complete. best selection score:", best_low_vaf_recall)

    # ---- Threshold calibration on the best checkpoint ----
    # The model trains on artificially balanced batches (--balanced_sampling)
    # but is scored against the true class distribution -- a fixed 0.5 cutoff
    # is close to arbitrary here. Find the threshold that actually maximizes
    # F1 on validation, using the BEST checkpoint, and report calibrated
    # metrics so the number you show a panel reflects a real decision
    # boundary, not a default that happened to ship with BCEWithLogitsLoss.
    if os.path.exists(best_path):
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model_state"])
        logits, labels, vaf = collect_predictions(model, val_dl, device)
        best_threshold, best_f1 = find_best_threshold(logits, labels, vaf)
        calibrated_metrics = metrics_at_threshold(logits, labels, vaf, threshold=best_threshold)

        print(f"\ncalibration: best threshold = {best_threshold:.4f} (vs. default 0.5)")
        print(f"  at default 0.5   -> {metrics_at_threshold(logits, labels, vaf, 0.5)['overall']}")
        print(f"  at calibrated {best_threshold:.3f} -> {calibrated_metrics['overall']}")

        best_state["calibrated_threshold"] = best_threshold
        best_state["calibrated_val_metrics"] = calibrated_metrics
        torch.save(best_state, best_path)
        with open(os.path.join(args.ckpt_dir, "calibration.json"), "w") as f:
            json.dump({"threshold": best_threshold, "val_metrics": calibrated_metrics}, f, indent=2)
        print(f"saved calibrated threshold + metrics to {args.ckpt_dir}/calibration.json")
    else:
        print("no best.pt was ever saved -- nothing to calibrate. "
              "(shouldn't happen after the fallback fix above, but flagging just in case.)")


if __name__ == "__main__":
    main()