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
def evaluate(model, dl, device):
    model.eval()
    all_logits, all_labels, all_vaf = [], [], []
    for pileup, ctx, label, vaf in dl:
        pileup, ctx = pileup.to(device), ctx.to(device)
        logit = model(pileup, ctx)
        all_logits.append(logit.cpu())
        all_labels.append(label)
        all_vaf.append(vaf)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    vaf = torch.cat(all_vaf)
    preds = (torch.sigmoid(logits) > 0.5).float()

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
            "n_low_vaf": int(low_vaf_mask.sum()), "n_high_vaf": int(high_vaf_mask.sum())}


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
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = CachedSomaticDataset(args.train_h5)
    val_ds = CachedSomaticDataset(args.val_h5)
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
        if val_metrics["low_vaf_recall"] == val_metrics["low_vaf_recall"] and \
           val_metrics["low_vaf_recall"] > best_low_vaf_recall:  # NaN-safe check
            best_low_vaf_recall = val_metrics["low_vaf_recall"]
            save_checkpoint(best_path, model, opt, sched, epoch, best_low_vaf_recall)
            print(f"  -> new best (low_vaf_recall={best_low_vaf_recall:.4f}), saved to {best_path}")

    print("training complete. best low-VAF recall:", best_low_vaf_recall)


if __name__ == "__main__":
    main()