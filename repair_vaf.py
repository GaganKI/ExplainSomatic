"""
repair_vaf.py
--------------
One-time fix for caches built before the VAF-source bug was fixed in
real_data.py. Recomputes the real observed VAF for every cached example
directly from its pileup tensor's center column (channel 0 = match/mismatch,
already sitting in the .h5 file) -- no BAM access needed, so this is fast.

Usage:
    python3 repair_vaf.py --h5 train.h5
    python3 repair_vaf.py --h5 val.h5
    python3 repair_vaf.py --h5 test.h5
"""

import argparse
import h5py
import numpy as np

CENTER_COL = 21 // 2  # PILEUP_WIDTH // 2, matches real_data.py


def repair(h5_path, batch=2000):
    with h5py.File(h5_path, "r+") as h5:
        n = h5["label"].shape[0]
        old_vaf = h5["vaf"][:]
        new_vaf = np.zeros(n, dtype=np.float32)

        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = h5["pileup"][start:end, 0, :, CENTER_COL]  # (batch, 64) match/mismatch/-1
            covered = chunk >= 0
            mismatches = (chunk == 0) & covered
            depth = covered.sum(axis=1)
            alt = mismatches.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                vaf = np.where(depth > 0, alt / np.maximum(depth, 1), 0.0)
            new_vaf[start:end] = vaf

        h5["vaf"][:] = new_vaf

        changed = int((np.abs(new_vaf - old_vaf) > 1e-6).sum())
        labels = h5["label"][:]
        pos_mask = labels == 1
        low_vaf_n = int(((new_vaf < 0.05) & pos_mask).sum())
        high_vaf_n = int(((new_vaf >= 0.05) & pos_mask).sum())

    print(f"{h5_path}: repaired {changed}/{n} vaf values")
    print(f"  positives now split: low-VAF(<5%)={low_vaf_n}  high-VAF(>=5%)={high_vaf_n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    args = ap.parse_args()
    repair(args.h5)