"""
real_data.py
-------------
Two jobs, deliberately kept separate for production efficiency:

  1. build_cache(...)   -- SLOW, I/O-bound, run ONCE per chromosome/region.
     Reads candidate sites (from candidate_generation.py) + the BAM +
     reference FASTA + SEQC2 truth VCF, extracts a pileup tensor and a
     +-150bp context window for every candidate, labels it against the
     truth set, and writes everything to a chunked HDF5 file.

  2. CachedSomaticDataset -- FAST, used every epoch. Reads directly from
     the HDF5 file with random access, so training doesn't re-touch the
     BAM file at all after caching. This is the difference between a
     prototype (re-parse BAM every epoch, unusable at scale) and something
     you can actually iterate on with a real training budget.

Tensor schema is IDENTICAL to data_sim.py's synthetic version on purpose --
models.py, losses.py, gradcam.py all work unchanged on real data.
"""

import argparse
import csv

import h5py
import numpy as np
import pysam
import torch
from torch.utils.data import Dataset

BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
PILEUP_WIDTH = 21
NUM_READS = 64
CTX_LEN = 301
N_CHANNELS = 5


def read_candidates(bed_path):
    rows = []
    with open(bed_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append((r["chrom"], int(r["pos"]), r["ref_base"]))
    return rows


def load_truth_set(vcf_path):
    """Returns dict: (chrom, pos_0based) -> true_vaf (float).
    Falls back to VAF=0.5 if the VCF doesn't carry an explicit AF/VAF INFO
    field (e.g. a purely binary truth-call VCF) -- flag this to yourself
    if you hit it, since it means low-VAF weighting will be less precise
    for that subset until you pull real AFs from the SEQC2 supplementary
    tables instead of the VCF alone."""
    truth = {}
    vcf = pysam.VariantFile(vcf_path)
    for rec in vcf.fetch():
        vaf = None
        for key in ("VAF", "AF"):
            if key in rec.info:
                val = rec.info[key]
                vaf = float(val[0] if isinstance(val, (tuple, list)) else val)
                break
        if vaf is None:
            vaf = 0.5
        truth[(rec.chrom, rec.pos - 1)] = vaf  # VCF is 1-based; we key 0-based like pysam pileup
    vcf.close()
    return truth


def extract_pileup_and_context(bam, ref, chrom, pos, ref_base,
                                width=PILEUP_WIDTH, num_reads=NUM_READS, ctx_len=CTX_LEN):
    """One real candidate site -> (pileup tensor, context tokens, observed_alt_frac)."""
    half_w = width // 2
    pileup = np.full((N_CHANNELS, num_reads, width), -1.0, dtype=np.float32)

    # gather reads overlapping the center position
    reads_here = []
    for pileup_col in bam.pileup(chrom, pos, pos + 1, min_base_quality=10,
                                  ignore_overlaps=True, truncate=True, max_depth=num_reads * 4):
        if pileup_col.reference_pos != pos:
            continue
        for pr in pileup_col.pileups:
            if pr.is_del or pr.is_refskip or pr.query_position is None:
                continue
            aln = pr.alignment
            if aln.is_duplicate or aln.is_secondary or aln.is_qcfail or aln.is_unmapped:
                continue
            reads_here.append((aln, pr.query_position))

    reads_here = reads_here[:num_reads]
    n_alt = 0
    for r_idx, (aln, qpos) in enumerate(reads_here):
        seq = aln.query_sequence
        quals = aln.query_qualities
        # walk the window of genome offsets around the candidate for this read
        for w in range(width):
            genome_pos = pos - half_w + w
            read_offset = qpos + (genome_pos - pos)
            if read_offset < 0 or read_offset >= len(seq):
                continue  # read doesn't cover this column
            base = seq[read_offset].upper()
            if base not in "ACGT":
                continue
            local_ref = ref.fetch(chrom, genome_pos, genome_pos + 1).upper()
            match = 1.0 if base == local_ref else 0.0
            if w == half_w and match == 0.0:
                n_alt += 1
            pileup[0, r_idx, w] = match
            pileup[1, r_idx, w] = (quals[read_offset] / 40.0) if quals else 0.7
            pileup[2, r_idx, w] = 1.0 if aln.is_reverse else 0.0
            pileup[3, r_idx, w] = min(aln.mapping_quality / 60.0, 1.0)
            pileup[4, r_idx, w] = 1.0 if w == half_w else 0.0

    depth_at_center = len(reads_here)
    observed_alt_frac = (n_alt / depth_at_center) if depth_at_center > 0 else 0.0

    # sequence context for Stream B: reference sequence, not per-read
    ctx_half = ctx_len // 2
    ctx_seq = ref.fetch(chrom, pos - ctx_half, pos + ctx_half + 1).upper()
    ctx_tokens = np.array([BASE2IDX.get(b, 0) for b in ctx_seq], dtype=np.int64)
    if len(ctx_tokens) < ctx_len:
        ctx_tokens = np.pad(ctx_tokens, (0, ctx_len - len(ctx_tokens)))

    return pileup, ctx_tokens, observed_alt_frac


def build_cache(candidates_bed, bam_path, ref_path, truth_vcf_path, out_h5,
                 label_from="truth_vcf", chunk_size=256):
    """label_from: 'truth_vcf' (recommended) uses the SEQC2 truth set for the
    binary label and its annotated VAF where available; falls back to the
    observed alt allele fraction at that site for the VAF value otherwise."""
    candidates = read_candidates(candidates_bed)
    truth = load_truth_set(truth_vcf_path) if truth_vcf_path else {}
    bam = pysam.AlignmentFile(bam_path, "rb", reference_filename=ref_path)
    ref = pysam.FastaFile(ref_path)

    n = len(candidates)
    with h5py.File(out_h5, "w") as h5:
        d_pileup = h5.create_dataset("pileup", shape=(n, N_CHANNELS, NUM_READS, PILEUP_WIDTH),
                                      dtype="float16", chunks=(min(chunk_size, n), N_CHANNELS, NUM_READS, PILEUP_WIDTH),
                                      compression="gzip", compression_opts=4)
        d_ctx = h5.create_dataset("context", shape=(n, CTX_LEN), dtype="int8",
                                   chunks=(min(chunk_size, n), CTX_LEN), compression="gzip")
        d_label = h5.create_dataset("label", shape=(n,), dtype="float32")
        d_vaf = h5.create_dataset("vaf", shape=(n,), dtype="float32")
        chrom_dt = h5py.string_dtype(encoding="utf-8")
        d_chrom = h5.create_dataset("chrom", shape=(n,), dtype=chrom_dt)
        d_pos = h5.create_dataset("pos", shape=(n,), dtype="int64")

        for i, (chrom, pos, ref_base) in enumerate(candidates):
            pileup, ctx_tokens, observed_vaf = extract_pileup_and_context(bam, ref, chrom, pos, ref_base)
            key = (chrom, pos)
            if key in truth:
                label = 1.0
                vaf = truth[key]
            else:
                label = 0.0
                vaf = 0.0 if label_from == "truth_vcf" else observed_vaf

            d_pileup[i] = pileup.astype("float16")
            d_ctx[i] = ctx_tokens.astype("int8")
            d_label[i] = label
            d_vaf[i] = vaf
            d_chrom[i] = chrom
            d_pos[i] = pos

            if (i + 1) % 500 == 0:
                print(f"  cached {i+1}/{n} candidates...")

    bam.close()
    ref.close()
    print(f"done: {n} examples -> {out_h5}")


class CachedSomaticDataset(Dataset):
    """Fast, random-access dataset over a pre-built HDF5 cache. Opens the
    file lazily per-worker (required for multi-worker DataLoader + h5py)."""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._h5 = None
        with h5py.File(h5_path, "r") as h5:
            self.length = h5["label"].shape[0]

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()
        pileup = torch.from_numpy(self._h5["pileup"][idx].astype(np.float32))
        ctx = torch.from_numpy(self._h5["context"][idx].astype(np.int64))
        label = torch.tensor(self._h5["label"][idx], dtype=torch.float32)
        vaf = torch.tensor(self._h5["vaf"][idx], dtype=torch.float32)
        return pileup, ctx, label, vaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="output of candidate_generation.py")
    ap.add_argument("--bam", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--truth_vcf", required=True, help="SEQC2 high-confidence truth VCF")
    ap.add_argument("--out", required=True, help="output .h5 cache path")
    args = ap.parse_args()
    build_cache(args.candidates, args.bam, args.ref, args.truth_vcf, args.out)


if __name__ == "__main__":
    main()