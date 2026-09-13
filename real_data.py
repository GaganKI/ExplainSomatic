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
    """Returns a SET of (chrom, pos_0based) for every position in the truth
    VCF -- used ONLY to decide label (real variant or not). We deliberately
    do NOT trust the VCF's own AF/VAF annotation for the numeric VAF value
    anymore (real SEQC2 truth VCFs don't reliably carry it under a
    predictable field name). Instead, build_cache() below uses the actual
    OBSERVED alt-allele fraction computed directly from the real tumour
    reads at that position -- which is the literal definition of VAF, and
    doesn't depend on guessing a VCF schema at all."""
    truth = set()
    vcf = pysam.VariantFile(vcf_path)
    for rec in vcf.fetch():
        truth.add((rec.chrom, rec.pos - 1))  # VCF is 1-based; we key 0-based like pysam pileup
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
                 label_from="truth_vcf", chunk_size=256, flush_every=200):
    """label_from: 'truth_vcf' (recommended) uses the SEQC2 truth set for the
    binary label and its annotated VAF where available; falls back to the
    observed alt allele fraction at that site for the VAF value otherwise.

    RESUMABLE: if out_h5 already exists (e.g. a previous run got cut off by
    a Colab disconnect), this picks up from the last completed index instead
    of starting over. A 'done' boolean dataset tracks progress; the file is
    flushed to disk every `flush_every` examples so a disconnect loses at
    most that many examples of work, not the whole run.
    """
    import os as _os

    candidates = read_candidates(candidates_bed)
    n = len(candidates)
    truth = load_truth_set(truth_vcf_path) if truth_vcf_path else {}
    bam = pysam.AlignmentFile(bam_path, "rb", reference_filename=ref_path)
    ref = pysam.FastaFile(ref_path)

    resuming = _os.path.exists(out_h5)
    h5 = h5py.File(out_h5, "a")  # append mode: create if missing, reuse if present

    if resuming and "label" in h5:
        existing_n = h5["label"].shape[0]
        if existing_n != n:
            raise ValueError(
                f"Existing cache at {out_h5} has {existing_n} rows but the candidates "
                f"file now has {n} -- looks like a different candidates.bed than the one "
                f"this cache was started from. Delete the .h5 and restart, or point at the "
                f"matching candidates file."
            )
        d_pileup, d_ctx = h5["pileup"], h5["context"]
        d_label, d_vaf = h5["label"], h5["vaf"]
        d_chrom, d_pos = h5["chrom"], h5["pos"]
        d_done = h5["done"]
        start_i = int(d_done[:].sum())  # done[] is written contiguously, so sum == first unfinished index
        print(f"resuming {out_h5}: {start_i}/{n} already cached")
    else:
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
        d_done = h5.create_dataset("done", shape=(n,), dtype="bool")
        start_i = 0
        print(f"starting new cache: {out_h5}, {n} candidates")

    for i in range(start_i, n):
        chrom, pos, ref_base = candidates[i]
        pileup, ctx_tokens, observed_vaf = extract_pileup_and_context(bam, ref, chrom, pos, ref_base)
        key = (chrom, pos)
        label = 1.0 if key in truth else 0.0
        # VAF is always the REAL observed alt-allele fraction from the actual
        # tumour reads at this site -- not a value trusted from the VCF's
        # own annotation, which we no longer assume exists in any particular form.
        vaf = observed_vaf

        d_pileup[i] = pileup.astype("float16")
        d_ctx[i] = ctx_tokens.astype("int8")
        d_label[i] = label
        d_vaf[i] = vaf
        d_chrom[i] = chrom
        d_pos[i] = pos
        d_done[i] = True

        if (i + 1) % flush_every == 0:
            h5.flush()
            print(f"  cached {i+1}/{n} candidates... (flushed)")

    h5.flush()
    h5.close()
    bam.close()
    ref.close()
    print(f"done: {n} examples -> {out_h5}")


class CachedSomaticDataset(Dataset):
    """Two modes:

    in_memory=True (default): loads the entire cache into RAM as plain numpy
    arrays once, then indexes directly from memory. Strongly recommended
    whenever the cache fits comfortably in RAM (a few GB) -- gzip-compressed
    HDF5 chunks decompress the WHOLE chunk on any access, and with a shuffled
    per-example DataLoader, that means near-constant chunk re-decompression
    regardless of whether the file is local or on Drive. This is very likely
    why training was slow even after moving files to local disk.

    in_memory=False: old behaviour, lazy per-worker h5py file handle with
    random access straight from disk. Only use this for a cache too large
    to fit in RAM (multi-chromosome, whole-genome scale).
    """
    def __init__(self, h5_path, in_memory=True):
        self.h5_path = h5_path
        self.in_memory = in_memory
        self._h5 = None

        if in_memory:
            with h5py.File(h5_path, "r") as h5:
                self.pileup = h5["pileup"][:].astype(np.float32)
                self.context = h5["context"][:].astype(np.int64)
                self.label = h5["label"][:].astype(np.float32)
                self.vaf = h5["vaf"][:].astype(np.float32)
            self.length = len(self.label)
        else:
            with h5py.File(h5_path, "r") as h5:
                self.length = h5["label"].shape[0]

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.in_memory:
            return (torch.from_numpy(self.pileup[idx]),
                    torch.from_numpy(self.context[idx]),
                    torch.tensor(self.label[idx]),
                    torch.tensor(self.vaf[idx]))
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