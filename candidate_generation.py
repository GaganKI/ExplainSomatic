"""
candidate_generation.py
------------------------
PRODUCTION NECESSITY, not an optional step: a genome has ~3.2 billion
positions. You cannot run a CNN+Transformer forward pass on every single one
-- that's exactly why DeepVariant, DeepSomatic, etc. all run a cheap
candidate-generation prefilter first, and only send the small surviving
subset (typically <1% of positions) to the expensive deep model.

This module scans a BAM/CRAM with pysam and flags a position as a candidate
if it has at least `min_alt_reads` reads disagreeing with the reference
allele, above `min_alt_frac` fraction of local depth. That's it -- deliberately
cheap and high-recall (it's fine to pass through some noise; the deep model's
job is precision). Everything downstream (real_data.py) only ever looks at
these candidate positions, not the whole genome.

Usage:
    python3 candidate_generation.py \
        --bam tumor_only.bam --ref GRCh38.fa --region chr21 \
        --out candidates_chr21.bed --min_alt_reads 2 --min_alt_frac 0.01
"""

import argparse
import bisect
from collections import defaultdict

import pysam


def load_regions_bed(bed_path):
    """Loads a BED file into {chrom: sorted list of (start, end)} for fast
    containment checks. Used to restrict candidates to the SEQC2
    High-Confidence_Regions BED -- outside these regions the truth VCF
    isn't reliable, so any 'negative' label there is meaningless noise,
    not a real hard negative."""
    regions = defaultdict(list)
    with open(bed_path) as f:
        for line in f:
            if line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            regions[chrom].append((start, end))
    for chrom in regions:
        regions[chrom].sort()
    return regions


def in_regions(chrom, pos, regions_by_chrom, starts_cache):
    """pos is 0-based. Returns True if pos falls inside any interval for
    this chromosome. starts_cache holds precomputed start-lists per
    chromosome so we don't rebuild them on every call."""
    intervals = regions_by_chrom.get(chrom)
    if not intervals:
        return False
    starts = starts_cache.get(chrom)
    if starts is None:
        starts = [s for s, e in intervals]
        starts_cache[chrom] = starts
    i = bisect.bisect_right(starts, pos) - 1
    if i < 0:
        return False
    start, end = intervals[i]
    return start <= pos < end


def generate_candidates(bam_path, ref_path, region, min_alt_reads=2, min_alt_frac=0.01,
                         min_depth=8, max_depth=2000, regions_bed=None):
    bam = pysam.AlignmentFile(bam_path, "rb", reference_filename=ref_path)
    ref = pysam.FastaFile(ref_path)

    regions_by_chrom = load_regions_bed(regions_bed) if regions_bed else None
    starts_cache = {}
    n_dropped_outside_regions = 0

    candidates = []
    for pileup_col in bam.pileup(region=region, min_base_quality=10, stepper="samtools",
                                  ignore_overlaps=True, max_depth=max_depth):
        chrom = pileup_col.reference_name
        pos = pileup_col.reference_pos  # 0-based
        ref_base = ref.fetch(chrom, pos, pos + 1).upper()
        if ref_base not in "ACGT":
            continue

        depth = 0
        alt_count = 0
        for pileup_read in pileup_col.pileups:
            if pileup_read.is_del or pileup_read.is_refskip:
                continue
            aln = pileup_read.alignment
            if aln.is_duplicate or aln.is_secondary or aln.is_qcfail or aln.is_unmapped:
                continue
            qpos = pileup_read.query_position
            if qpos is None:
                continue
            depth += 1
            read_base = aln.query_sequence[qpos].upper()
            if read_base != ref_base and read_base in "ACGT":
                alt_count += 1

        if depth < min_depth:
            continue
        alt_frac = alt_count / depth if depth else 0.0
        if alt_count >= min_alt_reads and alt_frac >= min_alt_frac:
            if regions_by_chrom is not None and not in_regions(chrom, pos, regions_by_chrom, starts_cache):
                n_dropped_outside_regions += 1
                continue
            candidates.append((chrom, pos, ref_base, depth, alt_count, round(alt_frac, 4)))

    bam.close()
    ref.close()
    if regions_bed:
        print(f"  ({n_dropped_outside_regions} candidates dropped for falling outside "
              f"the high-confidence regions BED)")
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", required=True, help="Tumour-only BAM/CRAM, indexed")
    ap.add_argument("--ref", required=True, help="Reference FASTA, indexed (.fai)")
    ap.add_argument("--region", required=True, help="e.g. chr21, or chr21:1-1000000 for a sub-range")
    ap.add_argument("--out", required=True, help="Output BED-like file of candidate positions")
    ap.add_argument("--min_alt_reads", type=int, default=2)
    ap.add_argument("--min_alt_frac", type=float, default=0.01,
                     help="Set this near/below your target low-VAF floor (e.g. 0.01 for 1%%), "
                          "or real low-VAF variants get filtered out before the model ever sees them.")
    ap.add_argument("--min_depth", type=int, default=8)
    ap.add_argument("--regions_bed", default=None,
                     help="e.g. High-Confidence_Regions_v1.2.bed from the SEQC2 release. "
                          "Strongly recommended for training data -- restricts candidates to "
                          "positions where the truth VCF is actually reliable, so a 'negative' "
                          "label means something rather than 'we have no idea, wasn't PASS-called here'.")
    args = ap.parse_args()

    cands = generate_candidates(args.bam, args.ref, args.region,
                                 min_alt_reads=args.min_alt_reads,
                                 min_alt_frac=args.min_alt_frac,
                                 min_depth=args.min_depth,
                                 regions_bed=args.regions_bed)

    with open(args.out, "w") as f:
        f.write("chrom\tpos\tref_base\tdepth\talt_count\talt_frac\n")
        for row in cands:
            f.write("\t".join(str(x) for x in row) + "\n")

    print(f"{len(cands)} candidate sites written to {args.out}")


if __name__ == "__main__":
    main()