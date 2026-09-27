"""
Streaming 3-way candidate-union evaluator.

Usage:
    python -m blocking.evaluate_union_3 \
        --candidates-a blocking/candidates_token_opt20.tsv \
        --candidates-b blocking/candidates_lsh_batched.tsv \
        --candidates-c blocking/candidates_embed_500_nprobe128.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --dev-s1 common/dev_s1.tsv
"""

import argparse
import statistics
from pathlib import Path


CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids"
GROUND_TRUTH_HEADER = "source1_entity_id\tmatched_entity_ids"


def load_dev_ids(path):
    dev_ids = set()
    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        try:
            idx = header.index("entity_id")
        except ValueError as exc:
            raise ValueError(f"{path} is missing entity_id") from exc

        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > idx and parts[idx]:
                dev_ids.add(parts[idx])

    return dev_ids


def load_ground_truth(path, dev_ids):
    truth = {}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")

        try:
            id_idx = header.index("source1_entity_id")
            match_idx = header.index("matched_entity_ids")
        except ValueError as exc:
            raise ValueError(
                f"{path} must contain source1_entity_id and matched_entity_ids"
            ) from exc

        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= max(id_idx, match_idx):
                continue

            s1_id = parts[id_idx]
            if s1_id not in dev_ids:
                continue

            raw = parts[match_idx]
            truth[s1_id] = (
                set(raw.split(",")) if raw else set()
            )

    return truth


def open_candidate_file(path):
    fh = open(path, "r", encoding="utf-8", newline="")
    header = fh.readline().rstrip("\r\n")

    if header != CANDIDATE_HEADER:
        fh.close()
        raise ValueError(
            f"{path}: unexpected header {header!r}; "
            f"expected {CANDIDATE_HEADER!r}"
        )

    return fh


def read_candidate_row(fh, path):
    line = fh.readline()

    if not line:
        return None

    parts = line.rstrip("\r\n").split("\t", 1)
    if len(parts) != 2:
        raise ValueError(f"{path}: malformed candidate row")

    s1_id, raw = parts

    candidates = set(raw.split(",")) if raw else set()
    return s1_id, candidates


def percentile(sorted_values, q):
    if not sorted_values:
        return 0

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower

    return (
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def evaluate(
    candidates_a,
    candidates_b,
    candidates_c,
    ground_truth,
    dev_s1,
):
    print("Loading 10k dev S1 IDs...")
    dev_ids = load_dev_ids(dev_s1)
    print(f"Dev S1 IDs: {len(dev_ids):,}")

    print("Loading only matching ground-truth rows...")
    truth = load_ground_truth(ground_truth, dev_ids)
    print(f"Ground-truth rows retained: {len(truth):,}")

    total_true_pairs = 0
    found_true_pairs = 0
    entities_with_match = 0
    fully_recalled = 0

    candidate_counts = []
    evaluated = 0

    fa = fb = fc = None

    try:
        fa = open_candidate_file(candidates_a)
        fb = open_candidate_file(candidates_b)
        fc = open_candidate_file(candidates_c)

        while True:
            row_a = read_candidate_row(fa, candidates_a)
            row_b = read_candidate_row(fb, candidates_b)
            row_c = read_candidate_row(fc, candidates_c)

            if row_a is None and row_b is None and row_c is None:
                break

            if row_a is None or row_b is None or row_c is None:
                raise ValueError(
                    "Candidate files have different numbers of rows."
                )

            id_a, cand_a = row_a
            id_b, cand_b = row_b
            id_c, cand_c = row_c

            if not (id_a == id_b == id_c):
                raise ValueError(
                    "Candidate files are not aligned: "
                    f"{id_a!r}, {id_b!r}, {id_c!r}"
                )

            s1_id = id_a

            if s1_id not in dev_ids:
                continue

            true_matches = truth.get(s1_id, set())

            union = cand_a
            union.update(cand_b)
            union.update(cand_c)

            candidate_counts.append(len(union))

            if true_matches:
                entities_with_match += 1
                total_true_pairs += len(true_matches)

                found = len(true_matches.intersection(union))
                found_true_pairs += found

                if found == len(true_matches):
                    fully_recalled += 1

            evaluated += 1

            if evaluated % 1000 == 0:
                print(f"  evaluated={evaluated:,}", flush=True)

    finally:
        for fh in (fa, fb, fc):
            if fh is not None:
                fh.close()

    if not evaluated:
        raise RuntimeError("No development rows were evaluated.")

    candidate_counts.sort()

    pairwise_recall = (
        found_true_pairs / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    full_recall = (
        fully_recalled / entities_with_match
        if entities_with_match
        else 0.0
    )

    mean_candidates = statistics.fmean(candidate_counts)

    print()
    print("=" * 70)
    print("STREAMING 3-WAY UNION BLOCKING RECALL REPORT")
    print("=" * 70)
    print(f"Queries evaluated:            {evaluated:,}")
    print(f"Entities with >=1 true match: {entities_with_match:,}")
    print(f"Total true pairs:              {total_true_pairs:,}")
    print(f"Found true pairs:              {found_true_pairs:,}")
    print(f"Pairwise recall:               {pairwise_recall:.6f}")
    print(f"Full-recall rate:              {full_recall:.6f}")
    print()
    print("Union candidate count distribution:")
    print(f"  mean:   {mean_candidates:.2f}")
    print(f"  median: {statistics.median(candidate_counts):.0f}")
    print(f"  p90:    {percentile(candidate_counts, 0.90):.0f}")
    print(f"  max:    {max(candidate_counts):,}")
    print()
    print("Input files:")
    print(f"  A: {candidates_a}")
    print(f"  B: {candidates_b}")
    print(f"  C: {candidates_c}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Streaming 3-way candidate union evaluator."
    )
    parser.add_argument("--candidates-a", required=True)
    parser.add_argument("--candidates-b", required=True)
    parser.add_argument("--candidates-c", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--dev-s1", required=True)

    args = parser.parse_args()

    evaluate(
        candidates_a=args.candidates_a,
        candidates_b=args.candidates_b,
        candidates_c=args.candidates_c,
        ground_truth=args.ground_truth,
        dev_s1=args.dev_s1,
    )


if __name__ == "__main__":
    main()
