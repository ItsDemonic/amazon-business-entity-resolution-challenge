"""
Memory-efficient blocking recall evaluator.

Designed for very large candidate files.

Unlike evaluate_recall.py, this version:
- loads only the 10k dev S1 ground truth into memory
- streams the candidate TSV one S1 row at a time
- never builds a giant candidate dictionary
- keeps only candidate counts and recall statistics in memory

Usage:

python -m blocking.evaluate_recall_streaming \
    --candidates blocking/candidates_lsh.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --dev-s1 common/dev_s1.tsv \
    --pool-size 10320219
"""

import argparse
from collections import defaultdict
from statistics import mean, median

import pandas as pd


def load_dev_ids(dev_s1_path):
    """
    Load only the 10k S1 IDs that were actually queried.
    """
    df = pd.read_csv(
        dev_s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "country"],
    )

    return (
        set(df["entity_id"]),
        dict(zip(df["entity_id"], df["country"])),
    )


def load_filtered_ground_truth(
    ground_truth_path,
    dev_ids,
):
    """
    Stream the full ground truth file but retain only rows
    belonging to the 10k dev S1 IDs.
    """

    ground_truth = {}

    for chunk in pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    ):
        chunk = chunk[
            chunk["source1_entity_id"].isin(dev_ids)
        ]

        for source1_id, matched in zip(
            chunk["source1_entity_id"],
            chunk["matched_entity_ids"],
        ):
            if matched:
                ground_truth[source1_id] = set(
                    matched.split(",")
                )
            else:
                ground_truth[source1_id] = set()

    return ground_truth


def evaluate_streaming(
    candidates_path,
    ground_truth,
    country_lookup,
    pool_size=None,
):
    """
    Stream the candidate TSV row-by-row.

    Only one candidate row is materialized at a time.
    """

    candidate_counts = []

    total_true_pairs = 0
    found_true_pairs = 0

    entities_with_matches = 0
    entities_fully_recalled = 0

    zero_candidates = 0

    per_country = defaultdict(
        lambda: {
            "total_true_pairs": 0,
            "found_true_pairs": 0,
            "entities_with_matches": 0,
            "entities_fully_recalled": 0,
        }
    )

    missed_examples = []

    queries_evaluated = 0

    with open(
        candidates_path,
        "r",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as fh:

        # Skip header.
        header = next(fh, None)

        for line_number, line in enumerate(
            fh,
            start=2,
        ):
            line = line.rstrip("\n\r")

            if not line:
                continue

            # Split only on the first tab.
            parts = line.split("\t", 1)

            if len(parts) != 2:
                continue

            source1_id = parts[0]
            candidate_string = parts[1]

            # This is the important part:
            # only the current row exists in memory.
            if candidate_string:
                candidate_ids = set(
                    candidate_string.split(",")
                )
            else:
                candidate_ids = set()

            candidate_count = len(candidate_ids)

            candidate_counts.append(
                candidate_count
            )

            queries_evaluated += 1

            if candidate_count == 0:
                zero_candidates += 1

            true_matches = ground_truth.get(
                source1_id,
                set(),
            )

            if not true_matches:
                continue

            entities_with_matches += 1

            total_true_pairs += len(
                true_matches
            )

            found = (
                true_matches
                & candidate_ids
            )

            found_true_pairs += len(found)

            fully_recalled = (
                len(found)
                == len(true_matches)
            )

            if fully_recalled:
                entities_fully_recalled += 1
            elif len(missed_examples) < 10:
                missed_examples.append(
                    {
                        "source1_entity_id": source1_id,
                        "missed": sorted(
                            true_matches
                            - candidate_ids
                        ),
                        "n_true": len(true_matches),
                        "n_found": len(found),
                    }
                )

            country = country_lookup.get(
                source1_id,
                "UNKNOWN",
            )

            bucket = per_country[country]

            bucket["total_true_pairs"] += len(
                true_matches
            )

            bucket["found_true_pairs"] += len(
                found
            )

            bucket["entities_with_matches"] += 1

            if fully_recalled:
                bucket[
                    "entities_fully_recalled"
                ] += 1

            if queries_evaluated % 1000 == 0:
                print(
                    f"  evaluated="
                    f"{queries_evaluated:,}",
                    flush=True,
                )

    if total_true_pairs:
        pairwise_recall = (
            found_true_pairs
            / total_true_pairs
        )
    else:
        pairwise_recall = None

    if entities_with_matches:
        full_recall = (
            entities_fully_recalled
            / entities_with_matches
        )
    else:
        full_recall = None

    sorted_counts = sorted(
        candidate_counts
    )

    n = len(sorted_counts)

    def percentile(p):
        if n == 0:
            return 0

        index = min(
            n - 1,
            int(round(p * (n - 1))),
        )

        return sorted_counts[index]

    print()
    print("=" * 70)
    print("STREAMING BLOCKING RECALL REPORT")
    print("=" * 70)

    print(
        f"Queries evaluated:            "
        f"{queries_evaluated:,}"
    )

    print(
        f"Entities with >=1 true match: "
        f"{entities_with_matches:,}"
    )

    print(
        f"Total true pairs:              "
        f"{total_true_pairs:,}"
    )

    print(
        f"Found true pairs:              "
        f"{found_true_pairs:,}"
    )

    if pairwise_recall is not None:
        print(
            f"Pairwise recall:               "
            f"{pairwise_recall:.6f}"
        )

    if full_recall is not None:
        print(
            f"Full-recall rate:              "
            f"{full_recall:.6f}"
        )

    print()
    print("Candidate count distribution:")

    print(
        f"  mean:   "
        f"{mean(candidate_counts):.2f}"
    )

    print(
        f"  median: "
        f"{median(candidate_counts)}"
    )

    print(
        f"  p90:    "
        f"{percentile(0.90)}"
    )

    print(
        f"  max:    "
        f"{max(candidate_counts)}"
    )

    print(
        f"  zero:   "
        f"{zero_candidates:,}"
    )

    if pool_size:
        reduction_ratio = (
            1
            - (
                mean(candidate_counts)
                / pool_size
            )
        )

        print(
            f"  reduction ratio: "
            f"{reduction_ratio:.6f}"
        )

    print()
    print("Per-country breakdown:")

    for country, stats in sorted(
        per_country.items()
    ):
        if stats["total_true_pairs"]:
            country_pairwise = (
                stats["found_true_pairs"]
                / stats["total_true_pairs"]
            )
        else:
            country_pairwise = None

        if stats["entities_with_matches"]:
            country_full = (
                stats[
                    "entities_fully_recalled"
                ]
                / stats[
                    "entities_with_matches"
                ]
            )
        else:
            country_full = None

        pairwise_str = (
            f"{country_pairwise:.6f}"
            if country_pairwise is not None
            else "n/a"
        )

        full_str = (
            f"{country_full:.6f}"
            if country_full is not None
            else "n/a"
        )

        print(
            f"  {country}: "
            f"pairwise_recall={pairwise_str}, "
            f"full_recall={full_str}, "
            f"n_entities="
            f"{stats['entities_with_matches']}"
        )

    if missed_examples:
        print()
        print("Example misses:")

        for ex in missed_examples:
            print(
                f"  {ex['source1_entity_id']}: "
                f"found "
                f"{ex['n_found']}/"
                f"{ex['n_true']}, "
                f"missed="
                f"{ex['missed']}"
            )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Memory-efficient blocking recall "
            "evaluation."
        )
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Candidate-pairs TSV.",
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Official train ground truth TSV.",
    )

    parser.add_argument(
        "--dev-s1",
        required=True,
        help="10k dev S1 TSV.",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Full S2+S3 pool size.",
    )

    args = parser.parse_args()

    print(
        "Loading 10k dev S1 IDs...",
        flush=True,
    )

    dev_ids, country_lookup = (
        load_dev_ids(args.dev_s1)
    )

    print(
        f"Dev S1 IDs: {len(dev_ids):,}",
        flush=True,
    )

    print(
        "Loading only matching ground-truth rows...",
        flush=True,
    )

    ground_truth = (
        load_filtered_ground_truth(
            args.ground_truth,
            dev_ids,
        )
    )

    print(
        f"Ground-truth rows retained: "
        f"{len(ground_truth):,}",
        flush=True,
    )

    print(
        "Streaming candidate file...",
        flush=True,
    )

    evaluate_streaming(
        candidates_path=args.candidates,
        ground_truth=ground_truth,
        country_lookup=country_lookup,
        pool_size=args.pool_size,
    )


if __name__ == "__main__":
    main()