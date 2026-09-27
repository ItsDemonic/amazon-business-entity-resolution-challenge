"""
blocking/evaluate_recall.py

Measures the blocking recall ceiling for a candidate-pairs TSV (produced by
token_index.py, minhash_lsh.py, embedding_faiss.py, or a union of them)
against the official train_ground_truth.tsv.

This is Stage 4B / Person 1's "first experiments" requirement: never claim
recall for a blocking method until it has actually been measured here.

Metrics reported:
  - pairwise recall: fraction of all true (S1, matched) pairs that appear
    in the candidate lists.
  - full-recall rate: fraction of S1 entities (with >=1 true match) for
    which ALL true matches were captured by blocking.
  - candidate-count distribution: mean/median/p90/max candidates per query,
    and reduction ratio vs. the full S2+S3 candidate pool size.
  - optional per-country breakdown (requires --dev-s1 for country lookup).

Usage:
    python evaluate_recall.py \
        --candidates candidates_token.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --dev-s1 common/dev_s1.tsv \
        --pool-size 9000000

Smoke test (tiny synthetic data, no real files needed):
    python evaluate_recall.py --smoke-test
"""

import argparse
import os
import sys
from collections import defaultdict
from statistics import mean, median

import pandas as pd

# Resolve the repo root from this file's location (robust alternative to a
# relative `sys.path.insert(0, "..")`, which only worked when this script was
# invoked with cwd == blocking/). Matches the fix already applied in
# token_index.py so both scripts behave consistently regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.io_schema import read_candidate_pairs


def load_ground_truth(path):
    """
    Load train_ground_truth.tsv into dict: source1_entity_id -> set(matched_ids)
    Rows with an empty matched_entity_ids field map to an empty set.
    """

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    ground_truth = {}

    for source1_id, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        if matched:
            ground_truth[source1_id] = set(matched.split(","))
        else:
            ground_truth[source1_id] = set()

    return ground_truth


def load_candidates(path):
    """
    Load a candidate-pairs TSV into dict: source1_entity_id -> set(candidate_ids)
    """

    df = read_candidate_pairs(path)

    candidates = {}

    for source1_id, candidate_ids in zip(df["source1_entity_id"], df["candidate_entity_ids"]):
        if candidate_ids:
            candidates[source1_id] = set(candidate_ids.split(","))
        else:
            candidates[source1_id] = set()

    return candidates


def load_country_lookup(dev_s1_path):
    """
    Load dev_s1.tsv into dict: entity_id -> country (for per-country breakdown).
    """

    df = pd.read_csv(
        dev_s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    return dict(zip(df["entity_id"], df["country"]))


def evaluate(candidates, ground_truth, country_lookup=None, pool_size=None):
    """
    candidates: dict[source1_id, set(candidate_ids)]
    ground_truth: dict[source1_id, set(true_match_ids)]

    Only S1 entities present in `candidates` (i.e. entities that were
    actually queried during blocking) are evaluated.
    """

    total_true_pairs = 0
    found_true_pairs = 0

    entities_with_matches = 0
    entities_fully_recalled = 0

    candidate_counts = []

    per_country = defaultdict(lambda: {
        "total_true_pairs": 0,
        "found_true_pairs": 0,
        "entities_with_matches": 0,
        "entities_fully_recalled": 0,
    })

    missed_examples = []

    for source1_id, candidate_ids in candidates.items():
        candidate_counts.append(len(candidate_ids))

        true_matches = ground_truth.get(source1_id, set())

        if not true_matches:
            continue

        entities_with_matches += 1
        total_true_pairs += len(true_matches)

        found = true_matches & candidate_ids
        found_true_pairs += len(found)

        fully_recalled = len(found) == len(true_matches)
        if fully_recalled:
            entities_fully_recalled += 1
        else:
            missed_examples.append({
                "source1_entity_id": source1_id,
                "missed": sorted(true_matches - candidate_ids),
                "n_true": len(true_matches),
                "n_found": len(found),
            })

        if country_lookup is not None:
            country = country_lookup.get(source1_id, "UNKNOWN")
            bucket = per_country[country]
            bucket["total_true_pairs"] += len(true_matches)
            bucket["found_true_pairs"] += len(found)
            bucket["entities_with_matches"] += 1
            if fully_recalled:
                bucket["entities_fully_recalled"] += 1

    pairwise_recall = (found_true_pairs / total_true_pairs) if total_true_pairs else None
    full_recall_rate = (entities_fully_recalled / entities_with_matches) if entities_with_matches else None

    sorted_counts = sorted(candidate_counts)
    n = len(sorted_counts)

    def percentile(p):
        if n == 0:
            return 0
        idx = min(n - 1, int(round(p * (n - 1))))
        return sorted_counts[idx]

    result = {
        "n_queries_evaluated": len(candidates),
        "entities_with_true_matches": entities_with_matches,
        "total_true_pairs": total_true_pairs,
        "found_true_pairs": found_true_pairs,
        "pairwise_recall": pairwise_recall,
        "full_recall_rate": full_recall_rate,
        "candidate_count_mean": mean(candidate_counts) if candidate_counts else 0,
        "candidate_count_median": median(candidate_counts) if candidate_counts else 0,
        "candidate_count_p90": percentile(0.9),
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "missed_examples": missed_examples[:10],
    }

    if pool_size:
        result["reduction_ratio"] = 1 - (result["candidate_count_mean"] / pool_size)

    if country_lookup is not None:
        result["per_country"] = {}
        for country, bucket in per_country.items():
            result["per_country"][country] = {
                "pairwise_recall": (bucket["found_true_pairs"] / bucket["total_true_pairs"])
                    if bucket["total_true_pairs"] else None,
                "full_recall_rate": (bucket["entities_fully_recalled"] / bucket["entities_with_matches"])
                    if bucket["entities_with_matches"] else None,
                "entities_with_true_matches": bucket["entities_with_matches"],
            }

    return result


def print_report(result):
    print("=== Blocking recall report ===")
    print(f"Queries evaluated:            {result['n_queries_evaluated']:,}")
    print(f"Entities with >=1 true match: {result['entities_with_true_matches']:,}")
    print(f"Total true pairs:             {result['total_true_pairs']:,}")
    print(f"Found true pairs:             {result['found_true_pairs']:,}")

    if result["pairwise_recall"] is not None:
        print(f"Pairwise recall:              {result['pairwise_recall']:.4f}")
    if result["full_recall_rate"] is not None:
        print(f"Full-recall rate (entities):  {result['full_recall_rate']:.4f}")

    print()
    print("Candidate count distribution:")
    print(f"  mean:   {result['candidate_count_mean']:.2f}")
    print(f"  median: {result['candidate_count_median']}")
    print(f"  p90:    {result['candidate_count_p90']}")
    print(f"  max:    {result['candidate_count_max']}")

    if "reduction_ratio" in result:
        print(f"  reduction ratio vs. full pool: {result['reduction_ratio']:.6f}")

    if "per_country" in result:
        print()
        print("Per-country breakdown:")
        for country, stats in sorted(result["per_country"].items()):
            recall_str = f"{stats['pairwise_recall']:.4f}" if stats["pairwise_recall"] is not None else "n/a"
            full_str = f"{stats['full_recall_rate']:.4f}" if stats["full_recall_rate"] is not None else "n/a"
            print(f"  {country}: pairwise_recall={recall_str}, full_recall_rate={full_str}, "
                  f"n_entities={stats['entities_with_true_matches']}")

    if result["missed_examples"]:
        print()
        print(f"Example misses (up to 10 of {len(result['missed_examples'])}+ shown):")
        for ex in result["missed_examples"]:
            print(f"  {ex['source1_entity_id']}: found {ex['n_found']}/{ex['n_true']}, "
                  f"missed={ex['missed']}")


def _run_smoke_test():
    import tempfile
    import os

    candidate_rows = [
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1"},
        {"source1_entity_id": "S1-2", "candidate_entity_ids": "S3-3"},
        {"source1_entity_id": "S1-3", "candidate_entity_ids": ""},
    ]
    ground_truth_rows = [
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S3-1,S2-5"},  # partial: 1/2 found
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S3-3"},        # full: 1/1 found
        {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-99"},       # zero: 0/1 found
    ]
    dev_s1_rows = [
        {"entity_id": "S1-1", "business_name": "x", "business_address": "x", "country": "India"},
        {"entity_id": "S1-2", "business_name": "x", "business_address": "x", "country": "US"},
        {"entity_id": "S1-3", "business_name": "x", "business_address": "x", "country": "France"},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        cand_path = os.path.join(tmp, "candidates.tsv")
        gt_path = os.path.join(tmp, "ground_truth.tsv")
        dev_path = os.path.join(tmp, "dev_s1.tsv")

        pd.DataFrame(candidate_rows).to_csv(cand_path, sep="\t", index=False)
        pd.DataFrame(ground_truth_rows).to_csv(gt_path, sep="\t", index=False)
        pd.DataFrame(dev_s1_rows).to_csv(dev_path, sep="\t", index=False)

        candidates = load_candidates(cand_path)
        ground_truth = load_ground_truth(gt_path)
        country_lookup = load_country_lookup(dev_path)

        result = evaluate(candidates, ground_truth, country_lookup=country_lookup, pool_size=6)
        print_report(result)

        # total true pairs = 2 + 1 + 1 = 4; found = 1 + 1 + 0 = 2 -> recall 0.5
        assert result["total_true_pairs"] == 4
        assert result["found_true_pairs"] == 2
        assert abs(result["pairwise_recall"] - 0.5) < 1e-9
        # only S1-2 fully recalled out of 3 entities-with-matches -> 1/3
        assert abs(result["full_recall_rate"] - (1 / 3)) < 1e-9
        assert result["per_country"]["US"]["full_recall_rate"] == 1.0
        assert result["per_country"]["France"]["pairwise_recall"] == 0.0

        print()
        print("All smoke test assertions passed.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate blocking recall against ground truth")
    parser.add_argument("--candidates", help="Path to a candidate-pairs TSV")
    parser.add_argument("--ground-truth", help="Path to train_ground_truth.tsv")
    parser.add_argument("--dev-s1", help="Optional path to common/dev_s1.tsv for per-country breakdown")
    parser.add_argument("--pool-size", type=int, default=None,
                         help="Optional full S2+S3 pool size, to report the reduction ratio")
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    if not args.candidates or not args.ground_truth:
        parser.error("--candidates and --ground-truth are required (or pass --smoke-test)")

    candidates = load_candidates(args.candidates)
    ground_truth = load_ground_truth(args.ground_truth)
    country_lookup = load_country_lookup(args.dev_s1) if args.dev_s1 else None

    result = evaluate(candidates, ground_truth, country_lookup=country_lookup, pool_size=args.pool_size)
    print_report(result)


if __name__ == "__main__":
    main()
