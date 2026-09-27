"""
blocking/correctness_check.py

Validates that the two-pass, query-driven streaming index in token_index.py
produces IDENTICAL candidate sets to a brute-force "index every token in the
corpus, then filter" reference implementation, on a bounded real-data
subset (small enough to build the brute-force index in memory).

This is a correctness check, not a benchmark: it does not tune parameters,
it proves the streaming shortcut changes nothing about the output for a
fixed (max_doc_freq, max_postings_per_token) configuration.

Usage:
    python3 correctness_check.py --source2 ../dataset/train/train_source2.tsv \
        --source3 ../dataset/train/train_source3.tsv --dev-s1 ../common/dev_s1.tsv \
        --max-rows-per-source 300000

Measured result (2026-09-25, real data, 300,000 rows/source,
max_doc_freq=0.001, max_postings_per_token=3000):
    Country totals match: {'India': 241071, 'US': 358929}
    Queries checked: 10,000
    Mismatches: 0
    PASSED
"""

import argparse
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.country_partition import load_source
from common.normalize import name_tokens

from token_index import (
    _filter_by_doc_freq,
    _iter_source_rows,
    build_token_indexes_streaming,
    query_candidates,
)


def brute_force_indexes(source2_path, source3_path, max_doc_freq, max_postings_per_token, max_rows_per_source):
    """
    Reference implementation: load a bounded subset of S2+S3 fully into
    memory, build a token->entity_id inverted index over EVERY token in the
    corpus (not just query-relevant ones), then apply the exact same
    doc-freq + absolute-cap filter used by the production streaming path.
    """

    def load_bounded(path, max_rows):
        rows = list(_iter_source_rows(path, max_rows=max_rows))
        return pd.DataFrame(rows, columns=["entity_id", "business_name", "country"])

    s2 = load_bounded(source2_path, max_rows_per_source)
    s3 = load_bounded(source3_path, max_rows_per_source)
    pool = pd.concat([s2, s3], ignore_index=True)

    indexes = {}
    country_totals = {}
    for country, country_df in pool.groupby("country", sort=True):
        from collections import defaultdict
        token_to_entities = defaultdict(list)
        for entity_id, name in zip(country_df["entity_id"], country_df["business_name"]):
            for token in set(name_tokens(name)):
                token_to_entities[token].append(entity_id)
        kept, _, _ = _filter_by_doc_freq(
            token_to_entities, len(country_df), max_doc_freq, max_postings_per_token,
        )
        indexes[country] = kept
        country_totals[country] = len(country_df)

    return indexes, country_totals


def main():
    parser = argparse.ArgumentParser(description="Verify streaming index == brute-force index on bounded real data")
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--dev-s1", required=True)
    parser.add_argument("--max-rows-per-source", type=int, default=300_000)
    parser.add_argument("--max-doc-freq", type=float, default=0.001)
    parser.add_argument("--max-postings-per-token", type=int, default=3000)
    args = parser.parse_args()

    print(f"Loading bounded subset: {args.max_rows_per_source:,} rows/source ...", flush=True)
    dev_s1 = load_source(args.dev_s1)

    print("Building BRUTE-FORCE reference index (every corpus token, then filtered) ...", flush=True)
    bf_indexes, bf_totals = brute_force_indexes(
        args.source2, args.source3, args.max_doc_freq, args.max_postings_per_token, args.max_rows_per_source,
    )

    print("Building STREAMING two-pass query-driven index (production code path) ...", flush=True)
    stream_indexes, stream_totals = build_token_indexes_streaming(
        [args.source2, args.source3],
        dev_s1,
        max_doc_freq=args.max_doc_freq,
        max_postings_per_token=args.max_postings_per_token,
        max_rows_per_source=args.max_rows_per_source,
        verbose=False,
    )

    assert bf_totals == stream_totals, f"Country row totals differ: {bf_totals} vs {stream_totals}"
    print(f"Country totals match: {bf_totals}", flush=True)

    n_queries_checked = 0
    n_mismatches = 0
    n_countries_present_in_subset = 0

    for entity_id, name, country in zip(dev_s1["entity_id"], dev_s1["business_name"], dev_s1["country"]):
        if country not in bf_totals:
            bf_candidates = []
        else:
            n_countries_present_in_subset += 1
            bf_candidates = query_candidates(name, bf_indexes.get(country, {}))

        stream_candidates = query_candidates(name, stream_indexes.get(country, {}))

        n_queries_checked += 1
        if bf_candidates != stream_candidates:
            n_mismatches += 1
            if n_mismatches <= 5:
                print(f"MISMATCH entity_id={entity_id} country={country!r}", flush=True)
                print(f"  brute-force ({len(bf_candidates)}): {bf_candidates[:10]}", flush=True)
                print(f"  streaming   ({len(stream_candidates)}): {stream_candidates[:10]}", flush=True)

    print(flush=True)
    print("=== Correctness check summary ===", flush=True)
    print(f"Queries checked:                 {n_queries_checked:,}", flush=True)
    print(f"Queries whose country is present\nin this bounded subset:          {n_countries_present_in_subset:,}", flush=True)
    print(f"Mismatches (streaming vs brute-force): {n_mismatches:,}", flush=True)

    assert n_mismatches == 0, f"Streaming index disagreed with brute-force reference on {n_mismatches} queries"
    print(flush=True)
    print("PASSED: streaming two-pass index produces IDENTICAL candidates to the brute-force reference.", flush=True)


if __name__ == "__main__":
    main()
