"""
Ranked P1 + P2 + P3 candidate generation.

Differences from the older pruning pipeline:
- P2 can use 1-band hits but is capped by ranked quality.
- P3 can retrieve more neighbors.
- The final candidate list is ALWAYS sorted by a deterministic blocker score.
- Supports --ids-file + --limit for small train/validation subsets.
- Writes only compact uint32 candidate IDs.

The three underlying blocker implementations remain the team's existing
validated P1/P2/P3 indexes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from blocking.streaming_blocker_pipeline_opt import (
    P1Source,
    P2Source,
    P3Source,
    merge_sources,
    rrf_score,
)

S1_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


class RankedBinaryWriter:
    def __init__(self, prefix: str, n_rows: int, id_width: int = 96):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

        self.s1_ids = np.lib.format.open_memmap(
            str(prefix) + ".s1_ids.npy",
            mode="w+",
            dtype=f"S{id_width}",
            shape=(n_rows,),
        )
        self.offsets = np.lib.format.open_memmap(
            str(prefix) + ".offsets.npy",
            mode="w+",
            dtype=np.uint64,
            shape=(n_rows + 1,),
        )
        self.offsets[0] = 0
        self.ids = open(
            str(prefix) + ".ids.bin",
            "wb",
            buffering=4 * 1024 * 1024,
        )
        self.row = 0
        self.total = 0

    def write(self, s1_id: str, ids: np.ndarray):
        ids = np.asarray(ids, dtype=np.uint32)
        self.s1_ids[self.row] = str(s1_id).encode("utf-8")
        if ids.size:
            ids.tofile(self.ids)
            self.total += int(ids.size)
        self.row += 1
        self.offsets[self.row] = self.total

    def close(self):
        self.ids.flush()
        self.ids.close()
        self.s1_ids.flush()
        self.offsets.flush()
        del self.s1_ids
        del self.offsets


def read_id_file(path: str) -> set[str]:
    out = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            x = line.strip()
            if x and x != "entity_id":
                out.add(x)
    return out


def count_selected_s1(path: str, wanted: set[str] | None):
    count = 0
    width = 1

    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        idx = header.index("entity_id")

        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= idx:
                continue
            sid = parts[idx]
            if wanted is None or sid in wanted:
                count += 1
                width = max(width, len(sid.encode("utf-8")))

    return count, width


def prepare_s1_rows(
    path: str,
    wanted: set[str] | None,
    limit: int | None,
):
    selected = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=S1_COLUMNS,
        chunksize=25_000,
    ):
        for row in chunk.itertuples(index=False, name=None):
            sid = str(row[0])
            if wanted is not None and sid not in wanted:
                continue

            selected.append(row)

            if limit is not None and len(selected) >= limit:
                return selected

    return selected


def sorted_candidate_ids(
    p1_result,
    p2_result,
    p3_result,
    max_candidates: int,
    p3_slack_quota: float = 0.10,
):
    p1_ids, p1_scores = p1_result
    p2_ids, p2_scores = p2_result
    p3_ids, p3_scores = p3_result

    # Keep the P2 expansion under control before merging. This is the same
    # safety cap used by the validated blocker.
    if len(p2_ids) > 5000:
        order = np.argsort(
            -np.asarray(p2_scores),
            kind="stable",
        )
        take = order[:5000]
        p2_ids = np.asarray(p2_ids, dtype=np.uint32)[take]
        p2_scores = np.asarray(p2_scores)[take]

    (
        ids,
        source_mask,
        r1,
        r2,
        r3,
        p3_scores_merged,
    ) = merge_sources(
        p1_ids,
        p1_scores,
        p2_ids,
        p2_scores,
        p3_ids,
        p3_scores,
    )

    if ids.size == 0:
        return ids

    if ids.size <= max_candidates:
        # Everything fits: keep the complete merged set.
        selected = ids
    else:
        has_p1 = (source_mask & 1) != 0
        has_p2 = (source_mask & 2) != 0
        has_p3 = (source_mask & 4) != 0

        core = has_p1 | has_p2
        core_idx = np.flatnonzero(core)
        p3_only_idx = np.flatnonzero(~core & has_p3)

        # Always reserve a small P3-only slice when the core is too large.
        # This protects semantic-only candidates on noisy/common-token cases.
        p3_slots = min(
            max(1, int(round(max_candidates * p3_slack_quota))),
            int(p3_only_idx.size),
        )
        core_budget = max_candidates - p3_slots

        core_scores = rrf_score(
            source_mask[core_idx],
            r1[core_idx],
            r2[core_idx],
            r3[core_idx],
            p3_weight=1.0,
        )
        core_order = np.argsort(-core_scores, kind="stable")
        chosen_core = core_idx[core_order[:core_budget]]

        if p3_slots:
            p3_order = np.argsort(
                -p3_scores_merged[p3_only_idx],
                kind="stable",
            )
            chosen_p3 = p3_only_idx[p3_order[:p3_slots]]
            chosen_idx = np.concatenate([chosen_core, chosen_p3])
        else:
            chosen_idx = chosen_core

        selected = ids[chosen_idx]

    if len(selected) == 0:
        return selected

    # Now rank the *selected* candidates for the matcher. This preserves the
    # validated candidate membership while giving the model a meaningful
    # blocker rank.
    pos = np.searchsorted(ids, selected)
    score = rrf_score(
        source_mask[pos],
        r1[pos],
        r2[pos],
        r3[pos],
        p3_weight=1.0,
    )
    final_score = score

    # Tiny P3 similarity tiebreaker only; RRF remains dominant.
    finite_p3 = np.isfinite(p3_scores_merged[pos])
    if np.any(finite_p3):
        p3_tie = np.where(
            finite_p3,
            p3_scores_merged[pos],
            0.0,
        ).astype(np.float32)
        final_score = final_score + (0.0005 * p3_tie)

    order = np.argsort(-final_score, kind="stable")
    return np.asarray(selected, dtype=np.uint32)[order]


def run(args):
    wanted = read_id_file(args.ids_file) if args.ids_file else None

    if args.limit is not None:
        rows = prepare_s1_rows(args.s1, wanted, args.limit)
        n_rows = len(rows)
        id_width = max(
            [len(str(r[0]).encode("utf-8")) for r in rows] + [1]
        )
    else:
        n_rows, id_width = count_selected_s1(args.s1, wanted)
        rows = None

    print("=" * 70)
    print("RANKED P1 + P2 + P3 BLOCKER")
    print("=" * 70)
    print(f"Queries        : {n_rows:,}")
    print(f"P2 band hits   : {args.p2_min_band_hits}")
    print(f"P3 top-k       : {args.p3_top_k}")
    print(f"P3 nprobe      : {args.p3_nprobe}")
    print(f"Candidate cap  : {args.max_candidates:,}")
    print(f"P3 slack quota : {args.p3_slack_quota:.0%}")
    print(f"Output         : {args.output_prefix}")

    p1 = P1Source(args.p1_index)
    p2 = P2Source(args.p2_index, args.p2_min_band_hits)
    p3 = P3Source(
        args.p3_index,
        p1.index.entity_ids,
        top_k=args.p3_top_k,
        nprobe=args.p3_nprobe,
        batch_size=args.p3_batch_size,
    )

    writer = RankedBinaryWriter(
        args.output_prefix,
        n_rows,
        max(id_width, 1),
    )

    processed = 0

    def process_batch(batch_rows):
        nonlocal processed

        p1_results = [
            p1.query_scored(str(row[1]), str(row[3]))
            for row in batch_rows
        ]
        p2_results = p2.query_batch(batch_rows)
        p3_results = p3.query_batch(batch_rows)

        for row, a, b, c in zip(
            batch_rows,
            p1_results,
            p2_results,
            p3_results,
        ):
            candidates = sorted_candidate_ids(
                a,
                b,
                c,
                args.max_candidates,
                args.p3_slack_quota,
            )

            writer.write(str(row[0]), candidates)

            processed += 1
            if processed % 1000 == 0:
                print(
                    f"  processed {processed:,}/{n_rows:,}",
                    flush=True,
                )

    if rows is not None:
        for i in range(0, len(rows), args.chunk_size):
            process_batch(rows[i:i + args.chunk_size])
    else:
        for chunk in pd.read_csv(
            args.s1,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=S1_COLUMNS,
            chunksize=args.chunk_size,
        ):
            batch = []
            for row in chunk.itertuples(index=False, name=None):
                sid = str(row[0])
                if wanted is not None and sid not in wanted:
                    continue
                batch.append(row)

            if batch:
                process_batch(batch)

    writer.close()

    print()
    print("=" * 70)
    print("RANKED BLOCKER COMPLETE")
    print("=" * 70)
    print(f"Queries       : {processed:,}")
    print(f"Total pairs   : {writer.total:,}")
    print(f"Mean/query    : {writer.total / processed if processed else 0:.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--s1", required=True)
    parser.add_argument("--ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--p1-index", required=True)
    parser.add_argument("--p2-index", required=True)
    parser.add_argument("--p3-index", required=True)

    parser.add_argument("--max-candidates", type=int, default=1500)
    parser.add_argument("--p2-min-band-hits", type=int, default=2)
    parser.add_argument("--p3-top-k", type=int, default=500)
    parser.add_argument("--p3-nprobe", type=int, default=128)
    parser.add_argument("--p3-batch-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--p3-slack-quota", type=float, default=0.10)

    parser.add_argument(
        "--output-prefix",
        required=True,
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
