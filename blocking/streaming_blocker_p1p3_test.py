
"""Fast P1 + P3 ranked test candidate generation.

Used for emergency final submission when P2 indexing is too slow.
Keeps every country-correct P1 candidate plus a reserved P3-only slice,
then ranks by agreement-aware RRF.
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
    P3Source,
    merge_sources,
    rrf_score,
)

S1_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


class Writer:
    def __init__(self, prefix, n_rows, id_width=96):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.s1_ids = np.lib.format.open_memmap(
            str(prefix) + ".s1_ids.npy", mode="w+",
            dtype=f"S{id_width}", shape=(n_rows,)
        )
        self.offsets = np.lib.format.open_memmap(
            str(prefix) + ".offsets.npy", mode="w+",
            dtype=np.uint64, shape=(n_rows + 1,)
        )
        self.offsets[0] = 0
        self.fh = open(str(prefix) + ".ids.bin", "wb", buffering=4 * 1024 * 1024)
        self.row = 0
        self.total = 0
        self.prefix = str(prefix)

    def write(self, sid, ids):
        ids = np.asarray(ids, dtype=np.uint32)
        self.s1_ids[self.row] = str(sid).encode("utf-8")
        if ids.size:
            ids.tofile(self.fh)
            self.total += int(ids.size)
        self.row += 1
        self.offsets[self.row] = self.total

    def close(self):
        self.fh.flush()
        self.fh.close()
        self.s1_ids.flush()
        self.offsets.flush()
        del self.s1_ids, self.offsets


def count_s1(path):
    n = 0
    width = 1
    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        idx = header.index("entity_id")
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > idx:
                n += 1
                width = max(width, len(parts[idx].encode("utf-8")))
    return n, width


def rerank(p1_result, p3_result, cap, p3_slack):
    p1_ids, p1_scores = p1_result
    p3_ids, p3_scores = p3_result

    empty_u32 = np.empty(0, dtype=np.uint32)
    empty_f32 = np.empty(0, dtype=np.float32)

    merged = merge_sources(
        p1_ids, p1_scores,
        empty_u32, np.empty(0, dtype=np.uint32),
        p3_ids, p3_scores,
    )
    ids, source_mask, r1, r2, r3, p3_scores_m = merged

    if ids.size <= cap:
        chosen = np.arange(ids.size, dtype=np.int64)
    else:
        has_p1 = (source_mask & 1) != 0
        has_p3 = (source_mask & 4) != 0
        core_idx = np.flatnonzero(has_p1)
        p3_only_idx = np.flatnonzero(~has_p1 & has_p3)

        p3_slots = min(
            int(p3_only_idx.size),
            max(1, int(round(cap * p3_slack))),
        )
        core_budget = cap - p3_slots

        core_score = rrf_score(
            source_mask[core_idx],
            r1[core_idx],
            r2[core_idx],
            r3[core_idx],
            p3_weight=1.0,
        )
        core_order = np.argsort(-core_score, kind="stable")
        chosen_core = core_idx[core_order[:core_budget]]

        if p3_slots:
            p3_order = np.argsort(-p3_scores_m[p3_only_idx], kind="stable")
            chosen_p3 = p3_only_idx[p3_order[:p3_slots]]
            chosen = np.concatenate((chosen_core, chosen_p3))
        else:
            chosen = chosen_core

    if chosen.size == 0:
        return empty_u32

    score = rrf_score(
        source_mask[chosen],
        r1[chosen],
        r2[chosen],
        r3[chosen],
        p3_weight=1.0,
    )
    finite = np.isfinite(p3_scores_m[chosen])
    score = score + np.where(finite, 0.0005 * p3_scores_m[chosen], 0.0)

    order = np.argsort(-score, kind="stable")
    return np.asarray(ids[chosen][order], dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--p1-index", required=True)
    ap.add_argument("--p3-index", required=True)
    ap.add_argument("--max-candidates", type=int, default=1500)
    ap.add_argument("--p3-top-k", type=int, default=300)
    ap.add_argument("--p3-nprobe", type=int, default=128)
    ap.add_argument("--p3-batch-size", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--p3-slack-quota", type=float, default=0.10)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    n_rows, id_width = count_s1(args.s1)
    print(f"Test S1 queries: {n_rows:,}", flush=True)
    print("Loading P1...", flush=True)
    p1 = P1Source(args.p1_index)
    print("Loading P3...", flush=True)
    p3 = P3Source(
        args.p3_index,
        p1.index.entity_ids,
        top_k=args.p3_top_k,
        nprobe=args.p3_nprobe,
        batch_size=args.p3_batch_size,
    )

    writer = Writer(args.output_prefix, n_rows, max(id_width, 1))
    processed = 0

    for chunk in pd.read_csv(
        args.s1, sep="\t", dtype=str, keep_default_na=False,
        usecols=S1_COLUMNS, chunksize=args.chunk_size
    ):
        rows = list(chunk.itertuples(index=False, name=None))
        p1_results = [
            p1.query_scored(str(r[1]), str(r[3])) for r in rows
        ]
        p3_results = p3.query_batch(rows)

        for row, a, c in zip(rows, p1_results, p3_results):
            ids = rerank(a, c, args.max_candidates, args.p3_slack_quota)
            writer.write(row[0], ids)
            processed += 1
            if processed % 10000 == 0:
                print(
                    f"  processed {processed:,}/{n_rows:,} | "
                    f"avg={writer.total / processed:.1f}",
                    flush=True,
                )

    writer.close()
    print("=" * 70)
    print("P1 + P3 TEST BLOCKER COMPLETE")
    print("=" * 70)
    print(f"Queries   : {processed:,}")
    print(f"Pairs     : {writer.total:,}")
    print(f"Mean/query: {writer.total / processed if processed else 0:.2f}")


if __name__ == "__main__":
    main()
