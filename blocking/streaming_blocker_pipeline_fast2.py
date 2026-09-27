"""
Fast streaming P1 + P2 + P3 blocker pipeline.

Design goals:
    - no raw candidate-union TSV
    - no per-candidate Python dict/set for the union
    - keep blocker IDs as compact uint32 internal IDs
    - reuse the already-built P1/P2/P3 indexes
    - batch P2/P3 work
    - resolve public entity IDs only after pruning
    - optionally stream the pruned candidates to TSV
    - optionally evaluate dev recall in the same pass

IMPORTANT:
P1, P2, and P3 were all built from S2 followed by S3 in the same row order,
so their internal candidate IDs refer to the same 10.32M-record universe.
That lets us merge them directly as uint32 IDs without converting millions of
IDs to Python strings/bytes.

Validated settings:
    P1: optimized hybrid token index, fallback_threshold=20
    P2: MinHash ngram=3, threshold=.3, 64 perms, min_band_hits=2
    P3: top_k=500, nprobe=128
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


S1_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


class P1Source:
    def __init__(self, index_dir):
        import blocking.token_index as mod

        self.mod = mod
        self.index = mod.HybridTokenIndex(index_dir)

    def query(self, name, country):
        """
        Return (internal_ids, p1_strength).

        Reuses the validated P1 threshold-20 fallback implementation. The
        strength is only used for pruning; candidate membership is unchanged.
        """
        mod = self.mod
        idx = self.index

        tokens = mod._token_set(name)

        arrays = []
        for token in tokens:
            key_id = idx.token_key_to_id.get(
                (country, token)
            )
            if key_id is not None:
                arrays.append(
                    idx._slice(
                        idx.token_postings,
                        idx.token_offsets,
                        key_id,
                    )
                )

        if arrays:
            raw = np.concatenate(arrays)

            ids, counts = np.unique(
                raw,
                return_counts=True,
            )

            score_by_id = {
                int(internal_id): int(count)
                for internal_id, count
                in zip(ids, counts)
            }
        else:
            score_by_id = {}

        # Exact + pair fallback, matching the validated P1 threshold=20.
        if len(score_by_id) <= idx.fallback_threshold:
            norm = mod._normalized_name(name)

            if norm:
                key_id = idx.exact_key_to_id.get(
                    (country, norm)
                )

                if key_id is not None:
                    exact_ids = idx._slice(
                        idx.exact_postings,
                        idx.exact_offsets,
                        key_id,
                    )

                    for internal_id in exact_ids:
                        internal_id = int(internal_id)
                        score_by_id[internal_id] = max(
                            score_by_id.get(
                                internal_id,
                                0,
                            ),
                            100,
                        )

            if len(tokens) >= 2:
                for pair in mod._query_pairs(
                    tokens,
                    idx.max_pair_tokens,
                ):
                    key_id = idx.pair_key_to_id.get(
                        (country, pair)
                    )

                    if key_id is None:
                        continue

                    pair_ids = idx._slice(
                        idx.pair_postings,
                        idx.pair_offsets,
                        key_id,
                    )

                    for internal_id in pair_ids:
                        internal_id = int(internal_id)

                        score_by_id[internal_id] = (
                            score_by_id.get(
                                internal_id,
                                0,
                            )
                            + 10
                        )

        if not score_by_id:
            return (
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint16),
            )

        ids = np.fromiter(
            score_by_id.keys(),
            dtype=np.uint32,
            count=len(score_by_id),
        )
        scores = np.fromiter(
            score_by_id.values(),
            dtype=np.uint16,
            count=len(score_by_id),
        )

        return ids, scores


class P2Source:
    def __init__(self, index_dir, min_band_hits=2):
        import blocking.minhash_lsh as mod

        self.mod = mod
        self.index_dir = index_dir

        metadata = mod.load_metadata(index_dir)

        self.bands = int(metadata["bands"])
        self.rows = int(metadata["rows"])
        self.n_gram = int(metadata["n_gram"])
        self.num_perm = int(metadata["num_perm"])
        self.seed = int(metadata["seed"])
        self.min_band_hits = int(min_band_hits)

        self.cache = mod.QueryCache(
            index_dir,
            self.bands,
        )
        self.cache.open()

    def _hashes_for_row(self, row):
        shingles = self.mod.prepare_record_shingles(
            str(row[1]),
            str(row[2]),
            self.n_gram,
        )

        if not shingles:
            return []

        mh = self.mod.create_minhash(
            shingles,
            self.num_perm,
            self.seed,
        )

        return self.mod.get_band_hashes(
            mh,
            self.bands,
            self.rows,
        )

    def query_batch(self, rows):
        if not rows:
            return []

        countries = [
            str(row[3])
            for row in rows
        ]

        hashes = [
            self._hashes_for_row(row)
            for row in rows
        ]

        return self.mod.query_internal_ids_batch(
            self.cache,
            countries,
            hashes,
            self.min_band_hits,
            0,
        )


class P3Source:
    def __init__(
        self,
        index_dir,
        top_k=500,
        nprobe=128,
        batch_size=1024,
    ):
        import blocking.embedding_faiss as mod

        self.mod = mod
        self.top_k = int(top_k)
        self.nprobe = int(nprobe)
        self.batch_size = int(batch_size)

        self.encoder = mod.load_encoder(
            model_name=mod.DEFAULT_MODEL,
            device="cuda",
            use_half=True,
        )

        _, self.country_indexes = (
            mod.load_embedding_indexes(index_dir)
        )

        for loaded in self.country_indexes.values():
            index = loaded.index

            if hasattr(index, "nprobe"):
                index.nprobe = min(
                    self.nprobe,
                    int(index.nlist),
                )

    def query_batch(self, rows):
        result = [
            (
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32),
            )
            for _ in rows
        ]

        if not rows:
            return result

        by_country = {}

        for position, row in enumerate(rows):
            by_country.setdefault(
                str(row[3]),
                [],
            ).append(position)

        for country, positions in by_country.items():
            loaded = self.country_indexes.get(country)

            if loaded is None:
                continue

            texts = []

            for position in positions:
                row = rows[position]

                name = self.mod.normalize_name(
                    str(row[1])
                )
                address = self.mod.normalize_address(
                    str(row[2])
                )

                texts.append(
                    f"{name} {address}".strip()
                )

            vectors = self.mod.encode_texts(
                self.encoder,
                texts,
                batch_size=self.batch_size,
            )

            ntotal = int(
                loaded.index.ntotal
            )

            if ntotal == 0:
                continue

            k = min(
                self.top_k,
                ntotal,
            )

            distances, positions_out = (
                loaded.index.search(
                    np.ascontiguousarray(
                        vectors,
                        dtype=np.float32,
                    ),
                    k,
                )
            )

            for local, output_position in enumerate(
                positions
            ):
                valid = positions_out[local] >= 0

                if not np.any(valid):
                    continue

                candidate_ids = np.asarray(
                    positions_out[local][valid],
                    dtype=np.uint32,
                )
                candidate_scores = np.asarray(
                    distances[local][valid],
                    dtype=np.float32,
                )

                result[output_position] = (
                    candidate_ids,
                    candidate_scores,
                )

        return result


def merge_internal_ids(
    p1_ids,
    p1_scores,
    p2_ids,
    p2_hits,
    p3_ids,
    p3_scores,
):
    """
    Merge three already-deduplicated blocker outputs using numpy only.

    Returns:
        ids, source_mask, p1_score, p2_score, p3_score
    """
    total = (
        len(p1_ids)
        + len(p2_ids)
        + len(p3_ids)
    )

    if total == 0:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.float32),
        )

    ids = np.concatenate(
        [
            p1_ids,
            p2_ids,
            p3_ids,
        ]
    ).astype(
        np.uint32,
        copy=False,
    )

    source = np.concatenate(
        [
            np.full(
                len(p1_ids),
                1,
                dtype=np.uint8,
            ),
            np.full(
                len(p2_ids),
                2,
                dtype=np.uint8,
            ),
            np.full(
                len(p3_ids),
                4,
                dtype=np.uint8,
            ),
        ]
    )

    p1 = np.concatenate(
        [
            p1_scores,
            np.zeros(
                len(p2_ids) + len(p3_ids),
                dtype=np.uint16,
            ),
        ]
    )

    p2 = np.concatenate(
        [
            np.zeros(
                len(p1_ids),
                dtype=np.uint8,
            ),
            p2_hits.astype(
                np.uint8,
                copy=False,
            ),
            np.zeros(
                len(p3_ids),
                dtype=np.uint8,
            ),
        ]
    )

    p3 = np.concatenate(
        [
            np.full(
                len(p1_ids) + len(p2_ids),
                -np.inf,
                dtype=np.float32,
            ),
            p3_scores,
        ]
    )

    order = np.argsort(
        ids,
        kind="mergesort",
    )

    ids_sorted = ids[order]
    source_sorted = source[order]
    p1_sorted = p1[order]
    p2_sorted = p2[order]
    p3_sorted = p3[order]

    boundaries = np.flatnonzero(
        ids_sorted[1:] != ids_sorted[:-1]
    ) + 1

    starts = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            boundaries,
        ]
    )

    # Each input blocker is already unique, so max is sufficient for each
    # score. Source masks need bitwise OR.
    source_out = np.bitwise_or.reduceat(
        source_sorted,
        starts,
    )

    p1_out = np.maximum.reduceat(
        p1_sorted,
        starts,
    )

    p2_out = np.maximum.reduceat(
        p2_sorted,
        starts,
    )

    # np.maximum.reduceat does not preserve -inf correctly if all entries are
    # -inf, which is fine here, but use nan-free values explicitly.
    p3_out = np.maximum.reduceat(
        p3_sorted,
        starts,
    )

    ids_out = ids_sorted[starts]

    return (
        ids_out,
        source_out,
        p1_out,
        p2_out,
        p3_out,
    )


def prune_candidates(
    ids,
    source,
    p1_score,
    p2_score,
    p3_score,
    max_candidates,
    p3_quota,
):
    if ids.size == 0:
        return ids

    source_count = (
        (source & 1 > 0).astype(np.uint8)
        + (source & 2 > 0).astype(np.uint8)
        + (source & 4 > 0).astype(np.uint8)
    )

    if ids.size <= max_candidates:
        order = np.lexsort(
            (
                ids,
                -p3_score,
                -p1_score,
                -p2_score,
                -source_count,
            )
        )
        return ids[order]

    # First protect every candidate supported by >=2 independent blockers.
    agreement_mask = source_count >= 2
    agreement_idx = np.flatnonzero(
        agreement_mask
    )

    if agreement_idx.size:
        agreement_order = np.lexsort(
            (
                ids[agreement_idx],
                -p3_score[agreement_idx],
                -p1_score[agreement_idx],
                -p2_score[agreement_idx],
                -source_count[agreement_idx],
            )
        )
        agreement_idx = agreement_idx[
            agreement_order
        ]

    if agreement_idx.size >= max_candidates:
        return ids[
            agreement_idx[:max_candidates]
        ]

    chosen = agreement_idx.tolist()
    chosen_mask = np.zeros(
        ids.size,
        dtype=bool,
    )
    chosen_mask[agreement_idx] = True

    remaining_slots = (
        max_candidates
        - len(chosen)
    )

    remaining_idx = np.flatnonzero(
        ~chosen_mask
    )

    if remaining_slots <= 0:
        return ids[
            np.asarray(chosen[:max_candidates])
        ]

    p3_only_mask = (
        source[remaining_idx] == 4
    )
    p3_only = remaining_idx[
        p3_only_mask
    ]
    non_p3_only = remaining_idx[
        ~p3_only_mask
    ]

    p3_slots = min(
        len(p3_only),
        int(round(
            remaining_slots * p3_quota
        )),
    )

    other_slots = (
        remaining_slots
        - p3_slots
    )

    if len(non_p3_only):
        other_order = np.lexsort(
            (
                ids[non_p3_only],
                -p3_score[non_p3_only],
                -p1_score[non_p3_only],
                -p2_score[non_p3_only],
                -source_count[non_p3_only],
            )
        )
        non_p3_only = non_p3_only[
            other_order
        ]

    if len(p3_only):
        p3_order = np.lexsort(
            (
                ids[p3_only],
                -p1_score[p3_only],
                -p2_score[p3_only],
                -p3_score[p3_only],
            )
        )
        p3_only = p3_only[
            p3_order
        ]

    selected = chosen + non_p3_only[
        :other_slots
    ].tolist()

    if p3_slots:
        selected += p3_only[
            :p3_slots
        ].tolist()

    # Return unused slots, if one side did not have enough candidates.
    if len(selected) < max_candidates:
        selected_mask = np.zeros(
            ids.size,
            dtype=bool,
        )
        selected_mask[
            np.asarray(selected)
        ] = True

        leftovers = np.flatnonzero(
            ~selected_mask
        )

        if leftovers.size:
            leftover_order = np.lexsort(
                (
                    ids[leftovers],
                    -p3_score[leftovers],
                    -p1_score[leftovers],
                    -p2_score[leftovers],
                    -source_count[leftovers],
                )
            )

            selected += leftovers[
                leftover_order[
                    : max_candidates
                    - len(selected)
                ]
            ].tolist()

    return ids[
        np.asarray(
            selected[:max_candidates],
            dtype=np.int64,
        )
    ]


def load_truth(path, wanted):
    truth = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as fh:
        header = (
            fh.readline()
            .rstrip("\r\n")
            .split("\t")
        )

        s1_idx = header.index(
            "source1_entity_id"
        )
        match_idx = header.index(
            "matched_entity_ids"
        )

        for line in fh:
            parts = (
                line
                .rstrip("\r\n")
                .split("\t")
            )

            if len(parts) <= max(
                s1_idx,
                match_idx,
            ):
                continue

            s1_id = parts[s1_idx]

            if s1_id not in wanted:
                continue

            raw = parts[match_idx]

            truth[s1_id] = (
                set(raw.split(","))
                if raw
                else set()
            )

    return truth


def run(args):
    print("=" * 70)
    print("FAST STREAMING 3-BLOCKER PIPELINE")
    print("=" * 70)
    print(
        f"Candidate cap : {args.max_candidates:,}"
    )
    print(
        f"P3 quota      : {args.p3_quota:.2f}"
    )
    print(
        f"Chunk size    : {args.chunk_size:,}"
    )

    print("Loading P1...", flush=True)
    p1 = P1Source(args.p1_index)

    print("Loading P2...", flush=True)
    p2 = P2Source(
        args.p2_index,
        args.p2_min_band_hits,
    )

    print("Loading P3...", flush=True)
    p3 = P3Source(
        args.p3_index,
        args.p3_top_k,
        args.p3_nprobe,
        args.p3_batch_size,
    )

    truth = None

    if args.ground_truth:
        wanted = set()

        with open(
            args.s1,
            "r",
            encoding="utf-8",
            newline="",
        ) as fh:
            header = (
                fh.readline()
                .rstrip("\r\n")
                .split("\t")
            )

            id_idx = header.index(
                "entity_id"
            )

            for line in fh:
                parts = (
                    line
                    .rstrip("\r\n")
                    .split("\t")
                )

                if (
                    len(parts) > id_idx
                    and parts[id_idx]
                ):
                    wanted.add(
                        parts[id_idx]
                    )

        truth = load_truth(
            args.ground_truth,
            wanted,
        )

        print(
            f"Ground-truth rows retained: "
            f"{len(truth):,}",
            flush=True,
        )

    output = None

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = open(
            output_path,
            "w",
            encoding="utf-8",
            buffering=1024 * 1024,
        )

        output.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

    total = 0
    total_candidates = 0
    zero = 0

    true_pairs = 0
    found_pairs = 0
    matched_entities = 0
    fully_recalled = 0

    start = time.time()

    try:
        for chunk in pd.read_csv(
            args.s1,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=S1_COLUMNS,
            chunksize=args.chunk_size,
        ):
            rows = list(
                chunk.itertuples(
                    index=False,
                    name=None,
                )
            )

            p1_results = [
                p1.query(
                    str(row[1]),
                    str(row[3]),
                )
                for row in rows
            ]

            p2_results = p2.query_batch(
                rows
            )
            p3_results = p3.query_batch(
                rows
            )

            for row, p1_result, p2_result, p3_result in zip(
                rows,
                p1_results,
                p2_results,
                p3_results,
            ):
                p1_ids, p1_scores = p1_result
                p2_ids = np.asarray(
                    p2_result,
                    dtype=np.uint32,
                )
                p2_hits = np.full(
                    len(p2_ids),
                    2,
                    dtype=np.uint8,
                )
                p3_ids, p3_scores = p3_result

                (
                    ids,
                    source,
                    p1_score,
                    p2_score,
                    p3_score,
                ) = merge_internal_ids(
                    p1_ids,
                    p1_scores,
                    p2_ids,
                    p2_hits,
                    p3_ids,
                    p3_scores,
                )

                candidates = prune_candidates(
                    ids,
                    source,
                    p1_score,
                    p2_score,
                    p3_score,
                    args.max_candidates,
                    args.p3_quota,
                )

                total += 1
                total_candidates += len(
                    candidates
                )

                if not len(candidates):
                    zero += 1

                if output is not None:
                    raw_ids = (
                        p1.index.entity_ids[
                            candidates
                        ]
                    )

                    output.write(
                        str(row[0])
                        + "\t"
                        + ",".join(
                            value.decode("utf-8")
                            for value in raw_ids.tolist()
                        )
                        + "\n"
                    )

                if truth is not None:
                    true = truth.get(
                        str(row[0]),
                        set(),
                    )

                    if true:
                        matched_entities += 1
                        true_pairs += len(true)

                        raw_ids = (
                            p1.index.entity_ids[
                                candidates
                            ]
                        )

                        found = sum(
                            1
                            for value in raw_ids.tolist()
                            if value.decode(
                                "utf-8"
                            ) in true
                        )

                        found_pairs += found

                        if found == len(true):
                            fully_recalled += 1

                if total % 1000 == 0:
                    elapsed = (
                        time.time() - start
                    )
                    print(
                        f"  queried={total:,}"
                        f" | avg={total_candidates / total:.2f}"
                        f" | rate={total / elapsed:.1f}/s"
                        f" | zero={zero:,}",
                        flush=True,
                    )
    finally:
        if output is not None:
            output.close()

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("FAST STREAMING 3-BLOCKER COMPLETE")
    print("=" * 70)
    print(
        f"S1 queries        : {total:,}"
    )
    print(
        f"Average candidates: "
        f"{total_candidates / total:.2f}"
        if total
        else "Average candidates: 0"
    )
    print(
        f"Candidate cap     : "
        f"{args.max_candidates:,}"
    )
    print(
        f"Zero candidates   : {zero:,}"
    )
    print(
        f"Query time        : {elapsed:.1f}s"
    )
    print(
        f"Query rate        : "
        f"{total / elapsed:.2f} S1/s"
        if elapsed
        else "Query rate        : 0 S1/s"
    )

    if truth is not None:
        pairwise = (
            found_pairs / true_pairs
            if true_pairs
            else 0.0
        )

        full = (
            fully_recalled
            / matched_entities
            if matched_entities
            else 0.0
        )

        print()
        print("DEV RECALL")
        print(
            f"True pairs       : {true_pairs:,}"
        )
        print(
            f"Found pairs      : {found_pairs:,}"
        )
        print(
            f"Pairwise recall  : {pairwise:.6f}"
        )
        print(
            f"Full recall      : {full:.6f}"
        )

    if output is not None:
        print(
            f"Output            : {args.output}"
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--s1",
        required=True,
    )
    parser.add_argument(
        "--p1-index",
        required=True,
    )
    parser.add_argument(
        "--p2-index",
        required=True,
    )
    parser.add_argument(
        "--p3-index",
        required=True,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--p3-quota",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--p2-min-band-hits",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--p3-top-k",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--p3-nprobe",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--p3-batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
    )

    args = parser.parse_args()

    if args.max_candidates < 1:
        parser.error(
            "--max-candidates must be >= 1"
        )

    if not 0.0 <= args.p3_quota <= 1.0:
        parser.error(
            "--p3-quota must be between 0 and 1"
        )

    if args.p2_min_band_hits < 1:
        parser.error(
            "--p2-min-band-hits must be >= 1"
        )

    if args.p3_top_k < 1:
        parser.error(
            "--p3-top-k must be >= 1"
        )

    if args.p3_nprobe < 1:
        parser.error(
            "--p3-nprobe must be >= 1"
        )

    if args.p3_batch_size < 1:
        parser.error(
            "--p3-batch-size must be >= 1"
        )

    if args.chunk_size < 1:
        parser.error(
            "--chunk-size must be >= 1"
        )

    run(args)


if __name__ == "__main__":
    main()
