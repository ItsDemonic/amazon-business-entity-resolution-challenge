"""
Correct, fast, memory-bounded P1 + P2 + P3 candidate union with pruning.

Important implementation detail:
    P1/P2 use one global S2->S3 internal-ID space.
    P3 uses country-local FAISS positions.
    This module creates a one-time country-local-P3 -> global-ID mapping,
    so all three blockers are merged in the SAME uint32 ID space.

Pruning:
    1. If P1+P2 core fits in the cap, keep the complete P1+P2 core.
       Use remaining slots for the strongest P3-only candidates.
    2. If P1+P2 core exceeds the cap, prune the core using agreement-first
       Reciprocal Rank Fusion (RRF). P3 is used only when there is spare room.
    This is intentionally conservative: P3 is complementary, but P1+P2 is
    the stronger measured base.

Output:
    Optional compact CSR-like binary artifact:
        <prefix>.ids.bin       uint32 global candidate IDs
        <prefix>.offsets.npy   uint64 offsets, one per S1 row
        <prefix>.s1_ids.npy    fixed-width S1 IDs
        <prefix>.meta.json     metadata

This avoids the enormous text TSV representation.

Example DEV:
    python -m blocking.streaming_blocker_pipeline_opt \
      --s1 common/dev_s1.tsv \
      --p1-index blocking/.candidates_token_opt20_index \
      --p2-index blocking/minhash_index \
      --p3-index blocking/embedding_index \
      --max-candidates 1500 \
      --p2-min-band-hits 2 \
      --p3-top-k 500 \
      --p3-nprobe 128 \
      --p3-batch-size 1024 \
      --chunk-size 1024 \
      --p3-slack-quota 0.10 \
      --ground-truth dataset/train/train_ground_truth.tsv
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


S1_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EMPTY_U32 = np.empty(0, dtype=np.uint32)
EMPTY_U16 = np.empty(0, dtype=np.uint16)
EMPTY_F32 = np.empty(0, dtype=np.float32)


def country_digest(country):
    return hashlib.sha1(
        str(country).encode("utf-8")
    ).hexdigest()[:16]


def first_pass_s1_metadata(path):
    count = 0
    max_width = 1

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

        idx = header.index("entity_id")

        for line in fh:
            parts = (
                line
                .rstrip("\r\n")
                .split("\t")
            )

            if len(parts) <= idx:
                continue

            value = parts[idx]
            max_width = max(
                max_width,
                len(value.encode("utf-8")),
            )
            count += 1

    return count, max_width


class BinaryCandidateWriter:
    def __init__(self, prefix, n_rows, id_width):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.s1_ids = np.lib.format.open_memmap(
            str(self.prefix) + ".s1_ids.npy",
            mode="w+",
            dtype=f"S{id_width}",
            shape=(n_rows,),
        )

        self.offsets = np.lib.format.open_memmap(
            str(self.prefix) + ".offsets.npy",
            mode="w+",
            dtype=np.uint64,
            shape=(n_rows + 1,),
        )
        self.offsets[0] = 0

        self.candidate_fh = open(
            str(self.prefix) + ".ids.bin",
            "wb",
            buffering=1024 * 1024,
        )

        self.row = 0
        self.total_candidates = 0

    def write(self, s1_id, candidate_ids):
        candidate_ids = np.asarray(
            candidate_ids,
            dtype=np.uint32,
        )

        self.s1_ids[self.row] = str(s1_id).encode(
            "utf-8"
        )

        if candidate_ids.size:
            candidate_ids.tofile(
                self.candidate_fh
            )
            self.total_candidates += int(
                candidate_ids.size
            )

        self.row += 1
        self.offsets[self.row] = (
            self.total_candidates
        )

    def close(self, metadata):
        self.candidate_fh.flush()
        self.candidate_fh.close()

        self.s1_ids.flush()
        self.offsets.flush()

        with open(
            str(self.prefix) + ".meta.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                metadata,
                fh,
                indent=2,
            )

        del self.s1_ids
        del self.offsets


class P1Source:
    def __init__(self, index_dir):
        import blocking.token_index as mod

        self.mod = mod
        self.index = mod.HybridTokenIndex(index_dir)

    def query_scored(self, name, country):
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

            scores = counts.astype(
                np.uint16,
                copy=False,
            )
        else:
            ids = EMPTY_U32
            scores = EMPTY_U16

        score_by_id = None

        # Exact/pair fallback is intentionally identical to the validated
        # threshold-20 P1 logic.
        if ids.size <= idx.fallback_threshold:
            if ids.size:
                score_by_id = {
                    int(i): int(s)
                    for i, s in zip(
                        ids,
                        scores,
                    )
                }
            else:
                score_by_id = {}

            norm = mod._normalized_name(name)

            if norm:
                key_id = idx.exact_key_to_id.get(
                    (country, norm)
                )
                if key_id is not None:
                    exact = idx._slice(
                        idx.exact_postings,
                        idx.exact_offsets,
                        key_id,
                    )

                    for internal_id in exact:
                        internal_id = int(
                            internal_id
                        )
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
                        internal_id = int(
                            internal_id
                        )
                        score_by_id[internal_id] = (
                            score_by_id.get(
                                internal_id,
                                0,
                            )
                            + 10
                        )

            if not score_by_id:
                return EMPTY_U32, EMPTY_U16

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

        return (
            np.asarray(ids, dtype=np.uint32),
            np.asarray(scores, dtype=np.uint16),
        )


class P2Source:
    def __init__(
        self,
        index_dir,
        min_band_hits=2,
    ):
        import blocking.minhash_lsh as mod

        self.mod = mod
        self.index_dir = Path(index_dir)

        metadata = mod.load_metadata(index_dir)

        self.bands = int(metadata["bands"])
        self.rows = int(metadata["rows"])
        self.n_gram = int(metadata["n_gram"])
        self.num_perm = int(metadata["num_perm"])
        self.seed = int(metadata["seed"])
        self.min_band_hits = int(
            min_band_hits
        )

        self.cache = mod.QueryCache(
            index_dir,
            self.bands,
        )
        self.cache.open()

    def _band_hashes_for_row(self, row):
        shingles = (
            self.mod.prepare_record_shingles(
                str(row[1]),
                str(row[2]),
                self.n_gram,
            )
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
        n = len(rows)

        results = [
            (
                EMPTY_U32,
                EMPTY_U16,
            )
            for _ in range(n)
        ]

        if not n:
            return results

        countries = [
            str(row[3])
            for row in rows
        ]

        hashes_batch = [
            self._band_hashes_for_row(row)
            for row in rows
        ]

        active = [
            i
            for i, hashes in enumerate(
                hashes_batch
            )
            if hashes
        ]

        if not active:
            return results

        per_query_parts = [
            []
            for _ in range(n)
        ]

        for band_number in range(
            self.bands
        ):
            query_indices = []
            buckets = np.empty(
                len(active),
                dtype=np.uint64,
            )
            bucket_count = 0

            for query_index in active:
                hashes = hashes_batch[
                    query_index
                ]

                if band_number >= len(hashes):
                    continue

                buckets[
                    bucket_count
                ] = self.mod.bucket_for_band(
                    countries[query_index],
                    hashes[band_number],
                    band_number,
                )

                query_indices.append(
                    query_index
                )
                bucket_count += 1

            if not bucket_count:
                continue

            matches = self.cache.lookup_many(
                band_number,
                buckets[:bucket_count],
            )

            for local_index, values in matches:
                if values.size:
                    per_query_parts[
                        query_indices[
                            local_index
                        ]
                    ].append(values)

        for query_index, parts in enumerate(
            per_query_parts
        ):
            if not parts:
                continue

            raw = np.concatenate(parts)

            ids, counts = np.unique(
                raw,
                return_counts=True,
            )

            keep = counts >= (
                self.min_band_hits
            )

            ids = ids[keep]
            counts = counts[keep]

            results[query_index] = (
                np.asarray(
                    ids,
                    dtype=np.uint32,
                ),
                np.asarray(
                    counts,
                    dtype=np.uint16,
                ),
            )

        return results


class P3Source:
    def __init__(
        self,
        index_dir,
        p1_entity_ids,
        top_k=500,
        nprobe=128,
        batch_size=1024,
        map_chunk=250_000,
    ):
        import blocking.embedding_faiss as mod

        self.mod = mod
        self.index_dir = Path(index_dir)
        self.top_k = int(top_k)
        self.nprobe = int(nprobe)
        self.batch_size = int(batch_size)

        self.encoder = mod.load_encoder(
            model_name=mod.DEFAULT_MODEL,
            device="cuda",
            use_half=True,
        )

        _, self.country_indexes = (
            mod.load_embedding_indexes(
                index_dir
            )
        )

        self.global_maps = {}
        self.map_chunk = int(map_chunk)

        self._ensure_global_maps(
            p1_entity_ids
        )

        for country_index in (
            self.country_indexes.values()
        ):
            index = country_index.index

            if hasattr(index, "nprobe"):
                index.nprobe = min(
                    self.nprobe,
                    int(index.nlist),
                )

    def _ensure_global_maps(
        self,
        p1_entity_ids,
    ):
        map_dir = (
            self.index_dir
            / "global_id_maps"
        )
        map_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        required = {}

        for country, loaded in (
            self.country_indexes.items()
        ):
            path = (
                map_dir
                / f"map_{country_digest(country)}.npy"
            )
            required[country] = path

        if all(path.exists() for path in required.values()):
            self.global_maps = {
                country: np.load(
                    str(path),
                    mmap_mode="r",
                )
                for country, path in required.items()
            }
            return

        print(
            "Building one-time P3 local->global ID maps...",
            flush=True,
        )

        start = time.time()

        # Sort the 10.32M public IDs once in RAM. The sorted arrays are
        # temporary; only the compact per-country uint32 maps are kept.
        entity_ids = p1_entity_ids

        order = np.argsort(
            entity_ids,
            kind="stable",
        )

        sorted_ids = np.asarray(
            entity_ids[order]
        )

        for country, loaded in (
            self.country_indexes.items()
        ):
            path = required[country]

            count = int(
                loaded.index.ntotal
            )

            output = np.lib.format.open_memmap(
                str(path),
                mode="w+",
                dtype=np.uint32,
                shape=(count,),
            )

            try:
                for start_pos in range(
                    0,
                    count,
                    self.map_chunk,
                ):
                    end_pos = min(
                        start_pos
                        + self.map_chunk,
                        count,
                    )

                    local_ids = np.asarray(
                        loaded.id_store[
                            start_pos:end_pos
                        ]
                    )

                    positions = np.searchsorted(
                        sorted_ids,
                        local_ids,
                    )

                    if np.any(
                        positions >= len(sorted_ids)
                    ):
                        raise RuntimeError(
                            "P3 ID mapping failed: "
                            "position outside global ID array."
                        )

                    matched = (
                        sorted_ids[positions]
                        == local_ids
                    )

                    if not np.all(matched):
                        bad = np.flatnonzero(
                            ~matched
                        )[:5]

                        raise RuntimeError(
                            "P3 ID mapping failed: "
                            "entity ID not found in P1 global ID space. "
                            f"country={country!r}, "
                            f"local_positions={bad.tolist()}"
                        )

                    output[
                        start_pos:end_pos
                    ] = np.asarray(
                        order[positions],
                        dtype=np.uint32,
                    )

                output.flush()

            finally:
                del output

            self.global_maps[country] = (
                np.load(
                    str(path),
                    mmap_mode="r",
                )
            )

        del sorted_ids
        del order

        print(
            f"P3 global-ID maps ready in "
            f"{time.time() - start:.1f}s.",
            flush=True,
        )

    def query_batch(self, rows):
        result = [
            (
                EMPTY_U32,
                EMPTY_F32,
            )
            for _ in rows
        ]

        if not rows:
            return result

        by_country = {}

        for position, row in enumerate(
            rows
        ):
            by_country.setdefault(
                str(row[3]),
                [],
            ).append(position)

        for country, row_positions in (
            by_country.items()
        ):
            loaded = self.country_indexes.get(
                country
            )

            if loaded is None:
                continue

            texts = []

            for position in row_positions:
                row = rows[position]

                name = (
                    self.mod.normalize_name(
                        str(row[1])
                    )
                )
                address = (
                    self.mod.normalize_address(
                        str(row[2])
                    )
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

            distances, local_positions = (
                loaded.index.search(
                    np.ascontiguousarray(
                        vectors,
                        dtype=np.float32,
                    ),
                    k,
                )
            )

            global_map = (
                self.global_maps[country]
            )

            for local_row, output_position in (
                enumerate(row_positions)
            ):
                valid = (
                    local_positions[
                        local_row
                    ] >= 0
                )

                if not np.any(valid):
                    continue

                local_ids = np.asarray(
                    local_positions[
                        local_row
                    ][valid],
                    dtype=np.int64,
                )

                global_ids = np.asarray(
                    global_map[local_ids],
                    dtype=np.uint32,
                )

                scores = np.asarray(
                    distances[
                        local_row
                    ][valid],
                    dtype=np.float32,
                )

                result[output_position] = (
                    global_ids,
                    scores,
                )

        return result


def rank_source(ids, scores, descending=True):
    if len(ids) == 0:
        return EMPTY_U32

    if descending:
        order = np.argsort(
            -np.asarray(scores),
            kind="stable",
        )
    else:
        order = np.argsort(
            np.asarray(scores),
            kind="stable",
        )

    ranks = np.empty(
        len(ids),
        dtype=np.uint32,
    )
    ranks[order] = (
        np.arange(
            len(ids),
            dtype=np.uint32,
        )
        + 1
    )

    return ranks


def merge_sources(
    p1_ids,
    p1_scores,
    p2_ids,
    p2_scores,
    p3_ids,
    p3_scores,
):
    n1 = len(p1_ids)
    n2 = len(p2_ids)
    n3 = len(p3_ids)

    if n1 + n2 + n3 == 0:
        return (
            EMPTY_U32,
            np.empty(
                0,
                dtype=np.uint8,
            ),
            np.empty(
                0,
                dtype=np.uint32,
            ),
            np.empty(
                0,
                dtype=np.uint32,
            ),
            np.empty(
                0,
                dtype=np.uint32,
            ),
            EMPTY_F32,
        )

    p1_rank = rank_source(
        p1_ids,
        p1_scores,
        descending=True,
    )
    p2_rank = rank_source(
        p2_ids,
        p2_scores,
        descending=True,
    )
    p3_rank = rank_source(
        p3_ids,
        p3_scores,
        descending=True,
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
            np.ones(
                n1,
                dtype=np.uint8,
            ),
            np.full(
                n2,
                2,
                dtype=np.uint8,
            ),
            np.full(
                n3,
                4,
                dtype=np.uint8,
            ),
        ]
    )

    r1 = np.concatenate(
        [
            p1_rank,
            np.zeros(
                n2 + n3,
                dtype=np.uint32,
            ),
        ]
    )

    r2 = np.concatenate(
        [
            np.zeros(
                n1,
                dtype=np.uint32,
            ),
            p2_rank,
            np.zeros(
                n3,
                dtype=np.uint32,
            ),
        ]
    )

    r3 = np.concatenate(
        [
            np.zeros(
                n1 + n2,
                dtype=np.uint32,
            ),
            p3_rank,
        ]
    )

    p3_score = np.concatenate(
        [
            np.full(
                n1 + n2,
                -np.inf,
                dtype=np.float32,
            ),
            p3_scores,
        ]
    )

    p1_score = np.concatenate(
        [
            np.asarray(
                p1_scores,
                dtype=np.uint32,
            ),
            np.zeros(
                n2 + n3,
                dtype=np.uint32,
            ),
        ]
    )

    p2_score = np.concatenate(
        [
            np.zeros(
                n1,
                dtype=np.uint32,
            ),
            np.asarray(
                p2_scores,
                dtype=np.uint32,
            ),
            np.zeros(
                n3,
                dtype=np.uint32,
            ),
        ]
    )

    order = np.argsort(
        ids,
        kind="mergesort",
    )

    ids = ids[order]
    source = source[order]
    r1 = r1[order]
    r2 = r2[order]
    r3 = r3[order]
    p1_score = p1_score[order]
    p2_score = p2_score[order]
    p3_score = p3_score[order]

    boundaries = (
        np.flatnonzero(
            ids[1:] != ids[:-1]
        )
        + 1
    )

    starts = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            boundaries,
        ]
    )

    unique_ids = ids[starts]

    source_mask = np.bitwise_or.reduceat(
        source,
        starts,
    )

    # Per-source ranks/scores are defined for one occurrence per source.
    big_rank = np.uint32(2**32 - 1)

    r1_safe = np.where(
        r1 == 0,
        big_rank,
        r1,
    )
    r2_safe = np.where(
        r2 == 0,
        big_rank,
        r2,
    )
    r3_safe = np.where(
        r3 == 0,
        big_rank,
        r3,
    )

    r1_out = np.minimum.reduceat(
        r1_safe,
        starts,
    )
    r2_out = np.minimum.reduceat(
        r2_safe,
        starts,
    )
    r3_out = np.minimum.reduceat(
        r3_safe,
        starts,
    )

    # Recover 0 for absent sources.
    r1_out = np.where(
        r1_out == big_rank,
        0,
        r1_out,
    ).astype(np.uint32)

    r2_out = np.where(
        r2_out == big_rank,
        0,
        r2_out,
    ).astype(np.uint32)

    r3_out = np.where(
        r3_out == big_rank,
        0,
        r3_out,
    ).astype(np.uint32)

    p1_out = np.maximum.reduceat(
        p1_score,
        starts,
    )

    p2_out = np.maximum.reduceat(
        p2_score,
        starts,
    )

    p3_out = np.maximum.reduceat(
        p3_score,
        starts,
    )

    return (
        unique_ids,
        source_mask,
        r1_out,
        r2_out,
        r3_out,
        p3_out,
    )


def rrf_score(
    source_mask,
    r1,
    r2,
    r3,
    p3_weight=1.0,
    rrf_k=60.0,
):
    score = np.zeros(
        len(source_mask),
        dtype=np.float32,
    )

    has1 = r1 > 0
    has2 = r2 > 0
    has3 = r3 > 0

    score[has1] += (
        1.0
        / (rrf_k + r1[has1])
    )

    score[has2] += (
        1.0
        / (rrf_k + r2[has2])
    )

    score[has3] += (
        p3_weight
        / (rrf_k + r3[has3])
    )

    source_count = (
        has1.astype(np.uint8)
        + has2.astype(np.uint8)
        + has3.astype(np.uint8)
    )

    # Small explicit agreement bonus. The reciprocal-rank terms already
    # strongly reward multi-source candidates; this simply makes ties safer.
    score += (
        0.005
        * source_count.astype(
            np.float32
        )
    )

    return score


def prune_candidates(
    ids,
    source_mask,
    r1,
    r2,
    r3,
    p3_score,
    max_candidates,
    p3_slack_quota=0.10,
    p3_weight=1.0,
):
    if ids.size == 0:
        return ids

    if ids.size <= max_candidates:
        return ids

    has_p1 = (source_mask & 1) != 0
    has_p2 = (source_mask & 2) != 0
    has_p3 = (source_mask & 4) != 0

    # Core = P1 ∪ P2. P3 is complementary enrichment.
    core = has_p1 | has_p2
    core_idx = np.flatnonzero(core)
    p3_only_idx = np.flatnonzero(
        ~core & has_p3
    )

    # Conservative strategy:
    # - Never eject P3 into the core when the core itself fits.
    # - When the core does not fit, rank the core with RRF and prune it.
    if core_idx.size <= max_candidates:
        chosen = core_idx.tolist()
        remaining = (
            max_candidates - len(chosen)
        )

        if remaining > 0 and p3_only_idx.size:
            p3_only_scores = p3_score[
                p3_only_idx
            ]

            p3_rank_order = np.argsort(
                -p3_only_scores,
                kind="stable",
            )

            # Use only a controlled slack quota for P3-only candidates.
            p3_slots = min(
                remaining,
                max(
                    1,
                    int(
                        round(
                            max_candidates
                            * p3_slack_quota
                        )
                    ),
                ),
            )

            chosen.extend(
                p3_only_idx[
                    p3_rank_order[
                        :p3_slots
                    ]
                ].tolist()
            )

        if len(chosen) < max_candidates:
            all_idx = np.arange(
                ids.size,
                dtype=np.int64,
            )

            chosen_set = np.zeros(
                ids.size,
                dtype=bool,
            )
            chosen_set[
                np.asarray(
                    chosen,
                    dtype=np.int64,
                )
            ] = True

            leftovers = all_idx[
                ~chosen_set
            ]

            if leftovers.size:
                scores = rrf_score(
                    source_mask[
                        leftovers
                    ],
                    r1[leftovers],
                    r2[leftovers],
                    r3[leftovers],
                    p3_weight=p3_weight,
                )

                order = np.argsort(
                    -scores,
                    kind="stable",
                )

                chosen.extend(
                    leftovers[
                        order[
                            : (
                                max_candidates
                                - len(chosen)
                            )
                        ]
                    ].tolist()
                )

        return ids[
            np.asarray(
                chosen[
                    :max_candidates
                ],
                dtype=np.int64,
            )
        ]

    # Core itself exceeds the cap: no P3-only candidate is allowed to displace
    # a core candidate. This protects the already validated P1+P2 recall base.
    core_scores = rrf_score(
        source_mask[core_idx],
        r1[core_idx],
        r2[core_idx],
        r3[core_idx],
        p3_weight=p3_weight,
    )

    order = np.argsort(
        -core_scores,
        kind="stable",
    )

    return ids[
        core_idx[
            order[:max_candidates]
        ]
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
        default=1500,
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
        "--p3-slack-quota",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--p3-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Writes compact binary candidate files.",
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

    if not 0.0 <= args.p3_slack_quota <= 1.0:
        parser.error(
            "--p3-slack-quota must be between 0 and 1"
        )

    s1_count = None
    id_width = None
    writer = None

    if args.output_prefix:
        s1_count, id_width = (
            first_pass_s1_metadata(
                args.s1
            )
        )

    print("=" * 70)
    print("CORRECT FAST 3-BLOCKER + PRUNING")
    print("=" * 70)
    print(
        f"Candidate cap : "
        f"{args.max_candidates:,}"
    )
    print(
        f"P3 slack quota: "
        f"{args.p3_slack_quota:.2f}"
    )
    print(
        "P3 IDs        : mapped to P1/P2 global ID space"
    )

    print("Loading P1...", flush=True)
    p1 = P1Source(
        args.p1_index
    )

    print("Loading P2...", flush=True)
    p2 = P2Source(
        args.p2_index,
        args.p2_min_band_hits,
    )

    print("Loading P3...", flush=True)
    p3 = P3Source(
        args.p3_index,
        p1.index.entity_ids,
        top_k=args.p3_top_k,
        nprobe=args.p3_nprobe,
        batch_size=args.p3_batch_size,
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

    if args.output_prefix:
        writer = BinaryCandidateWriter(
            args.output_prefix,
            s1_count,
            id_width,
        )

    total = 0
    total_candidates = 0
    total_zero = 0
    max_seen = 0

    true_pairs = 0
    found_pairs = 0
    matched_entities = 0
    fully_recalled = 0

    start = time.time()

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
            p1.query_scored(
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

        for (
            row,
            p1_result,
            p2_result,
            p3_result,
        ) in zip(
            rows,
            p1_results,
            p2_results,
            p3_results,
        ):
            p1_ids, p1_scores = (
                p1_result
            )
            p2_ids, p2_scores = (
                p2_result
            )
            p3_ids, p3_scores = (
                p3_result
            )

            (
                merged_ids,
                source_mask,
                r1,
                r2,
                r3,
                merged_p3_scores,
            ) = merge_sources(
                p1_ids,
                p1_scores,
                p2_ids,
                p2_scores,
                p3_ids,
                p3_scores,
            )

            candidates = prune_candidates(
                merged_ids,
                source_mask,
                r1,
                r2,
                r3,
                merged_p3_scores,
                args.max_candidates,
                p3_slack_quota=args.p3_slack_quota,
                p3_weight=args.p3_weight,
            )

            total += 1
            count = len(candidates)

            total_candidates += count
            max_seen = max(
                max_seen,
                count,
            )

            if count == 0:
                total_zero += 1

            if writer is not None:
                writer.write(
                    str(row[0]),
                    candidates,
                )

            if truth is not None:
                true = truth.get(
                    str(row[0]),
                    set(),
                )

                if true:
                    matched_entities += 1
                    true_pairs += len(true)

                    raw_ids = p1.index.entity_ids[
                        candidates
                    ]

                    found = 0
                    for raw_id in raw_ids.tolist():
                        if (
                            raw_id.decode(
                                "utf-8"
                            )
                            in true
                        ):
                            found += 1

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
                    f" | zero={total_zero:,}",
                    flush=True,
                )

    elapsed = time.time() - start

    if writer is not None:
        writer.close(
            {
                "format": "uint32_csr",
                "candidate_id_space": (
                    "P1/P2 global S2+S3 row order"
                ),
                "s1_rows": total,
                "candidate_rows": total_candidates,
                "max_candidates": (
                    args.max_candidates
                ),
            }
        )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(
        f"S1 queries        : {total:,}"
    )
    print(
        f"Average candidates: "
        f"{total_candidates / total:.2f}"
    )
    print(
        f"Candidate cap     : "
        f"{args.max_candidates:,}"
    )
    print(
        f"Max seen          : {max_seen:,}"
    )
    print(
        f"Zero candidates   : {total_zero:,}"
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
            fully_recalled / matched_entities
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

    if args.output_prefix:
        print(
            f"Binary prefix    : "
            f"{args.output_prefix}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
