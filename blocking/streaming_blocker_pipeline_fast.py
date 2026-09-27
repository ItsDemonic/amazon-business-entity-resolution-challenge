"""
Fast streaming P1 + P2 + P3 candidate pipeline.

IMPORTANT:
    This module intentionally does NOT create a full raw candidate-union TSV.
    It keeps only one bounded S1 chunk in memory, merges the three blockers
    per S1 row, prunes to a configurable cap, and writes/yields only the
    pruned candidates.

Validated blockers:
    P1 = optimized hybrid token index, fallback_threshold=20
    P2 = batched MinHash LSH, min_band_hits=2
    P3 = persistent FAISS IVF-PQ, top-k=500, nprobe=128

The default pruning strategy is agreement-first, then a controlled quota for
P1/P2-only versus P3-only candidates. This avoids the catastrophic recall loss
seen with "sort everything by votes then take the first 500".
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(slots=True)
class CandidateInfo:
    sources: int = 0       # 1=P1, 2=P2, 4=P3
    p1_score: int = 0      # token hits; exact/pair fallbacks boost this
    p2_score: int = 0      # MinHash band hits
    p3_score: float = -math.inf


class P1Fast:
    def __init__(self, index_dir):
        import blocking.token_index as mod

        self.mod = mod
        self.index = mod.HybridTokenIndex(index_dir)

    def query(self, name, country):
        mod = self.mod
        idx = self.index
        tokens = mod._token_set(name)

        arrays = []
        for token in tokens:
            key_id = idx.token_key_to_id.get((country, token))
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
            result = {
                int(internal_id): int(count)
                for internal_id, count in zip(ids, counts)
            }
        else:
            result = {}

        # Mirror the validated threshold-20 P1 fallback exactly.
        if len(result) <= idx.fallback_threshold:
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
                        internal_id = int(internal_id)
                        result[internal_id] = max(
                            result.get(internal_id, 0),
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
                        result[internal_id] = (
                            result.get(internal_id, 0) + 10
                        )

        return list(result.items())


class P2Fast:
    def __init__(self, index_dir, min_band_hits=2):
        import blocking.minhash_lsh as mod

        self.mod = mod
        self.index_dir = Path(index_dir)

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

        ids_path = self.index_dir / "entity_ids.npy"
        if not ids_path.exists():
            ids_path = mod.convert_entity_ids_to_npy(index_dir)

        self.entity_ids = np.load(
            ids_path,
            mmap_mode="r",
        )

        n_entities = int(self.entity_ids.shape[0])

        # Reused across all queries. 20 MB total for 10.3M entities.
        self.marks = np.zeros(
            n_entities,
            dtype=np.uint32,
        )
        self.hit_counts = np.zeros(
            n_entities,
            dtype=np.uint8,
        )
        self.marker = np.uint32(0)

    def _next_marker(self):
        self.marker = np.uint32(
            int(self.marker) + 1
        )
        if self.marker == 0:
            self.marks.fill(0)
            self.marker = np.uint32(1)
        return self.marker

    def _band_hashes(self, name, address):
        mod = self.mod

        shingles = mod.prepare_record_shingles(
            name,
            address,
            self.n_gram,
        )
        if not shingles:
            return []

        mh = mod.create_minhash(
            shingles,
            self.num_perm,
            self.seed,
        )

        return mod.get_band_hashes(
            mh,
            self.bands,
            self.rows,
        )

    def query_batch(self, rows):
        n = len(rows)
        output = [[] for _ in range(n)]

        if not n:
            return output

        countries = []
        hashes_batch = []

        for row in rows:
            countries.append(str(row[3]))
            hashes_batch.append(
                self._band_hashes(
                    str(row[1]),
                    str(row[2]),
                )
            )

        active = [
            i
            for i, hashes in enumerate(hashes_batch)
            if hashes
        ]

        if not active:
            return output

        per_query_parts = [[] for _ in range(n)]

        for band in range(self.bands):
            query_indices = []
            buckets = np.empty(
                len(active),
                dtype=np.uint64,
            )
            bucket_count = 0

            for query_index in active:
                hashes = hashes_batch[query_index]

                if band >= len(hashes):
                    continue

                buckets[bucket_count] = (
                    self.mod.bucket_for_band(
                        countries[query_index],
                        hashes[band],
                        band,
                    )
                )
                query_indices.append(query_index)
                bucket_count += 1

            if not bucket_count:
                continue

            matches = self.cache.lookup_many(
                band,
                buckets[:bucket_count],
            )

            for local_index, values in matches:
                if values.size:
                    per_query_parts[
                        query_indices[local_index]
                    ].append(values)

        for query_index, parts in enumerate(per_query_parts):
            if not parts:
                continue

            marker = self._next_marker()
            touched = []

            for arr in parts:
                unseen = self.marks[arr] != marker

                if np.any(unseen):
                    new_ids = np.asarray(
                        arr[unseen],
                        dtype=np.uint64,
                    )
                    self.marks[new_ids] = marker
                    touched.append(new_ids)

                self.hit_counts[arr] += 1

            if not touched:
                continue

            candidates = np.concatenate(touched)
            hits = self.hit_counts[candidates]
            keep = hits >= self.min_band_hits

            output[query_index] = [
                (
                    int(internal_id),
                    int(hit_count),
                )
                for internal_id, hit_count in zip(
                    candidates[keep],
                    hits[keep],
                )
            ]

            self.hit_counts[candidates] = 0

        return output


class P3Fast:
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

        for country_index in self.country_indexes.values():
            index = country_index.index

            if hasattr(index, "nprobe"):
                index.nprobe = min(
                    self.nprobe,
                    int(index.nlist),
                )

    def query_batch(self, rows):
        output = [[] for _ in rows]

        if not rows:
            return output

        by_country = {}

        for position, row in enumerate(rows):
            by_country.setdefault(
                str(row[3]),
                [],
            ).append(position)

        for country, positions in by_country.items():
            country_index = self.country_indexes.get(country)

            if country_index is None:
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
                country_index.index.ntotal
            )

            if ntotal == 0:
                continue

            k = min(
                self.top_k,
                ntotal,
            )

            distances, positions_out = (
                country_index.index.search(
                    np.ascontiguousarray(
                        vectors,
                        dtype=np.float32,
                    ),
                    k,
                )
            )

            for local, row_position in enumerate(positions):
                valid = positions_out[local] >= 0

                if not np.any(valid):
                    continue

                raw_positions = positions_out[local][valid]
                raw_ids = country_index.id_store[
                    raw_positions
                ]
                scores = distances[local][valid]

                output[row_position] = [
                    (
                        bytes(raw_id),
                        float(score),
                    )
                    for raw_id, score in zip(
                        raw_ids,
                        scores,
                    )
                ]

        return output


class FastStreamingPipeline:
    def __init__(
        self,
        p1_index,
        p2_index,
        p3_index,
        max_candidates=1000,
        p2_min_band_hits=2,
        p3_top_k=500,
        p3_nprobe=128,
        p3_batch_size=1024,
        chunk_size=512,
        p3_quota=0.30,
    ):
        print("Loading P1...", flush=True)
        self.p1 = P1Fast(p1_index)

        print("Loading P2...", flush=True)
        self.p2 = P2Fast(
            p2_index,
            min_band_hits=p2_min_band_hits,
        )

        print("Loading P3...", flush=True)
        self.p3 = P3Fast(
            p3_index,
            top_k=p3_top_k,
            nprobe=p3_nprobe,
            batch_size=p3_batch_size,
        )

        self.max_candidates = int(max_candidates)
        self.chunk_size = int(chunk_size)
        self.p3_quota = float(p3_quota)

        if not 0.0 <= self.p3_quota <= 1.0:
            raise ValueError("p3_quota must be between 0 and 1")

    @staticmethod
    def _merge(p1_values, p2_values, p3_values, p1_ids, p2_ids):
        merged = {}

        # P1 -> public entity ID.
        for internal_id, score in p1_values:
            candidate_id = bytes(
                p1_ids[internal_id]
            )

            info = merged.get(candidate_id)

            if info is None:
                merged[candidate_id] = CandidateInfo(
                    sources=1,
                    p1_score=int(score),
                )
            else:
                info.sources |= 1
                info.p1_score = max(
                    info.p1_score,
                    int(score),
                )

        # P2 -> public entity ID.
        for internal_id, band_hits in p2_values:
            candidate_id = bytes(
                p2_ids[internal_id]
            )

            info = merged.get(candidate_id)

            if info is None:
                merged[candidate_id] = CandidateInfo(
                    sources=2,
                    p2_score=int(band_hits),
                )
            else:
                info.sources |= 2
                info.p2_score = max(
                    info.p2_score,
                    int(band_hits),
                )

        # P3 -> public entity ID + cosine score.
        for candidate_id, score in p3_values:
            info = merged.get(candidate_id)

            if info is None:
                merged[candidate_id] = CandidateInfo(
                    sources=4,
                    p3_score=float(score),
                )
            else:
                info.sources |= 4
                info.p3_score = max(
                    info.p3_score,
                    float(score),
                )

        return merged

    @staticmethod
    def _source_count(sources):
        return (
            int(bool(sources & 1))
            + int(bool(sources & 2))
            + int(bool(sources & 4))
        )

    @classmethod
    def _rank_agreement(cls, item):
        candidate_id, info = item

        return (
            cls._source_count(info.sources),
            info.p2_score,
            min(info.p1_score, 100),
            info.p3_score,
            candidate_id,
        )

    @staticmethod
    def _rank_lexical(item):
        candidate_id, info = item

        return (
            min(info.p1_score, 100),
            info.p2_score,
            info.p3_score,
            candidate_id,
        )

    @staticmethod
    def _rank_embedding(item):
        candidate_id, info = item

        return (
            info.p3_score,
            info.p2_score,
            min(info.p1_score, 100),
            candidate_id,
        )

    def prune(self, merged):
        if not merged:
            return []

        items = list(merged.items())

        if len(items) <= self.max_candidates:
            items.sort(
                key=self._rank_agreement,
                reverse=True,
            )
            return [
                candidate_id
                for candidate_id, _
                in items
            ]

        # Always protect candidates found by >=2 independent blockers.
        agreement = [
            item
            for item in items
            if self._source_count(item[1].sources) >= 2
        ]

        if len(agreement) >= self.max_candidates:
            agreement.sort(
                key=self._rank_agreement,
                reverse=True,
            )
            return [
                candidate_id
                for candidate_id, _
                in agreement[:self.max_candidates]
            ]

        chosen = list(agreement)
        chosen_ids = {
            candidate_id
            for candidate_id, _
            in chosen
        }

        remaining = [
            item
            for item in items
            if item[0] not in chosen_ids
        ]

        p12_only = [
            item
            for item in remaining
            if not (item[1].sources & 4)
        ]

        p3_only = [
            item
            for item in remaining
            if item[1].sources == 4
        ]

        slots = self.max_candidates - len(chosen)

        # Reserve a modest fraction specifically for P3-only candidates.
        p3_slots = min(
            len(p3_only),
            int(round(slots * self.p3_quota)),
        )

        p12_slots = slots - p3_slots

        p12_only.sort(
            key=self._rank_lexical,
            reverse=True,
        )
        p3_only.sort(
            key=self._rank_embedding,
            reverse=True,
        )

        # If one side cannot fill its quota, return the unused slots to the
        # other side rather than under-filling the candidate set.
        chosen.extend(
            p12_only[:p12_slots]
        )
        chosen.extend(
            p3_only[:p3_slots]
        )

        if len(chosen) < self.max_candidates:
            already = {
                candidate_id
                for candidate_id, _
                in chosen
            }

            leftovers = [
                item
                for item in remaining
                if item[0] not in already
            ]
            leftovers.sort(
                key=self._rank_agreement,
                reverse=True,
            )

            chosen.extend(
                leftovers[
                    : self.max_candidates - len(chosen)
                ]
            )

        return [
            candidate_id
            for candidate_id, _
            in chosen[:self.max_candidates]
        ]

    def iter_rows(self, source1_path):
        for chunk in pd.read_csv(
            source1_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=S1_COLUMNS,
            chunksize=self.chunk_size,
        ):
            rows = list(
                chunk.itertuples(
                    index=False,
                    name=None,
                )
            )

            p1_results = [
                self.p1.query(
                    str(row[1]),
                    str(row[3]),
                )
                for row in rows
            ]

            p2_results = self.p2.query_batch(rows)
            p3_results = self.p3.query_batch(rows)

            for row, p1_values, p2_values, p3_values in zip(
                rows,
                p1_results,
                p2_results,
                p3_results,
            ):
                merged = self._merge(
                    p1_values,
                    p2_values,
                    p3_values,
                    self.p1.index.entity_ids,
                    self.p2.entity_ids,
                )

                yield row, self.prune(merged)


def _load_truth(path, wanted):
    truth = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as fh:
        header = fh.readline().rstrip("\r\n").split("\t")

        s1_i = header.index("source1_entity_id")
        match_i = header.index("matched_entity_ids")

        for line in fh:
            parts = line.rstrip("\r\n").split("\t")

            if len(parts) <= max(s1_i, match_i):
                continue

            s1_id = parts[s1_i]

            if s1_id not in wanted:
                continue

            raw = parts[match_i]
            truth[s1_id] = (
                set(raw.split(","))
                if raw
                else set()
            )

    return truth


def run(args):
    pipeline = FastStreamingPipeline(
        p1_index=args.p1_index,
        p2_index=args.p2_index,
        p3_index=args.p3_index,
        max_candidates=args.max_candidates,
        p2_min_band_hits=args.p2_min_band_hits,
        p3_top_k=args.p3_top_k,
        p3_nprobe=args.p3_nprobe,
        p3_batch_size=args.p3_batch_size,
        chunk_size=args.chunk_size,
        p3_quota=args.p3_quota,
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
            header = fh.readline().rstrip("\r\n").split("\t")
            id_i = header.index("entity_id")

            for line in fh:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > id_i and parts[id_i]:
                    wanted.add(parts[id_i])

        truth = _load_truth(
            args.ground_truth,
            wanted,
        )

    out = None

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        out = open(
            output_path,
            "w",
            encoding="utf-8",
            buffering=1024 * 1024,
        )
        out.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

    total = 0
    total_candidates = 0
    zeros = 0

    total_true = 0
    found_true = 0
    entities_with_match = 0
    fully_recalled = 0

    start = time.time()

    try:
        for row, candidates in pipeline.iter_rows(
            args.s1
        ):
            s1_id = str(row[0])

            total += 1
            total_candidates += len(candidates)

            if not candidates:
                zeros += 1

            if out is not None:
                out.write(
                    s1_id
                    + "\t"
                    + ",".join(
                        candidate.decode("utf-8")
                        for candidate in candidates
                    )
                    + "\n"
                )

            if truth is not None:
                true = truth.get(
                    s1_id,
                    set(),
                )

                if true:
                    entities_with_match += 1
                    total_true += len(true)

                    found = sum(
                        1
                        for candidate in candidates
                        if candidate.decode("utf-8") in true
                    )

                    found_true += found

                    if found == len(true):
                        fully_recalled += 1

            if total % 1000 == 0:
                elapsed = time.time() - start
                print(
                    f"  queried={total:,}"
                    f" | avg={total_candidates / total:.2f}"
                    f" | rate={total / elapsed:.1f}/s"
                    f" | zero={zeros:,}",
                    flush=True,
                )
    finally:
        if out is not None:
            out.close()

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("FAST STREAMING 3-BLOCKER PIPELINE")
    print("=" * 70)
    print(f"S1 queries        : {total:,}")
    print(
        f"Average candidates: "
        f"{total_candidates / total:.2f}"
        if total
        else "Average candidates: 0"
    )
    print(f"Candidate cap     : {args.max_candidates:,}")
    print(f"Zero candidates   : {zeros:,}")
    print(f"Query time        : {elapsed:.1f}s")
    print(
        f"Query rate        : "
        f"{total / elapsed:.2f} S1/s"
        if elapsed
        else "Query rate        : 0 S1/s"
    )

    if truth is not None:
        pairwise = (
            found_true / total_true
            if total_true
            else 0.0
        )
        full_rate = (
            fully_recalled / entities_with_match
            if entities_with_match
            else 0.0
        )

        print()
        print("DEV RECALL")
        print(f"True pairs        : {total_true:,}")
        print(f"Found pairs       : {found_true:,}")
        print(f"Pairwise recall   : {pairwise:.6f}")
        print(f"Full recall       : {full_rate:.6f}")

    if out is not None:
        print(f"Output            : {args.output}")

    print("=" * 70)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
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
        help="Per-S1 pruning cap. Default: 1000.",
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
        default=512,
    )
    parser.add_argument(
        "--p3-quota",
        type=float,
        default=0.30,
        help="Fraction of single-source slots reserved for P3-only.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Dev-only candidate TSV. Avoid for full train/test.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.max_candidates < 1:
        parser.error("--max-candidates must be >= 1")

    if args.p2_min_band_hits < 1:
        parser.error("--p2-min-band-hits must be >= 1")

    if args.p3_top_k < 1:
        parser.error("--p3-top-k must be >= 1")

    if args.p3_nprobe < 1:
        parser.error("--p3-nprobe must be >= 1")

    if args.p3_batch_size < 1:
        parser.error("--p3-batch-size must be >= 1")

    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    if not 0.0 <= args.p3_quota <= 1.0:
        parser.error("--p3-quota must be between 0 and 1")

    run(args)


if __name__ == "__main__":
    main()
