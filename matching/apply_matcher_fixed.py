"""
Production matcher inference.

Important scaling choice:
- Reads a ranked compact candidate artifact.
- Keeps only top rank-k candidates for matching.
- Finds exact normalized-name/address candidates via uint64 hashes.
- Computes full string features only for the top model-k candidates.
- Streams Source 1.
- Builds output/candidate_pairs.tsv and output/matching_results.tsv.

No Python feature calculation is performed on millions of all-1500 pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from common.normalize import (
    normalize_address,
    normalize_name,
)
from features.similarity_features import (
    FEATURE_NAMES,
    pair_features_prepared,
    prepare_record,
)
from matching.build_matcher_data_fast import (
    load_artifact,
)


def stable_hash64(value: str) -> np.uint64:
    # Stable across processes/machines.
    return np.frombuffer(
        hashlib.blake2b(
            value.encode("utf-8"),
            digest_size=8,
        ).digest(),
        dtype="<u8",
    )[0]


class CandidateStore:
    """
    Disk-backed normalized candidate strings for only the candidates that
    occur in the top-rank window.
    """

    def __init__(self, root: str):
        root = Path(root)
        self.root = root
        self.name_blob = open(
            root / "names.bin",
            "rb",
        )
        self.addr_blob = open(
            root / "addresses.bin",
            "rb",
        )
        self.name_offsets = np.load(
            root / "name_offsets.npy",
            mmap_mode="r",
        )
        self.addr_offsets = np.load(
            root / "addr_offsets.npy",
            mmap_mode="r",
        )
        self.needed = np.load(
            root / "needed.npy",
            mmap_mode="r",
        )
        self.name_hash = np.load(
            root / "name_hash.npy",
            mmap_mode="r",
        )
        self.addr_hash = np.load(
            root / "addr_hash.npy",
            mmap_mode="r",
        )

    @lru_cache(maxsize=100_000)
    def get_name(self, gid: int) -> str:
        a = int(self.name_offsets[gid])
        b = int(self.name_offsets[gid + 1])
        if a == b:
            return ""
        self.name_blob.seek(a)
        return self.name_blob.read(b - a).decode("utf-8")

    @lru_cache(maxsize=100_000)
    def get_address(self, gid: int) -> str:
        a = int(self.addr_offsets[gid])
        b = int(self.addr_offsets[gid + 1])
        if a == b:
            return ""
        self.addr_blob.seek(a)
        return self.addr_blob.read(b - a).decode("utf-8")


def prepare_store(
    store_dir: str,
    candidate_prefix: str,
    s2_path: str,
    s3_path: str,
    top_k: int,
):
    """
    One-time sequential pass:
    1) discover all GIDs used by the top-k ranked candidate window
    2) stream S2/S3 once and store only needed normalized strings
    """
    root = Path(store_dir)
    root.mkdir(parents=True, exist_ok=True)

    ids, offsets, _ = load_artifact(candidate_prefix)

    # Determine global candidate count from the highest GID observed in
    # the artifact. The test pool is normally ~10M records; allocate the
    # exact size later after scanning the source files.
    highest = 0
    needed = None

    # Build needed set as a packed boolean array using the known candidate
    # count inferred from S2/S3 row counts.
    s2_count = sum(1 for _ in open(s2_path, "r", encoding="utf-8")) - 1
    s3_count = sum(1 for _ in open(s3_path, "r", encoding="utf-8")) - 1
    total_candidates = s2_count + s3_count

    needed = np.zeros(
        total_candidates,
        dtype=np.uint8,
    )

    for row in range(len(offsets) - 1):
        a = int(offsets[row])
        b = int(offsets[row + 1])
        end = min(b, a + top_k)
        if end > a:
            used = np.asarray(
                ids[a:end],
                dtype=np.uint32,
            )
            needed[used] = 1

    needed_count = int(needed.sum())
    print(
        f"Top-{top_k} unique candidates needed: {needed_count:,}",
        flush=True,
    )

    name_offsets = np.lib.format.open_memmap(
        root / "name_offsets.npy",
        mode="w+",
        dtype=np.uint64,
        shape=(total_candidates + 1,),
    )
    addr_offsets = np.lib.format.open_memmap(
        root / "addr_offsets.npy",
        mode="w+",
        dtype=np.uint64,
        shape=(total_candidates + 1,),
    )
    name_hash = np.lib.format.open_memmap(
        root / "name_hash.npy",
        mode="w+",
        dtype=np.uint64,
        shape=(total_candidates,),
    )
    addr_hash = np.lib.format.open_memmap(
        root / "addr_hash.npy",
        mode="w+",
        dtype=np.uint64,
        shape=(total_candidates,),
    )

    with open(root / "names.bin", "wb", buffering=4 * 1024 * 1024) as nf, \
         open(root / "addresses.bin", "wb", buffering=4 * 1024 * 1024) as af:

        npos = 0
        apos = 0
        name_offsets[0] = 0
        addr_offsets[0] = 0

        gid = 0

        for path in (s2_path, s3_path):
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")

                for row in reader:
                    name = normalize_name(row.get("business_name", ""))
                    address = normalize_address(
                        row.get("business_address", "")
                    )

                    if needed[gid]:
                        nb = name.encode("utf-8")
                        ab = address.encode("utf-8")

                        nf.write(nb)
                        af.write(ab)

                        npos += len(nb)
                        apos += len(ab)

                        name_hash[gid] = stable_hash64(name)
                        addr_hash[gid] = stable_hash64(address)
                    else:
                        name_hash[gid] = stable_hash64("")
                        addr_hash[gid] = stable_hash64("")

                    name_offsets[gid + 1] = npos
                    addr_offsets[gid + 1] = apos
                    gid += 1

            print(
                f"Stored through {path}: global IDs now {gid:,}",
                flush=True,
            )

    name_offsets.flush()
    addr_offsets.flush()
    name_hash.flush()
    addr_hash.flush()

    np.save(
        root / "needed.npy",
        needed,
    )

    with open(root / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "total_candidates": total_candidates,
                "needed_candidates": needed_count,
                "top_k": top_k,
            },
            fh,
            indent=2,
        )

    del name_offsets
    del addr_offsets
    del name_hash
    del addr_hash

    print("Candidate store ready.", flush=True)


def read_thresholds(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default="matching/matcher_model.joblib")
    p.add_argument("--threshold-json", default="matching/threshold.json")

    p.add_argument(
        "--candidate-prefix",
        default="blocking/final_test_ranked",
    )
    p.add_argument(
        "--entity-ids",
        required=True,
        help="Test P1 index entity_ids.npy (same global S2/S3 ordering).",
    )

    p.add_argument("--s1", default="dataset/test/test_source1.tsv")
    p.add_argument("--s2", default="dataset/test/test_source2.tsv")
    p.add_argument("--s3", default="dataset/test/test_source3.tsv")

    p.add_argument(
        "--store-dir",
        default="matching/test_candidate_store",
    )

    p.add_argument("--rank-k", type=int, default=None)
    p.add_argument("--model-k", type=int, default=None)
    p.add_argument("--max-matches", type=int, default=None)
    p.add_argument("--relative-gate", type=float, default=None)

    p.add_argument(
        "--candidate-output",
        default="output/candidate_pairs.tsv",
    )
    p.add_argument(
        "--matching-output",
        default="output/matching_results.tsv",
    )
    p.add_argument(
        "--reuse-store",
        action="store_true",
    )

    args = p.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]

    if [str(x) for x in bundle["feature_names"]] != FEATURE_NAMES:
        raise ValueError("Model feature schema mismatch.")

    cfg = read_thresholds(args.threshold_json)

    rank_k = (
        args.rank_k
        if args.rank_k is not None
        else int(cfg.get("rank_k", 64))
    )
    max_matches = (
        args.max_matches
        if args.max_matches is not None
        else int(cfg.get("max_matches", 12))
    )
    relative_gate = (
        args.relative_gate
        if args.relative_gate is not None
        else float(cfg.get("relative_gate", 0.0))
    )

    # Evaluation tunes rank_k, so production must score the same number of
    # candidates. A separate default of 16 silently produced a different
    # pipeline and dropped true matches ranked 17..rank_k.
    model_k = (
        args.model_k
        if args.model_k is not None
        else rank_k
    )
    model_k = min(model_k, rank_k)

    if not args.reuse_store or not Path(args.store_dir, "meta.json").exists():
        prepare_store(
            args.store_dir,
            args.candidate_prefix,
            args.s2,
            args.s3,
            rank_k,
        )

    store = CandidateStore(args.store_dir)

    entity_ids = np.load(
        args.entity_ids,
        mmap_mode="r",
    )

    ids, offsets, s1_ids = load_artifact(
        args.candidate_prefix
    )

    Path(args.candidate_output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(args.matching_output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.s1, "r", encoding="utf-8", newline="") as sf, \
         open(args.candidate_output, "w", encoding="utf-8", buffering=2 * 1024 * 1024) as cf, \
         open(args.matching_output, "w", encoding="utf-8", buffering=2 * 1024 * 1024) as mf:

        reader = csv.DictReader(sf, delimiter="\t")

        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        mf.write("source1_entity_id\tmatched_entity_ids\n")

        for row, s1row in enumerate(reader):
            if row >= len(s1_ids):
                raise ValueError("S1 has more rows than the candidate artifact.")

            sid = bytes(s1_ids[row]).rstrip(b"\x00").decode("utf-8")
            source_sid = s1row["entity_id"]

            if sid != source_sid:
                raise ValueError(
                    f"S1/artifact order mismatch at row {row}: "
                    f"{source_sid} != {sid}"
                )

            a = int(offsets[row])
            b = int(offsets[row + 1])

            candidate_gids = np.asarray(
                ids[a:min(b, a + rank_k)],
                dtype=np.uint32,
            )

            # Candidate output is exactly the same ranked set used by the matcher.
            output_ids = [
                entity_ids[int(gid)].decode("utf-8")
                for gid in candidate_gids.tolist()
            ]

            cf.write(
                sid
                + "\t"
                + ",".join(output_ids)
                + "\n"
            )

            if len(candidate_gids) == 0:
                mf.write(sid + "\t\n")
                continue

            q_name = s1row.get("business_name", "")
            q_addr = s1row.get("business_address", "")
            q_country = s1row.get("country", "")

            qprep = prepare_record(q_name, q_addr, q_country)

            q_name_hash = stable_hash64(qprep.name)
            q_addr_hash = stable_hash64(qprep.address)

            # Only use an automatic shortcut when BOTH normalized name and
            # address are exact. Shared addresses alone are too risky for
            # precision-heavy F0.5.
            exact = (
                (store.name_hash[candidate_gids] == q_name_hash)
                & (store.addr_hash[candidate_gids] == q_addr_hash)
                & (q_name_hash != stable_hash64(""))
                & (q_addr_hash != stable_hash64(""))
            )

            selected = set(
                int(x)
                for x in candidate_gids[exact].tolist()
            )

            model_gids = candidate_gids[
                : min(model_k, len(candidate_gids))
            ]

            X = np.empty(
                (len(model_gids), len(FEATURE_NAMES)),
                dtype=np.float32,
            )

            for j, gid in enumerate(model_gids.tolist()):
                cand = prepare_record(
                    store.get_name(int(gid)),
                    store.get_address(int(gid)),
                    q_country,
                )
                X[j] = pair_features_prepared(
                    qprep,
                    cand,
                    blocker_rank=j + 1,
                )

            if len(X):
                scores = model.predict_proba(X)[:, 1]
            else:
                scores = np.empty(0, dtype=np.float32)

            for gid, score in zip(model_gids.tolist(), scores.tolist()):
                if float(score) >= float(cfg["threshold"]):
                    selected.add(int(gid))

            # Preserve model ranking order for output.
            ordered = []
            for gid in candidate_gids.tolist():
                if int(gid) in selected:
                    ordered.append(int(gid))

            # Optional per-query limit.
            if max_matches > 0 and len(ordered) > max_matches:
                ranked_scores = []
                model_pos = {
                    int(g): float(s)
                    for g, s in zip(
                        model_gids.tolist(),
                        scores.tolist(),
                    )
                }

                ordered.sort(
                    key=lambda g: model_pos.get(g, 1.0),
                    reverse=True,
                )
                ordered = ordered[:max_matches]

            matched_entity_ids = [
                entity_ids[int(gid)].decode("utf-8")
                for gid in ordered
            ]

            mf.write(
                sid
                + "\t"
                + ",".join(matched_entity_ids)
                + "\n"
            )

            if (row + 1) % 5000 == 0:
                print(
                    f"  matched {row + 1:,}/{len(s1_ids):,}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
