"""
Build supervised matcher training data from the compact blocker artifact.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common.normalize import normalize_address, normalize_name
from features.similarity_features import FEATURE_NAMES, pair_features


def read_id_file(path: str) -> list[str]:
    ids = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value and value != "entity_id":
                ids.append(value)
    return ids


def load_selected_s1(path: str, selected_ids: set[str]) -> dict[str, dict[str, str]]:
    out = {}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing required columns.")

        for row in reader:
            entity_id = row["entity_id"]
            if entity_id in selected_ids:
                out[entity_id] = {
                    "business_name": row.get("business_name", ""),
                    "business_address": row.get("business_address", ""),
                    "country": row.get("country", ""),
                }
                if len(out) == len(selected_ids):
                    break

    missing = selected_ids - set(out)
    if missing:
        raise ValueError(f"{len(missing)} selected S1 IDs were not found in {path}.")
    return out


def load_selected_truth(path: str, selected_ids: set[str]) -> dict[str, set[str]]:
    truth = {sid: set() for sid in selected_ids}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing required ground-truth columns.")

        for row in reader:
            sid = row["source1_entity_id"]
            if sid not in truth:
                continue
            raw = (row.get("matched_entity_ids") or "").strip()
            if raw:
                truth[sid] = {x.strip() for x in raw.split(",") if x.strip()}

    return truth


def load_candidate_memmaps(prefix: str):
    prefix = Path(prefix)
    offsets = np.load(str(prefix) + ".offsets.npy", mmap_mode="r")
    s1_ids = np.load(str(prefix) + ".s1_ids.npy", mmap_mode="r")
    candidate_ids = np.memmap(
        str(prefix) + ".ids.bin",
        dtype=np.uint32,
        mode="r",
    )
    return candidate_ids, offsets, s1_ids


def get_query_row_lookup(selected_ids: list[str], s1_ids_memmap: np.ndarray) -> dict[int, str]:
    wanted = {sid.encode("utf-8"): sid for sid in selected_ids}
    row_to_sid = {}
    remaining = set(wanted)

    for i, raw in enumerate(s1_ids_memmap):
        key = bytes(raw).rstrip(b"\x00")
        if key in remaining:
            row_to_sid[i] = wanted[key]
            remaining.remove(key)
            if not remaining:
                break

    if remaining:
        raise ValueError(
            f"{len(remaining)} selected S1 IDs were not found in the candidate artifact."
        )
    return row_to_sid


def collect_selected_candidate_ids(
    selected_row_to_sid,
    offsets,
    candidate_ids,
    truth,
    negatives_per_query,
    rng,
):
    chosen_by_query = {}
    selected_global = []

    for row_idx, sid in selected_row_to_sid.items():
        start = int(offsets[row_idx])
        end = int(offsets[row_idx + 1])
        ids = np.asarray(candidate_ids[start:end], dtype=np.uint32)

        if ids.size == 0:
            chosen_by_query[sid] = ids
            continue

        if len(ids) <= negatives_per_query:
            chosen = ids
        else:
            choice = rng.choice(
                len(ids),
                size=negatives_per_query,
                replace=False,
            )
            chosen = ids[np.sort(choice)]

        chosen_by_query[sid] = np.asarray(chosen, dtype=np.uint32)
        selected_global.append(chosen)

    unique = (
        np.unique(np.concatenate(selected_global))
        if selected_global
        else np.empty(0, dtype=np.uint32)
    )
    return unique, chosen_by_query


def load_candidate_records(
    s2_path: str,
    s3_path: str,
    needed_ids: np.ndarray,
) -> dict[int, tuple[str, str, str, str]]:
    needed = set(int(x) for x in needed_ids.tolist())
    out = {}
    global_id = 0

    for path in (s2_path, s3_path):
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")

            required = {"entity_id", "business_name", "business_address", "country"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} is missing required columns.")

            for row in reader:
                if global_id in needed:
                    out[global_id] = (
                        row["entity_id"],
                        row.get("business_name", ""),
                        row.get("business_address", ""),
                        row.get("country", ""),
                    )
                global_id += 1

        print(f"Scanned {path}: global IDs now {global_id:,}", flush=True)

    missing = needed - set(out)
    if missing:
        raise ValueError(f"{len(missing)} selected candidate IDs were not found in S2/S3.")

    return out


def build_dataset(
    candidate_prefix,
    s1_path,
    s2_path,
    s3_path,
    ground_truth_path,
    train_ids_path,
    entity_ids_path,
    output_path,
    n_queries,
    negatives_per_query,
    seed,
):
    rng = np.random.default_rng(seed)

    train_ids = read_id_file(train_ids_path)
    n_queries = min(n_queries, len(train_ids))

    chosen_query_ids = rng.choice(
        np.asarray(train_ids, dtype=object),
        size=n_queries,
        replace=False,
    ).tolist()
    chosen_query_ids = [str(x) for x in chosen_query_ids]
    selected_set = set(chosen_query_ids)

    print(f"Selected training queries: {len(chosen_query_ids):,}", flush=True)

    s1 = load_selected_s1(s1_path, selected_set)
    truth = load_selected_truth(ground_truth_path, selected_set)

    candidate_ids, offsets, candidate_s1_ids = load_candidate_memmaps(candidate_prefix)
    row_to_sid = get_query_row_lookup(chosen_query_ids, candidate_s1_ids)

    # Internal IDs -> challenge entity IDs.
    entity_ids = np.load(entity_ids_path, mmap_mode="r")

    _, chosen_by_query = collect_selected_candidate_ids(
        row_to_sid,
        offsets,
        candidate_ids,
        truth,
        negatives_per_query,
        rng,
    )

    # Restore every available positive, even if it was not selected by the
    # random negative sampler.
    final_by_query = {}
    sid_to_row = {sid: row_idx for row_idx, sid in row_to_sid.items()}

    for sid, sampled_ids in chosen_by_query.items():
        row_idx = sid_to_row[sid]
        start = int(offsets[row_idx])
        end = int(offsets[row_idx + 1])
        all_ids = np.asarray(candidate_ids[start:end], dtype=np.uint32)

        true_ids = truth.get(sid, set())

        positives = [
            int(internal_id)
            for internal_id in all_ids.tolist()
            if str(entity_ids[internal_id].decode("utf-8")) in true_ids
        ]

        negative_pool = [
            int(x) for x in sampled_ids.tolist() if int(x) not in set(positives)
        ]

        if len(negative_pool) > negatives_per_query:
            negative_pool = rng.choice(
                np.asarray(negative_pool, dtype=np.uint32),
                size=negatives_per_query,
                replace=False,
            ).tolist()

        final_by_query[sid] = np.asarray(
            sorted(set(positives) | set(negative_pool)),
            dtype=np.uint32,
        )

    needed = (
        np.unique(np.concatenate(list(final_by_query.values())))
        if final_by_query
        else np.empty(0, dtype=np.uint32)
    )

    print(f"Unique candidate records needed: {len(needed):,}", flush=True)

    candidate_records = load_candidate_records(s2_path, s3_path, needed)

    total_pairs = sum(len(x) for x in final_by_query.values())
    X = np.empty((total_pairs, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(total_pairs, dtype=np.uint8)

    offset = 0

    for i, sid in enumerate(chosen_query_ids, start=1):
        q = s1[sid]
        chosen = final_by_query.get(sid, np.empty(0, dtype=np.uint32))
        true_ids = truth.get(sid, set())

        for internal_id in chosen.tolist():
            candidate = candidate_records[int(internal_id)]
            cand_entity_id, cand_name, cand_addr, cand_country = candidate

            X[offset] = pair_features(
                q["business_name"],
                q["business_address"],
                q["country"],
                cand_name,
                cand_addr,
                cand_country,
            )
            y[offset] = int(cand_entity_id in true_ids)
            offset += 1

        if i % 1000 == 0:
            print(f"  features: {i:,}/{len(chosen_query_ids):,}", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        n_queries=np.asarray(len(chosen_query_ids)),
        seed=np.asarray(seed),
    )

    positives = int(y.sum())

    print()
    print("=" * 60)
    print("MATCHER DATASET COMPLETE")
    print("=" * 60)
    print(f"Queries  : {len(chosen_query_ids):,}")
    print(f"Pairs    : {len(y):,}")
    print(f"Positives: {positives:,}")
    print(f"Negatives: {len(y) - positives:,}")
    print(f"Features : {len(FEATURE_NAMES)}")
    print(f"Output   : {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--train-s1-ids", required=True)
    parser.add_argument("--entity-ids", required=True)
    parser.add_argument("--output", default="matching/matcher_train.npz")
    parser.add_argument("--queries", type=int, default=20_000)
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    build_dataset(
        candidate_prefix=args.candidate_prefix,
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        ground_truth_path=args.ground_truth,
        train_ids_path=args.train_s1_ids,
        entity_ids_path=args.entity_ids,
        output_path=args.output,
        n_queries=args.queries,
        negatives_per_query=args.negatives,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
