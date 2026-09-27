"""
Fast hard-negative matcher dataset builder.

Key idea:
- use a ranked candidate artifact
- keep ALL available positives
- keep top ranked hard negatives
- add a small random tail sample
- prepare each string once, not once per pair
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from features.similarity_features import (
    FEATURE_NAMES,
    pair_features_prepared,
    prepare_record,
)


def read_id_file(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            x = line.strip()
            if x and x != "entity_id":
                out.append(x)
    return out


def load_s1(path: str, wanted: set[str]):
    out = {}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row["entity_id"]
            if sid in wanted:
                out[sid] = {
                    "business_name": row.get("business_name", ""),
                    "business_address": row.get("business_address", ""),
                    "country": row.get("country", ""),
                }
                if len(out) == len(wanted):
                    break

    return out


def load_truth(path: str, wanted: set[str]):
    truth = {sid: set() for sid in wanted}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row["source1_entity_id"]
            if sid not in truth:
                continue
            raw = (row.get("matched_entity_ids") or "").strip()
            truth[sid] = (
                {x.strip() for x in raw.split(",") if x.strip()}
                if raw
                else set()
            )

    return truth


def load_artifact(prefix: str):
    prefix = Path(prefix)
    offsets = np.load(str(prefix) + ".offsets.npy", mmap_mode="r")
    s1_ids = np.load(str(prefix) + ".s1_ids.npy", mmap_mode="r")
    ids = np.memmap(
        str(prefix) + ".ids.bin",
        dtype=np.uint32,
        mode="r",
    )
    return ids, offsets, s1_ids


def artifact_lookup(s1_ids, wanted: set[str]):
    wanted_bytes = {x.encode("utf-8"): x for x in wanted}
    out = {}
    remaining = set(wanted_bytes)

    for i, raw in enumerate(s1_ids):
        key = bytes(raw).rstrip(b"\x00")
        if key in remaining:
            out[wanted_bytes[key]] = i
            remaining.remove(key)
            if not remaining:
                break

    if remaining:
        raise ValueError(
            f"{len(remaining)} selected S1 IDs are missing from artifact."
        )

    return out


def load_candidate_rows(
    s2: str,
    s3: str,
    needed: set[int],
):
    out = {}
    gid = 0

    for path in (s2, s3):
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")

            for row in reader:
                if gid in needed:
                    out[gid] = (
                        row["entity_id"],
                        prepare_record(
                            row.get("business_name", ""),
                            row.get("business_address", ""),
                            row.get("country", ""),
                        ),
                    )
                gid += 1

        print(
            f"Scanned {path}: global IDs now {gid:,}",
            flush=True,
        )

    missing = needed - set(out)
    if missing:
        raise ValueError(
            f"{len(missing)} candidate IDs were not found in S2/S3."
        )

    return out


def build_dataset(args):
    rng = np.random.default_rng(args.seed)

    # The candidate artifact may contain only a sampled subset of the
    # training IDs (for example, the first 20k rows selected with --limit).
    # Select matcher queries FROM THE ARTIFACT ITSELF so every chosen S1 row
    # is guaranteed to have candidates.
    train_ids = set(read_id_file(args.train_ids))

    artifact_ids, offsets, artifact_s1_ids = load_artifact(
        args.candidate_prefix
    )

    available = []
    train_bytes = {x.encode("utf-8") for x in train_ids}

    for raw in artifact_s1_ids:
        key = bytes(raw).rstrip(b"\x00")
        if key in train_bytes:
            available.append(key.decode("utf-8"))

    if not available:
        raise ValueError(
            "No candidate-artifact S1 IDs overlap the supplied training ID file."
        )

    n_queries = min(args.queries, len(available))

    chosen = rng.choice(
        np.asarray(available, dtype=object),
        size=n_queries,
        replace=False,
    ).tolist()
    chosen = [str(x) for x in chosen]
    wanted = set(chosen)

    s1 = load_s1(args.s1, wanted)
    truth = load_truth(args.ground_truth, wanted)
    rows = artifact_lookup(artifact_s1_ids, wanted)

    print(
        f"Artifact-overlap training queries: {len(available):,}",
        flush=True,
    )

    entity_ids = np.load(
        args.entity_ids,
        mmap_mode="r",
    )

    final_by_sid: dict[str, list[tuple[int, int]]] = {}
    needed_global: set[int] = set()

    # Select hard negatives from the ranked prefix and restore every
    # positive present anywhere in the 1500-candidate list. We can determine
    # labels directly from the compact entity_ids.npy; no giant Python
    # dictionary of all 1500 candidates/query is built.
    for sid in chosen:
        row_idx = rows[sid]
        start = int(offsets[row_idx])
        end = int(offsets[row_idx + 1])

        all_ids = np.asarray(
            artifact_ids[start:end],
            dtype=np.uint32,
        )

        true_entity_ids = truth[sid]
        selected = []

        for rank0, gid0 in enumerate(all_ids.tolist()):
            gid = int(gid0)
            entity_id = entity_ids[gid].decode("utf-8")

            if entity_id in true_entity_ids:
                selected.append((gid, rank0 + 1))
                needed_global.add(gid)
            elif rank0 < args.hard_negatives_scan:
                selected.append((gid, rank0 + 1))
                needed_global.add(gid)

        # Random tail negatives are selected from the remainder only after
        # determining positives, but they are added later below.
        final_by_sid[sid] = selected

    # Rebuild exact positive/negative selections now that entity IDs are known.
    final_pairs: dict[str, list[tuple[int, int]]] = {}

    for sid in chosen:
        row_idx = rows[sid]
        start = int(offsets[row_idx])
        end = int(offsets[row_idx + 1])
        all_ids = np.asarray(
            artifact_ids[start:end],
            dtype=np.uint32,
        )

        true_ids = truth[sid]

        # Preserve the already selected positives + hard negatives.
        selected = list(final_by_sid[sid])
        selected_set = {gid for gid, _ in selected}

        tail_pool = []

        for rank0, gid0 in enumerate(all_ids.tolist()):
            gid = int(gid0)
            if gid in selected_set:
                continue

            rank = rank0 + 1
            entity_id = entity_ids[gid].decode("utf-8")

            if entity_id not in true_ids:
                tail_pool.append((gid, rank))

        if len(tail_pool) > args.tail_negatives:
            idx = rng.choice(
                len(tail_pool),
                size=args.tail_negatives,
                replace=False,
            )
            idx.sort()
            tail_pool = [tail_pool[int(i)] for i in idx]

        selected.extend(tail_pool)

        for gid, _ in tail_pool:
            needed_global.add(gid)

        final_pairs[sid] = selected

    # Load candidate rows only AFTER tail negatives have been added.
    # Otherwise tail negatives are not present in `candidate_records`.
    candidate_records = load_candidate_rows(
        args.s2,
        args.s3,
        needed_global,
    )

    total_pairs = sum(len(v) for v in final_pairs.values())

    X = np.empty(
        (total_pairs, len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    y = np.empty(total_pairs, dtype=np.uint8)

    out = 0

    for qi, sid in enumerate(chosen, start=1):
        qraw = s1[sid]
        qprep = prepare_record(
            qraw["business_name"],
            qraw["business_address"],
            qraw["country"],
        )

        true_ids = truth[sid]

        for gid, rank in final_pairs[sid]:
            entity_id, cand = candidate_records[gid]

            X[out] = pair_features_prepared(
                qprep,
                cand,
                blocker_rank=rank,
            )
            y[out] = int(entity_id in true_ids)
            out += 1

        if qi % 1000 == 0:
            print(
                f"  features: {qi:,}/{len(chosen):,}",
                flush=True,
            )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        feature_names=np.asarray(
            FEATURE_NAMES,
            dtype=object,
        ),
        n_queries=np.asarray(len(chosen)),
        seed=np.asarray(args.seed),
    )

    print()
    print("=" * 60)
    print("FAST MATCHER DATA COMPLETE")
    print("=" * 60)
    print(f"Queries  : {len(chosen):,}")
    print(f"Pairs    : {len(y):,}")
    print(f"Positives: {int(y.sum()):,}")
    print(f"Negatives: {int((y == 0).sum()):,}")
    print(f"Features : {len(FEATURE_NAMES)}")
    print(f"Output   : {args.output}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--candidate-prefix", required=True)
    p.add_argument("--s1", required=True)
    p.add_argument("--s2", required=True)
    p.add_argument("--s3", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--train-ids", required=True)
    p.add_argument("--output", default="matching/matcher_train.npz")
    p.add_argument(
        "--entity-ids",
        required=True,
        help="Global candidate ID -> entity_id array matching the blocker artifact.",
    )

    p.add_argument("--queries", type=int, default=20_000)
    p.add_argument("--hard-negatives-scan", type=int, default=160)
    p.add_argument("--tail-negatives", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)

    build_dataset(p.parse_args())


if __name__ == "__main__":
    main()
