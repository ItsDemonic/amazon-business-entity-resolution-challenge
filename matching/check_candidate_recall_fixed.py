"""
Fast candidate-recall diagnostic.

Reads only:
  - ranked candidate artifact
  - validation S1 IDs
  - train ground truth
  - P1 global-ID -> raw entity-ID map

It does NOT compute matcher features or model scores.
Use this before changing the matcher: if true matches are missing from
the candidate artifact, the blocker is the bottleneck.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_id_file(path: str) -> set[str]:
    out = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            x = line.strip()
            if x and x != "entity_id":
                out.add(x)
    return out


def load_truth(path: str, wanted: set[str]):
    truth = {sid: set() for sid in wanted}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row.get("source1_entity_id", "")
            if sid not in truth:
                continue
            raw = (row.get("matched_entity_ids") or "").strip()
            if raw:
                truth[sid] = {
                    x.strip() for x in raw.split(",") if x.strip()
                }
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


def sbytes(x) -> bytes:
    return bytes(x).rstrip(b"\x00")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-prefix", required=True)
    p.add_argument("--validation-ids", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--entity-ids", required=True)
    args = p.parse_args()

    artifact_ids, offsets, artifact_s1_ids = load_artifact(
        args.candidate_prefix
    )
    # The candidate artifact contains the exact queries being evaluated.
    # --validation-ids may contain the full 441k validation split while the
    # blocker was intentionally run with --limit 10000.
    wanted = {
        sbytes(x).decode("utf-8", errors="ignore")
        for x in artifact_s1_ids
    }
    truth = load_truth(args.ground_truth, wanted)

    # Only truth IDs for the artifact's actual 10k queries are needed.
    wanted_raw = set()
    for vals in truth.values():
        wanted_raw.update(vals)

    entity_ids = np.load(args.entity_ids, mmap_mode="r")
    if len(entity_ids) == 0:
        raise RuntimeError("entity_ids.npy is empty")

    print(f"Artifact queries : {len(artifact_s1_ids):,}")
    print(f"Validation IDs   : {len(wanted):,}")
    print(f"Truth entities   : {len(wanted_raw):,}")
    print(f"Global ID count  : {len(entity_ids):,}")
    print("Building truth entity -> global ID map...")

    raw_to_gid = {}
    remaining = set(wanted_raw)

    # Scan the 10.3M ID map once. This is cheap compared with feature scoring.
    for gid, raw in enumerate(entity_ids):
        key = sbytes(raw).decode("utf-8", errors="ignore")
        if key in remaining:
            raw_to_gid[key] = gid
            remaining.remove(key)
            if not remaining:
                break

    if remaining:
        print(
            f"WARNING: {len(remaining):,} truth entity IDs were not found "
            "in entity_ids.npy"
        )

    # Convert truth sets to global integer IDs.
    truth_gids = {}
    for sid, vals in truth.items():
        truth_gids[sid] = {
            raw_to_gid[v] for v in vals if v in raw_to_gid
        }

    # Keep artifact order for fast row lookup.
    row_for_sid = {
        sbytes(raw).decode("utf-8", errors="ignore"): i
        for i, raw in enumerate(artifact_s1_ids)
    }

    ks = [8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 1500]

    total_truth = sum(len(v) for v in truth_gids.values())
    matched_queries = sum(1 for v in truth_gids.values() if v)
    print()
    print("=" * 78)
    print("CANDIDATE RECALL DIAGNOSTIC")
    print("=" * 78)
    print(f"Matched validation queries: {matched_queries:,}")
    print(f"Total ground-truth pairs   : {total_truth:,}")
    print()

    for k in ks:
        covered_pairs = 0
        fully_covered = 0
        nonempty = 0
        empty_correct = 0

        for sid, tgids in truth_gids.items():
            row = row_for_sid[sid]
            lo = int(offsets[row])
            hi = min(int(offsets[row + 1]), lo + k)
            cand = artifact_ids[lo:hi]

            found = len(tgids.intersection(map(int, cand)))

            if tgids:
                nonempty += 1
                covered_pairs += found
                if found == len(tgids):
                    fully_covered += 1
            else:
                # A singleton/no-match query only needs zero candidates at
                # the final matcher stage; candidate recall itself is 1.0.
                empty_correct += 1

        pair_recall = (
            covered_pairs / total_truth if total_truth else 1.0
        )
        full_recall = (
            fully_covered / nonempty if nonempty else 1.0
        )

        print(
            f"top-{k:<4} "
            f"pair_recall={pair_recall:8.4%}  "
            f"full_recall={full_recall:8.4%}  "
            f"fully_covered={fully_covered:5,}/{nonempty:,}"
        )

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
