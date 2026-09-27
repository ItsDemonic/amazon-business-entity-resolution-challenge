"""
Fast validation evaluator.

Computes full features only once for the largest requested rank window,
then tunes:
- rank-k
- probability threshold
- per-query maximum matches
- optional relative-to-top score gate

The evaluator is intentionally a validation tool, not a millions-row
production job.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np

from features.similarity_features import (
    FEATURE_NAMES,
    pair_features_prepared,
    prepare_record,
)
from matching.build_matcher_data_fast import (
    artifact_lookup,
    load_artifact,
    load_candidate_rows,
    load_s1,
    load_truth,
    read_id_file,
)


def f05(tp: int, predicted: int, truth_count: int) -> float:
    if truth_count == 0:
        return 1.0 if predicted == 0 else 0.0
    if predicted == 0 or tp == 0:
        return 0.0
    p = tp / predicted
    r = tp / truth_count
    return 1.25 * p * r / (0.25 * p + r)


def evaluate(
    score_rows,
    label_rows,
    truth_counts,
    threshold,
    max_matches,
    relative_gate,
    rank_k,
):
    total = 0.0
    zero = 0

    for scores, labels, truth_count in zip(
        score_rows,
        label_rows,
        truth_counts,
    ):
        n = min(rank_k, len(scores))
        s = scores[:n]
        y = labels[:n]

        if not len(s):
            zero += 1
            total += f05(0, 0, truth_count)
            continue

        mask = s >= threshold

        if relative_gate > 0:
            top = float(s[0])
            mask &= s >= top * relative_gate

        if max_matches > 0 and int(mask.sum()) > max_matches:
            idx = np.flatnonzero(mask)
            keep = idx[np.argsort(-s[idx], kind="stable")[:max_matches]]
            mask2 = np.zeros_like(mask)
            mask2[keep] = True
            mask = mask2

        predicted = int(mask.sum())
        tp = int(y[mask].sum()) if predicted else 0

        if predicted == 0:
            zero += 1

        total += f05(tp, predicted, truth_count)

    count = len(score_rows)
    return total / count if count else 0.0, zero


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default="matching/matcher_model.joblib")
    p.add_argument("--candidate-prefix", default="blocking/matcher_val_ranked")
    p.add_argument("--s1", default="dataset/train/train_source1.tsv")
    p.add_argument("--s2", default="dataset/train/train_source2.tsv")
    p.add_argument("--s3", default="dataset/train/train_source3.tsv")
    p.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    p.add_argument(
        "--validation-ids",
        default="features/validation_output/validation_s1_ids.tsv",
    )

    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--max-rank", type=int, default=256)
    p.add_argument("--output", default="matching/threshold.json")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    start = time.perf_counter()

    bundle = joblib.load(args.model)
    model = bundle["model"]

    if [str(x) for x in bundle["feature_names"]] != FEATURE_NAMES:
        raise ValueError("Model feature schema mismatch.")

    rng = np.random.default_rng(args.seed)

    validation_ids = read_id_file(args.validation_ids)

    artifact_ids, offsets, artifact_s1_ids = load_artifact(
        args.candidate_prefix
    )

    # The ranked artifact may contain only a subset of validation IDs
    # (e.g. when the blocker was run with --limit 10000).
    # Sample only from the IDs actually present in the artifact.
    validation_set = set(validation_ids)
    artifact_validation_ids = []
    for raw in artifact_s1_ids.tolist():
        if isinstance(raw, (bytes, np.bytes_)):
            sid = bytes(raw).rstrip(b"\x00").decode("utf-8")
        else:
            sid = str(raw)
        if sid in validation_set:
            artifact_validation_ids.append(sid)

    if not artifact_validation_ids:
        raise ValueError(
            "No validation S1 IDs from --validation-ids are present in the candidate artifact."
        )

    n = min(args.queries, len(artifact_validation_ids))
    chosen = rng.choice(
        np.asarray(artifact_validation_ids, dtype=object),
        size=n,
        replace=False,
    ).tolist()
    chosen = [str(x) for x in chosen]
    wanted = set(chosen)

    print(f"Validation queries: {len(chosen):,}", flush=True)
    print(
        f"Artifact-overlap validation queries: {len(artifact_validation_ids):,}",
        flush=True,
    )

    s1 = load_s1(args.s1, wanted)
    truth = load_truth(args.ground_truth, wanted)

    rows = artifact_lookup(artifact_s1_ids, wanted)

    needed = set()

    per_query_ids = {}

    for sid in chosen:
        row_idx = rows[sid]
        a = int(offsets[row_idx])
        b = int(offsets[row_idx + 1])

        ids = np.asarray(
            artifact_ids[a:min(b, a + args.max_rank)],
            dtype=np.uint32,
        )
        per_query_ids[sid] = ids
        needed.update(int(x) for x in ids.tolist())

    print(
        f"Unique candidate records needed: {len(needed):,}",
        flush=True,
    )

    candidate_records = load_candidate_rows(
        args.s2,
        args.s3,
        needed,
    )

    score_rows = []
    label_rows = []
    truth_counts = []

    for i, sid in enumerate(chosen, start=1):
        qraw = s1[sid]
        q = prepare_record(
            qraw["business_name"],
            qraw["business_address"],
            qraw["country"],
        )

        ids = per_query_ids[sid]
        scores_input = np.empty(
            (len(ids), len(FEATURE_NAMES)),
            dtype=np.float32,
        )
        labels = np.empty(len(ids), dtype=np.uint8)

        true = truth[sid]

        for j, gid in enumerate(ids.tolist()):
            entity_id, cand = candidate_records[int(gid)]

            scores_input[j] = pair_features_prepared(
                q,
                cand,
                blocker_rank=j + 1,
            )
            labels[j] = int(entity_id in true)

        scores = (
            model.predict_proba(scores_input)[:, 1]
            if len(scores_input)
            else np.empty(0, dtype=np.float32)
        )

        score_rows.append(
            np.asarray(scores, dtype=np.float32)
        )
        label_rows.append(labels)
        truth_counts.append(len(true))

        if i % 500 == 0:
            print(
                f"  scored {i:,}/{len(chosen):,}",
                flush=True,
            )

    # Search rank-k + threshold.
    # The previous evaluator bottomed out at 0.39, so this deliberately
    # searches much lower. The model was trained with negative_weight=8,
    # so probability calibration can be shifted downward.
    rank_grid = [k for k in (16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256) if k <= args.max_rank]
    threshold_grid = np.arange(
        0.05,
        0.951,
        0.005,
        dtype=np.float32,
    )

    best = None

    for k in rank_grid:
        for threshold in threshold_grid:
            for max_matches in (6, 8, 10, 12, 16):
                score, zero = evaluate(
                    score_rows,
                    label_rows,
                    truth_counts,
                    float(threshold),
                    max_matches,
                    0.0,
                    k,
                )

                key = (
                    score,
                    -zero,
                    -k,
                )

                if best is None or key > best[0]:
                    best = (
                        key,
                        {
                            "threshold": float(threshold),
                            "rank_k": int(k),
                            "max_matches": int(max_matches),
                            "relative_gate": 0.0,
                            "macro_f0.5": float(score),
                            "zero_outputs": int(zero),
                        },
                    )

    # Fine threshold pass around the best coarse point.
    coarse_t = best[1]["threshold"]
    fine_grid = np.arange(
        max(0.01, coarse_t - 0.01),
        min(0.999, coarse_t + 0.0101),
        0.0005,
        dtype=np.float32,
    )

    for threshold in fine_grid:
        for max_matches in (best[1]["max_matches"], 10, 12):
            for gate in (0.0, 0.90, 0.95):
                score, zero = evaluate(
                    score_rows,
                    label_rows,
                    truth_counts,
                    float(threshold),
                    int(max_matches),
                    float(gate),
                    int(best[1]["rank_k"]),
                )

                key = (
                    score,
                    -zero,
                    -int(best[1]["rank_k"]),
                )

                if key > best[0]:
                    best = (
                        key,
                        {
                            "threshold": float(threshold),
                            "rank_k": int(best[1]["rank_k"]),
                            "max_matches": int(max_matches),
                            "relative_gate": float(gate),
                            "macro_f0.5": float(score),
                            "zero_outputs": int(zero),
                        },
                    )

    result = best[1]
    result.update(
        {
            "queries": len(chosen),
            "max_rank_evaluated": args.max_rank,
            "seed": args.seed,
        }
    )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("=" * 60)
    print("FAST MATCHER EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Rank-k        : {result['rank_k']}")
    print(f"Threshold     : {result['threshold']:.4f}")
    print(f"Max matches   : {result['max_matches']}")
    print(f"Relative gate : {result['relative_gate']:.2f}")
    print(f"Macro F0.5    : {result['macro_f0.5']:.6f}")
    print(f"Zero outputs  : {result['zero_outputs']:,}")
    print(f"Time          : {time.perf_counter() - start:.1f}s")
    print(f"Output        : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
