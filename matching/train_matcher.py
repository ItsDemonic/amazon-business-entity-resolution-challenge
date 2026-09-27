"""
Train the final pairwise matcher on hard negatives.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from features.similarity_features import FEATURE_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="matching/matcher_train.npz")
    p.add_argument("--output", default="matching/matcher_model.joblib")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-weight", type=float, default=8.0)
    p.add_argument("--learning-rate", type=float, default=0.06)
    p.add_argument("--max-iter", type=int, default=260)
    p.add_argument("--max-leaf-nodes", type=int, default=63)
    p.add_argument("--min-samples-leaf", type=int, default=40)
    args = p.parse_args()

    start = time.perf_counter()

    data = np.load(args.input, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.uint8)
    names = [str(x) for x in data["feature_names"].tolist()]

    if names != FEATURE_NAMES:
        raise ValueError(
            "Feature schema mismatch.\n"
            f"Expected {FEATURE_NAMES}\n"
            f"Got      {names}"
        )

    weights = np.ones(len(y), dtype=np.float32)
    weights[y == 0] = np.float32(args.negative_weight)

    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=1.5,
        early_stopping=False,
        random_state=args.seed,
    )

    print(f"Pairs          : {len(y):,}", flush=True)
    print(f"Positives      : {int(y.sum()):,}", flush=True)
    print(f"Negatives      : {int((y == 0).sum()):,}", flush=True)
    print(f"Features       : {len(FEATURE_NAMES)}", flush=True)
    print("Training...", flush=True)

    model.fit(X, y, sample_weight=weights)

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "metadata": {
                "seed": args.seed,
                "negative_weight": args.negative_weight,
                "pairs": int(len(y)),
            },
        },
        args.output,
        compress=3,
    )

    print()
    print("=" * 60)
    print("FINAL MATCHER TRAINED")
    print("=" * 60)
    print(f"Model : {args.output}")
    print(f"Time  : {time.perf_counter() - start:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
