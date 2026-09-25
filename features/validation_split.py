import argparse
import json
from pathlib import Path

import numpy as np 
import pandas as pd

DEFAULT_VALIDATION_FRACTION = 0.20  
DEFAULT_SEED = 42

REQUIRED_S1_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]

REQUIRED_GT_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids"
]

def load_s1(path):
    
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    missing = set(REQUIRED_S1_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing)}"
        )
    return df

def load_ground_truth(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    missing = set(REQUIRED_GT_COLUMNS)- set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing)}"
        )
    return df

def get_match_count(matched_entity_ids):
    """
    Number of S2/S3 matches for an S1 entity
    """

    if not matched_entity_ids:
        return 0

    return len(
        [
            entity_id
            for entity_id in matched_entity_ids.split(",")
            if entity_id
        ]
    )

def get_source_coverage(matched_entity_ids):
    """
    Classify matches as:
        singleton
        S2 Only
        S3 Only
        both
    """

    if not matched_entity_ids:
        return "singleton"
    
    ids = [
        entity_id
        for entity_id in matched_entity_ids.split(",")
        if entity_id
    ]

    has_s2 = any(
        entity_id.startswith("S2-")
        for entity_id in ids
    )

    has_s3 = any(
        entity_id.startswith("S3-")
        for entity_id in ids
    )

    if has_s2 and has_s3:
        return "both"
    
    if has_s2:
        return "S2_only"
    
    if has_s3:
        return "S3_only"

    raise ValueError(
        f"Unexpected matched IDs: {matched_entity_ids}"
    )

def match_count_category(match_count):
    """
    Group match counts into stable categories.
    """

    if match_count == 0:
        return "singleton"

    if match_count == 1:
        return "one_match"

    if match_count <= 3:
        return "small_multi"

    if match_count <= 6:
        return "medium_multi"

    return "high_multi"


def add_ground_truth_metadata(s1, ground_truth):
    """
    Join ground-truth-derived metadata onto S1.
    """

    if len(ground_truth) != len(s1):
        raise ValueError(
            "Ground truth and Source 1 row counts differ."
        )

    df = s1.merge(
        ground_truth,
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
        validate="one_to_one",
    )

    if df["source1_entity_id"].isna().any():
        missing = df["source1_entity_id"].isna().sum()

        raise ValueError(
            f"{missing:,} S1 entities have no ground-truth row."
        )

    df["match_count"] = df[
        "matched_entity_ids"
    ].apply(get_match_count)

    df["source_coverage"] = df[
        "matched_entity_ids"
    ].apply(get_source_coverage)

    df["match_category"] = df[
        "match_count"
    ].apply(match_count_category)

    return df


def get_test_country_distribution(test_s1):
    """
    Get test country distribution.

    France is present in test but has no labeled
    training examples, so the returned validation
    target only contains countries available in training.
    """

    test_counts = test_s1["country"].value_counts()

    train_countries = set(
        ["US", "India"]
    )

    available = test_counts[
        test_counts.index.isin(train_countries)
    ]

    total_available = available.sum()

    if total_available == 0:
        raise ValueError(
            "No overlapping labeled countries "
            "between training and test."
        )

    return (
        available / total_available
    ).to_dict()


def create_validation_split(
    metadata,
    validation_fraction,
    test_country_distribution,
    seed,
):
    """
    Create a deterministic validation split.

    Stratification dimensions:
        country
        match-count category
        source coverage

    Country proportions are adjusted to match the
    test distribution among countries that have
    training labels.
    """

    rng = np.random.default_rng(seed)

    metadata = metadata.copy()

    metadata["split"] = "train"

    validation_parts = []

    # Process each labeled country separately.
    for country, country_target in (
        test_country_distribution.items()
    ):

        country_df = metadata[
            metadata["country"] == country
        ].copy()

        if len(country_df) == 0:
            continue

        target_validation_count = int(
            round(
                len(metadata)
                * validation_fraction
                * country_target
            )
        )

        # Stratify inside the country by match behavior.
        grouped = country_df.groupby(
            [
                "match_category",
                "source_coverage",
            ],
            sort=True,
        )

        group_sizes = grouped.size()

        # Allocate validation rows proportionally.
        raw_targets = (
            group_sizes
            / len(country_df)
            * target_validation_count
        )

        group_targets = np.floor(
            raw_targets
        ).astype(int)

        # Distribute rounding remainder.
        remainder = (
            target_validation_count
            - group_targets.sum()
        )

        if remainder > 0:

            fractions = (
                raw_targets
                - np.floor(raw_targets)
            )

            for group_key in (
                fractions
                .sort_values(ascending=False)
                .index[:remainder]
            ):
                group_targets.loc[group_key] += 1

        for group_key, group in grouped:

            n = group_targets.loc[group_key]

            if n == 0:
                continue

            selected_indices = rng.choice(
                group.index.to_numpy(),
                size=min(n, len(group)),
                replace=False,
            )

            validation_parts.append(
                metadata.loc[selected_indices]
            )

    if not validation_parts:
        raise ValueError(
            "Validation split is empty."
        )

    validation_df = pd.concat(
        validation_parts,
        ignore_index=False,
    )

    validation_indices = set(
        validation_df.index
    )

    metadata.loc[
        metadata.index.isin(validation_indices),
        "split",
    ] = "validation"

    train_df = metadata[
        metadata["split"] == "train"
    ].copy()

    validation_df = metadata[
        metadata["split"] == "validation"
    ].copy()

    return train_df, validation_df


def save_split(
    train_df,
    validation_df,
    output_dir,
):
    """
    Save split IDs and metadata.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df[
        ["entity_id"]
    ].to_csv(
        output_dir / "train_s1_ids.tsv",
        sep="\t",
        index=False,
    )

    validation_df[
        ["entity_id"]
    ].to_csv(
        output_dir / "validation_s1_ids.tsv",
        sep="\t",
        index=False,
    )

    metadata_columns = [
        "entity_id",
        "country",
        "match_count",
        "match_category",
        "source_coverage",
        "split",
    ]

    pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )[metadata_columns].to_csv(
        output_dir / "split_metadata.tsv",
        sep="\t",
        index=False,
    )


def print_summary(
    train_df,
    validation_df,
    test_country_distribution,
):
    print("\n=== SPLIT SUMMARY ===")

    print(
        f"Total S1: "
        f"{len(train_df) + len(validation_df):,}"
    )

    print(
        f"Training: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation: "
        f"{len(validation_df):,}"
    )

    print("\nTest-like country target:")
    for country, proportion in (
        test_country_distribution.items()
    ):
        print(
            f"  {country}: "
            f"{proportion * 100:.2f}%"
        )

    print("\nActual validation country distribution:")
    print(
        validation_df["country"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nValidation match categories:")
    print(
        validation_df["match_category"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nValidation source coverage:")
    print(
        validation_df["source_coverage"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-s1",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--test-s1",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="features/validation_output",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    # Load data.
    train_s1 = load_s1(
        args.train_s1
    )

    ground_truth = load_ground_truth(
        args.ground_truth
    )

    test_s1 = load_s1(
        args.test_s1
    )

    # Add ground-truth-derived metadata.
    metadata = add_ground_truth_metadata(
        train_s1,
        ground_truth,
    )

    # Determine target country distribution
    # from the test set.
    test_country_distribution = (
        get_test_country_distribution(
            test_s1
        )
    )

    # Create deterministic split.
    train_df, validation_df = (
        create_validation_split(
            metadata,
            args.validation_fraction,
            test_country_distribution,
            args.seed,
        )
    )

    # Save results.
    save_split(
        train_df,
        validation_df,
        args.output_dir,
    )

    print_summary(
        train_df,
        validation_df,
        test_country_distribution,
    )


if __name__ == "__main__":
    main()