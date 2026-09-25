import pandas as pd


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def load_source(path):
    """
    Load one challenge TSV and validate its basic schema.
    """

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    return df


def combine_candidate_sources(s2, s3):
    """
    Combine Source 2 and Source 3 into one candidate pool.
    """

    return pd.concat(
        [s2, s3],
        ignore_index=True
    )


def get_country_values(df):
    """
    Return all unique country values in the candidate pool.
    """

    return sorted(
        df["country"].dropna().unique()
    )


def partition_by_country(df):
    """
    Split the candidate pool into country buckets.

    Returns:
        dict[str, pd.DataFrame]
    """

    return {
        country: country_df
        for country, country_df in df.groupby(
            "country",
            sort=True
        )
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect country partitions for S2/S3."
    )

    parser.add_argument(
        "--source2",
        required=True,
    )

    parser.add_argument(
        "--source3",
        required=True,
    )

    args = parser.parse_args()

    # Load full S2 and S3
    s2 = load_source(args.source2)
    s3 = load_source(args.source3)

    # Combine them into one candidate pool
    candidate_pool = combine_candidate_sources(
        s2,
        s3
    )

    # Partition by country
    partitions = partition_by_country(
        candidate_pool
    )

    print(f"Total S2 records: {len(s2):,}")
    print(f"Total S3 records: {len(s3):,}")
    print(f"Total candidate records: {len(candidate_pool):,}")
    print()

    print("Country distribution:")

    for country, bucket in partitions.items():

        print(
            f"{country}: {len(bucket):,}"
        )