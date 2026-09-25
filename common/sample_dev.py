# common/sample_dev.py

import argparse
import pandas as pd


DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_SEED = 42


def create_dev_sample(
    source1_path,
    output_path,
    sample_size=DEFAULT_SAMPLE_SIZE,
    seed=DEFAULT_SEED,
):
    """
    Create a fixed S1 development sample.

    Only Source 1 is subsampled.
    Source 2 and Source 3 remain full-size.
    """

    df = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    if sample_size > len(df):
        raise ValueError(
            f"Requested {sample_size} rows, "
            f"but dataset only contains {len(df)} rows."
        )

    sample = df.sample(
        n=sample_size,
        random_state=seed,
    )

    # Sort so the generated file is deterministic/readable
    sample = sample.sort_values("entity_id")

    sample.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"Created dev sample: {len(sample):,} S1 entities")
    print(f"Seed: {seed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source1",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    create_dev_sample(
        source1_path=args.source1,
        output_path=args.output,
        sample_size=args.size,
        seed=args.seed,
    )