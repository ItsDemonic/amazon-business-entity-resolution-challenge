# common/io_schema.py

import pandas as pd


CANDIDATE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_ids",
]


def write_candidate_pairs(rows, output_path):
    """
    Write candidate pairs in the official challenge schema.

    rows:
        iterable of:
        (source1_entity_id, iterable_of_candidate_ids)
    """

    output_rows = []

    for source1_id, candidate_ids in rows:
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []

        for candidate_id in candidate_ids:
            if candidate_id not in seen:
                seen.add(candidate_id)
                unique_ids.append(candidate_id)

        output_rows.append({
            "source1_entity_id": source1_id,
            "candidate_entity_ids": ",".join(unique_ids),
        })

    df = pd.DataFrame(output_rows, columns=CANDIDATE_COLUMNS)

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )


def read_candidate_pairs(path):
    """
    Read a candidate-pair TSV.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )