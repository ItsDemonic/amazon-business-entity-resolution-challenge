import argparse
from collections import defaultdict


def load_dev_ids(path):
    dev_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")

        try:
            id_idx = header.index("entity_id")
        except ValueError:
            id_idx = 0

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > id_idx:
                dev_ids.add(parts[id_idx])

    return dev_ids


def load_ground_truth(path, dev_ids):
    """
    Load ground-truth matches for only the 10k dev S1 IDs.

    Format:
        source1_entity_id    matched_entity_ids

    matched_entity_ids is comma-separated.
    """
    truth = {}

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")

        s1_idx = header.index("source1_entity_id")
        matched_idx = header.index("matched_entity_ids")

        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) <= max(s1_idx, matched_idx):
                continue

            s1_id = parts[s1_idx]

            if s1_id not in dev_ids:
                continue

            matched_raw = parts[matched_idx]

            if matched_raw:
                truth[s1_id] = {
                    entity_id.strip()
                    for entity_id in matched_raw.split(",")
                    if entity_id.strip()
                }
            else:
                truth[s1_id] = set()

    return truth


def read_candidate_row(f):
    """
    Read one candidate TSV row.

    Expected:
        source1_entity_id \t candidate_id1 \t candidate_id2 ...

    Returns:
        (s1_id, set(candidate_ids))
        or None at EOF
    """
    line = f.readline()

    if not line:
        return None

    parts = line.rstrip("\n").split("\t")

    if not parts:
        return None

    s1_id = parts[0]
    candidates = set()

    if len(parts) > 1 and parts[1]:
        candidates = {
            candidate.strip()
            for candidate in parts[1].split(",")
            if candidate.strip()
        }

    return s1_id, candidates


def evaluate_union(
    candidate_a_path,
    candidate_b_path,
    ground_truth_path,
    dev_s1_path,
):
    print("Loading 10k dev S1 IDs...")
    dev_ids = load_dev_ids(dev_s1_path)
    print(f"Dev S1 IDs: {len(dev_ids):,}")

    print("Loading only matching ground-truth rows...")
    truth = load_ground_truth(ground_truth_path, dev_ids)

    print(f"Ground-truth rows retained: {len(truth):,}")

    total_true_pairs = 0
    found_true_pairs = 0
    entities_with_matches = 0
    full_recall_entities = 0

    total_candidates = 0
    candidate_counts = []

    evaluated = 0

    with (
        open(candidate_a_path, "r", encoding="utf-8") as fa,
        open(candidate_b_path, "r", encoding="utf-8") as fb,
    ):
        # Skip headers if present.
        pos_a = fa.tell()
        first_a = fa.readline()

        if first_a:
            first_a_parts = first_a.rstrip("\n").split("\t")
            if first_a_parts[0] not in dev_ids:
                pass
            else:
                fa.seek(pos_a)

        pos_b = fb.tell()
        first_b = fb.readline()

        if first_b:
            first_b_parts = first_b.rstrip("\n").split("\t")
            if first_b_parts[0] not in dev_ids:
                pass
            else:
                fb.seek(pos_b)

        while True:
            row_a = read_candidate_row(fa)
            row_b = read_candidate_row(fb)

            if row_a is None and row_b is None:
                break

            if row_a is None or row_b is None:
                raise ValueError(
                    "Candidate files have different numbers of rows."
                )

            s1_a, candidates_a = row_a
            s1_b, candidates_b = row_b

            if s1_a != s1_b:
                raise ValueError(
                    f"Candidate files are not aligned:\n"
                    f"  File A: {s1_a}\n"
                    f"  File B: {s1_b}"
                )

            s1_id = s1_a

            if s1_id not in dev_ids:
                continue

            candidates = candidates_a | candidates_b

            true_candidates = truth.get(s1_id, set())

            total_candidates += len(candidates)
            candidate_counts.append(len(candidates))

            if true_candidates:
                entities_with_matches += 1
                total_true_pairs += len(true_candidates)

                found = len(candidates & true_candidates)
                found_true_pairs += found

                if found == len(true_candidates):
                    full_recall_entities += 1

            evaluated += 1

            if evaluated % 1000 == 0:
                print(f"  evaluated={evaluated:,}")

    if evaluated == 0:
        raise ValueError("No candidate rows were evaluated.")

    pairwise_recall = (
        found_true_pairs / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    full_recall = (
        full_recall_entities / entities_with_matches
        if entities_with_matches
        else 0.0
    )

    candidate_counts.sort()

    def percentile(values, p):
        if not values:
            return 0

        index = int(p * (len(values) - 1))
        return values[index]

    mean_candidates = total_candidates / evaluated

    print()
    print("=" * 70)
    print("STREAMING UNION BLOCKING RECALL REPORT")
    print("=" * 70)
    print(f"Queries evaluated:            {evaluated:,}")
    print(f"Entities with >=1 true match: {entities_with_matches:,}")
    print(f"Total true pairs:              {total_true_pairs:,}")
    print(f"Found true pairs:              {found_true_pairs:,}")
    print(f"Pairwise recall:               {pairwise_recall:.6f}")
    print(f"Full-recall rate:              {full_recall:.6f}")
    print()
    print("Union candidate count distribution:")
    print(f"  mean:   {mean_candidates:.2f}")
    print(f"  median: {percentile(candidate_counts, 0.50):,}")
    print(f"  p90:    {percentile(candidate_counts, 0.90):,}")
    print(f"  max:    {max(candidate_counts):,}")
    print()
    print("Input files:")
    print(f"  A: {candidate_a_path}")
    print(f"  B: {candidate_b_path}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-safe streaming evaluator for union of two blocker outputs."
    )

    parser.add_argument(
        "--candidates-a",
        required=True,
        help="First candidate TSV.",
    )

    parser.add_argument(
        "--candidates-b",
        required=True,
        help="Second candidate TSV.",
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Ground-truth TSV.",
    )

    parser.add_argument(
        "--dev-s1",
        required=True,
        help="10k dev S1 TSV.",
    )

    args = parser.parse_args()

    evaluate_union(
        candidate_a_path=args.candidates_a,
        candidate_b_path=args.candidates_b,
        ground_truth_path=args.ground_truth,
        dev_s1_path=args.dev_s1,
    )


if __name__ == "__main__":
    main()