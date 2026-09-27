from pathlib import Path

p = Path("blocking/minhash_lsh.py")
s = p.read_text(encoding="utf-8")

old = r"""def _rank_and_filter_candidates(
    raw: np.ndarray,
    min_band_hits: int,
    max_candidates: int,
) -> np.ndarray:
    """Apply the existing per-query candidate filtering rules."""

    if raw.size == 0:
        return np.empty(0, dtype=np.uint64)

    if min_band_hits <= 1:
        return np.unique(raw)

    candidates, counts = np.unique(
        raw,
        return_counts=True,
    )

    keep = counts >= min_band_hits
    candidates = candidates[keep]

    if max_candidates > 0 and candidates.size > max_candidates:
        candidate_counts = counts[keep]
        order = np.lexsort(
            (
                candidates,
                -candidate_counts,
            )
        )
        candidates = candidates[order[:max_candidates]]

    return candidates
"""

new = r"""def _rank_and_filter_candidates(
    raw: np.ndarray,
    min_band_hits: int,
    max_candidates: int,
) -> np.ndarray:
    """Memory-safe per-query candidate filtering."""

    if raw.size == 0:
        return np.empty(0, dtype=np.uint64)

    if min_band_hits <= 1:
        # Avoid np.unique's potentially huge temporary hash table.
        # In-place sort preserves the exact candidate set with much lower
        # peak RAM usage.
        raw.sort(kind="quicksort")
        keep = np.empty(raw.size, dtype=bool)
        keep[0] = True
        if raw.size > 1:
            keep[1:] = raw[1:] != raw[:-1]
        candidates = raw[keep]

        if max_candidates > 0 and candidates.size > max_candidates:
            candidates = candidates[:max_candidates]

        return candidates

    candidates, counts = np.unique(
        raw,
        return_counts=True,
    )

    keep = counts >= min_band_hits
    candidates = candidates[keep]

    if max_candidates > 0 and candidates.size > max_candidates:
        candidate_counts = counts[keep]
        order = np.lexsort(
            (
                candidates,
                -candidate_counts,
            )
        )
        candidates = candidates[order[:max_candidates]]

    return candidates
"""

if old not in s:
    raise SystemExit("Could not find _rank_and_filter_candidates() exactly; no changes made.")

s = s.replace(old, new)

old2 = r"""        raw = np.concatenate(parts)
        results.append(
            _rank_and_filter_candidates(
                raw,
                min_band_hits,
                max_candidates,
            )
        )
"""

new2 = r"""        raw = np.concatenate(parts)

        # Release the individual band arrays before sorting/filtering the
        # combined array.
        per_query_parts[query_index] = []
        del parts

        results.append(
            _rank_and_filter_candidates(
                raw,
                min_band_hits,
                max_candidates,
            )
        )
"""

if old2 not in s:
    raise SystemExit("Could not find query concat block exactly; no changes made.")

s = s.replace(old2, new2)

backup = p.with_suffix(".py.before_memorysafe_patch")
if not backup.exists():
    backup.write_text(Path(p).read_text(encoding="utf-8"), encoding="utf-8")

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
print(f"Backup:  {backup}")
