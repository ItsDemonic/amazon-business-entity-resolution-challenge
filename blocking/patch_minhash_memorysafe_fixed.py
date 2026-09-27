
from pathlib import Path

p = Path("blocking/minhash_lsh.py")
s = p.read_text(encoding="utf-8")

start = s.index("def _rank_and_filter_candidates(")
end = s.index("\ndef query_internal_ids(", start)

new_func = '''
def _rank_and_filter_candidates(
    raw: np.ndarray,
    min_band_hits: int,
    max_candidates: int,
) -> np.ndarray:
    if raw.size == 0:
        return np.empty(0, dtype=np.uint64)

    if min_band_hits <= 1:
        # Same candidate set as np.unique(raw), but with lower peak RAM.
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
'''

s = s[:start] + new_func.lstrip("\n") + s[end:]

old = '''
        raw = np.concatenate(parts)
        results.append(
            _rank_and_filter_candidates(
                raw,
                min_band_hits,
                max_candidates,
            )
        )
'''

new = '''
        raw = np.concatenate(parts)
        per_query_parts[query_index] = []
        del parts

        results.append(
            _rank_and_filter_candidates(
                raw,
                min_band_hits,
                max_candidates,
            )
        )
'''

if old not in s:
    raise SystemExit("Could not find query filtering block. No changes made.")

s = s.replace(old, new, 1)

backup = p.with_suffix(".py.before_memorysafe_patch")
if not backup.exists():
    backup.write_text(
        p.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

p.write_text(s, encoding="utf-8")

print("Patched:", p)
print("Backup :", backup)
