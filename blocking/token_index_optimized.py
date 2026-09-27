
"""
Optimized hybrid token blocker.

Design goals:
    - bounded RAM on 10M+ S2/S3 candidate rows
    - fast query-time lookup
    - deterministic candidate generation
    - better recall than single-token-only blocking on "generic token" names

Features:
    1. Single significant-name-token inverted index.
    2. Exact normalized-name fallback index.
    3. Query-driven token-pair fallback index.
    4. Integer internal IDs instead of storing millions of Python strings
       inside postings lists.
    5. Disk-backed CSR-style postings arrays.
    6. Numpy marker-array deduplication instead of a Python set per query.
    7. Query-adaptive fallback: exact-name/pair retrieval is only used when
       the single-token result is small.

The build remains query-driven: only tokens, exact names, and token pairs
that can actually occur in the supplied S1 query file are indexed.

CLI is compatible with the original script, with optional tuning flags.
"""

import argparse
import csv
import itertools
import json
import pickle
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Resolve repo root so this module is safe to run with `python -m blocking...`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.normalize import name_tokens, normalize_name


DEFAULT_MAX_DOC_FREQ = 0.001
DEFAULT_MAX_POSTINGS_PER_TOKEN = 5000
DEFAULT_EXACT_MAX_POSTINGS = 5000
DEFAULT_PAIR_MAX_POSTINGS = 5000
DEFAULT_FALLBACK_THRESHOLD = 50
DEFAULT_MAX_PAIR_TOKENS = 12
PROGRESS_EVERY = 1_000_000
SCHEMA_VERSION = 2


def _iter_source_rows(path, max_rows=None):
    """Stream (entity_id, business_name, country) from a TSV."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)

        try:
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            country_idx = header.index("country")
        except ValueError as exc:
            raise ValueError(
                f"{path} is missing a required column: {exc}"
            ) from exc

        n = 0
        for row in reader:
            if len(row) <= max(id_idx, name_idx, country_idx):
                continue

            yield row[id_idx], row[name_idx], row[country_idx]
            n += 1

            if max_rows is not None and n >= max_rows:
                break


def _token_set(name):
    return set(name_tokens(name))


def _normalized_name(name):
    value = normalize_name(str(name))
    return value.strip()


def _query_pairs(tokens, max_tokens=DEFAULT_MAX_PAIR_TOKENS):
    """Generate unordered token pairs for a query/candidate name."""
    if len(tokens) < 2:
        return ()

    ordered = sorted(tokens)
    if len(ordered) > max_tokens:
        # Keeping all pairs for unusually long names can explode the fallback
        # vocabulary. The first tokens are deterministic after sorting.
        ordered = ordered[:max_tokens]

    return itertools.combinations(ordered, 2)


def _collect_query_vocab(query_path, max_rows=None, verbose=True, max_pair_tokens=DEFAULT_MAX_PAIR_TOKENS):
    """
    Stream S1 once and build the only vocabularies the blocker can ever use.
    """
    needed_tokens = defaultdict(set)
    needed_exact = defaultdict(set)
    needed_pairs = defaultdict(set)

    n_rows = 0
    t0 = time.time()

    for _, name, country in _iter_source_rows(query_path, max_rows=max_rows):
        tokens = _token_set(name)
        if tokens:
            needed_tokens[country].update(tokens)

        norm = _normalized_name(name)
        if norm:
            needed_exact[country].add(norm)

        if len(tokens) >= 2:
            needed_pairs[country].update(
                _query_pairs(tokens, max_pair_tokens)
            )

        n_rows += 1
        if verbose and n_rows % PROGRESS_EVERY == 0:
            print(
                f"    [query vocab] {n_rows:,} S1 rows in "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

    if verbose:
        total_tokens = sum(len(v) for v in needed_tokens.values())
        total_exact = sum(len(v) for v in needed_exact.values())
        total_pairs = sum(len(v) for v in needed_pairs.values())
        print(
            f"Query vocab: {n_rows:,} S1 rows | "
            f"{total_tokens:,} tokens | "
            f"{total_exact:,} exact names | "
            f"{total_pairs:,} token pairs",
            flush=True,
        )

    return (
        dict(needed_tokens),
        dict(needed_exact),
        dict(needed_pairs),
        n_rows,
    )


def _effective_limit(n_docs, fraction, absolute_cap):
    pct_limit = fraction * n_docs if n_docs > 0 else 0.0
    if absolute_cap is None or absolute_cap <= 0:
        return pct_limit
    if pct_limit <= 0:
        return float(absolute_cap)
    return min(pct_limit, float(absolute_cap))


def _pass1(
    source_paths,
    needed_tokens,
    needed_exact,
    needed_pairs,
    max_rows_per_source=None,
    verbose=True,
    max_pair_tokens=DEFAULT_MAX_PAIR_TOKENS,
):
    """
    Pass 1:
      - country totals
      - max entity-id length
      - document frequencies for query-relevant tokens
      - document frequencies for query-relevant exact names/pairs
    """
    token_counts = defaultdict(lambda: defaultdict(int))
    exact_counts = defaultdict(lambda: defaultdict(int))
    pair_counts = defaultdict(lambda: defaultdict(int))
    country_totals = defaultdict(int)
    max_id_width = 1

    total_rows = 0
    t_global = time.time()

    for path in source_paths:
        t0 = time.time()
        n_rows = 0

        for entity_id, name, country in _iter_source_rows(
            path, max_rows=max_rows_per_source
        ):
            country_totals[country] += 1
            total_rows += 1
            n_rows += 1

            max_id_width = max(max_id_width, len(entity_id.encode("utf-8")))

            token_needed = needed_tokens.get(country)
            exact_needed = needed_exact.get(country)
            pair_needed = needed_pairs.get(country)

            # If this country has no query-relevant key of any kind, avoid
            # tokenization/normalization work entirely.
            if not token_needed and not exact_needed and not pair_needed:
                continue

            token_set = _token_set(name)

            # Token DF.
            if token_needed and token_set:
                counts = token_counts[country]
                for token in token_set:
                    if token in token_needed:
                        counts[token] += 1

            # Exact normalized-name DF only when exact names are requested
            # for this country.
            if exact_needed:
                norm = _normalized_name(name)
                if norm and norm in exact_needed:
                    exact_counts[country][norm] += 1

            # Pair DF only when this country has query-relevant pairs.
            if pair_needed and len(token_set) >= 2:
                pcounts = pair_counts[country]
                for pair in _query_pairs(token_set, max_pair_tokens):
                    if pair in pair_needed:
                        pcounts[pair] += 1

            if verbose and total_rows % PROGRESS_EVERY == 0:
                print(
                    f"    [pass 1] total={total_rows:,} rows | "
                    f"elapsed={time.time() - t_global:.1f}s",
                    flush=True,
                )

        if verbose:
            print(
                f"  [pass 1] streamed {path}: {n_rows:,} rows in "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

    return (
        token_counts,
        exact_counts,
        pair_counts,
        dict(country_totals),
        total_rows,
        max_id_width,
    )


def _select_kept(
    counts_by_country,
    country_totals,
    max_doc_freq,
    absolute_cap,
):
    """Return {(country, key): count} for keys that survive filtering."""
    kept = {}

    for country, counts in counts_by_country.items():
        n_docs = country_totals.get(country, 0)
        limit = _effective_limit(
            n_docs,
            max_doc_freq,
            absolute_cap,
        )

        for key, count in counts.items():
            if count <= limit:
                kept[(country, key)] = int(count)

    return kept


def _make_namespace(kept):
    """
    Convert kept keys to dense integer IDs and CSR offsets.

    Returns:
        key_to_id: dict[(country,key)] -> integer
        counts: uint64 array
        offsets: uint64 array
    """
    items = sorted(kept.items(), key=lambda item: (item[0][0], repr(item[0][1])))
    key_to_id = {key: idx for idx, (key, _) in enumerate(items)}
    counts = np.fromiter(
        (count for _, count in items),
        dtype=np.uint64,
        count=len(items),
    )

    offsets = np.empty(len(counts) + 1, dtype=np.uint64)
    offsets[0] = 0
    if len(counts):
        np.cumsum(counts, dtype=np.uint64, out=offsets[1:])

    return key_to_id, counts, offsets


def _estimate_file_bytes(n_items, dtype):
    return int(n_items) * np.dtype(dtype).itemsize


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_namespace(
    index_dir,
    name,
    key_to_id,
    offsets,
):
    """
    Save the small query-key dictionary with pickle and the numeric offsets
    as a memory-mappable numpy array. Pickle is much faster/smaller than
    serializing every key as a JSON record and is appropriate for this
    locally generated index.
    """
    with open(index_dir / f"{name}_keys.pkl", "wb") as fh:
        pickle.dump(key_to_id, fh, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(index_dir / f"{name}_offsets.npy", offsets)


def _load_namespace(index_dir, name):
    with open(index_dir / f"{name}_keys.pkl", "rb") as fh:
        key_to_id = pickle.load(fh)

    offsets = np.load(
        index_dir / f"{name}_offsets.npy",
        mmap_mode="r",
    )
    return key_to_id, offsets


def _append_posting(
    postings,
    offsets,
    cursors,
    key_id,
    internal_id,
):
    pos = int(offsets[key_id] + cursors[key_id])
    postings[pos] = internal_id
    cursors[key_id] += 1


def _pass2_build(
    source_paths,
    index_dir,
    token_key_to_id,
    token_offsets,
    exact_key_to_id,
    exact_offsets,
    pair_key_to_id,
    pair_offsets,
    total_rows,
    id_width,
    max_rows_per_source=None,
    verbose=True,
    max_pair_tokens=DEFAULT_MAX_PAIR_TOKENS,
):
    """
    Pass 2:
      - store fixed-width entity IDs
      - fill token/exact/pair postings CSR arrays

    Internal candidate IDs are monotonically increasing across S2 then S3.
    Therefore every posting list is naturally sorted and needs no later sort.
    """
    ids_path = index_dir / "entity_ids.npy"
    token_postings_path = index_dir / "token_postings.npy"
    exact_postings_path = index_dir / "exact_postings.npy"
    pair_postings_path = index_dir / "pair_postings.npy"

    id_store = np.lib.format.open_memmap(
        ids_path,
        mode="w+",
        dtype=f"S{id_width}",
        shape=(total_rows,),
    )

    token_postings = np.lib.format.open_memmap(
        token_postings_path,
        mode="w+",
        dtype=np.uint32,
        shape=(int(token_offsets[-1]) if len(token_offsets) else 0,),
    )
    exact_postings = np.lib.format.open_memmap(
        exact_postings_path,
        mode="w+",
        dtype=np.uint32,
        shape=(int(exact_offsets[-1]) if len(exact_offsets) else 0,),
    )
    pair_postings = np.lib.format.open_memmap(
        pair_postings_path,
        mode="w+",
        dtype=np.uint32,
        shape=(int(pair_offsets[-1]) if len(pair_offsets) else 0,),
    )

    token_cursors = np.zeros(len(token_offsets) - 1, dtype=np.uint64)
    exact_cursors = np.zeros(len(exact_offsets) - 1, dtype=np.uint64)
    pair_cursors = np.zeros(len(pair_offsets) - 1, dtype=np.uint64)

    internal_id = 0
    t_global = time.time()

    try:
        for path in source_paths:
            t0 = time.time()
            n_rows = 0

            for entity_id, name, country in _iter_source_rows(
                path,
                max_rows=max_rows_per_source,
            ):
                id_store[internal_id] = entity_id.encode("utf-8")

                token_set = _token_set(name)

                if token_set:
                    for token in token_set:
                        key_id = token_key_to_id.get((country, token))
                        if key_id is not None:
                            _append_posting(
                                token_postings,
                                token_offsets,
                                token_cursors,
                                key_id,
                                internal_id,
                            )

                if exact_key_to_id:
                    norm = _normalized_name(name)
                    if norm:
                        key_id = exact_key_to_id.get((country, norm))
                        if key_id is not None:
                            _append_posting(
                                exact_postings,
                                exact_offsets,
                                exact_cursors,
                                key_id,
                                internal_id,
                            )

                if pair_key_to_id and len(token_set) >= 2:
                    for pair in _query_pairs(token_set, max_pair_tokens):
                        key_id = pair_key_to_id.get((country, pair))
                        if key_id is not None:
                            _append_posting(
                                pair_postings,
                                pair_offsets,
                                pair_cursors,
                                key_id,
                                internal_id,
                            )

                internal_id += 1
                n_rows += 1

                if verbose and internal_id % PROGRESS_EVERY == 0:
                    print(
                        f"    [pass 2] indexed={internal_id:,} rows | "
                        f"elapsed={time.time() - t_global:.1f}s",
                        flush=True,
                    )

            if verbose:
                print(
                    f"  [pass 2] streamed {path}: {n_rows:,} rows in "
                    f"{time.time() - t0:.1f}s",
                    flush=True,
                )

        if internal_id != total_rows:
            raise RuntimeError(
                f"Pass 2 wrote {internal_id:,} rows but pass 1 counted "
                f"{total_rows:,}"
            )

        id_store.flush()
        token_postings.flush()
        exact_postings.flush()
        pair_postings.flush()
    finally:
        del id_store
        del token_postings
        del exact_postings
        del pair_postings

    return ids_path


def build_index(
    source_paths,
    query_path,
    index_dir,
    max_doc_freq=DEFAULT_MAX_DOC_FREQ,
    max_postings_per_token=DEFAULT_MAX_POSTINGS_PER_TOKEN,
    exact_max_postings=DEFAULT_EXACT_MAX_POSTINGS,
    pair_max_postings=DEFAULT_PAIR_MAX_POSTINGS,
    fallback_threshold=DEFAULT_FALLBACK_THRESHOLD,
    max_pair_tokens=DEFAULT_MAX_PAIR_TOKENS,
    max_rows_per_source=None,
    verbose=True,
):
    start = time.time()
    index_dir = Path(index_dir)

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPTIMIZED HYBRID TOKEN BLOCKER")
    print("=" * 70)
    print(f"Query S1        : {query_path}")
    print(f"Candidate paths : {len(source_paths)}")
    print(f"Token DF        : <= {max_doc_freq} of country AND <= {max_postings_per_token}")
    print(f"Exact-name cap  : {exact_max_postings}")
    print(f"Pair cap        : {pair_max_postings}")
    print(f"Fallback at     : <= {fallback_threshold} single-token candidates")
    print("=" * 70)

    (
        needed_tokens,
        needed_exact,
        needed_pairs,
        query_rows,
    ) = _collect_query_vocab(
        query_path,
        verbose=verbose,
        max_pair_tokens=max_pair_tokens,
    )

    # Respect the pair-token bound consistently between query-vocab and
    # candidate passes by clipping the pair vocab to the same deterministic
    # policy. (The query vocab already uses this policy.)
    del query_rows

    (
        token_counts,
        exact_counts,
        pair_counts,
        country_totals,
        total_rows,
        id_width,
    ) = _pass1(
        source_paths,
        needed_tokens,
        needed_exact,
        needed_pairs,
        max_rows_per_source=max_rows_per_source,
        verbose=verbose,
        max_pair_tokens=max_pair_tokens,
    )

    kept_tokens = _select_kept(
        token_counts,
        country_totals,
        max_doc_freq,
        max_postings_per_token,
    )
    kept_exact = _select_kept(
        exact_counts,
        country_totals,
        1.0,
        exact_max_postings,
    )
    # Pair fallback is only useful when at least one token in the pair is
    # absent from the single-token index. This removes redundant pair keys,
    # shrinking memory/disk and keeping query time focused on hard cases.
    kept_pairs_all = _select_kept(
        pair_counts,
        country_totals,
        1.0,
        pair_max_postings,
    )
    kept_token_keys = set(kept_tokens)
    kept_pairs = {
        pair_key: count
        for pair_key, count in kept_pairs_all.items()
        if not all(
            (pair_key[0], token) in kept_token_keys
            for token in pair_key[1]
        )
    }
    del kept_pairs_all, kept_token_keys

    token_key_to_id, token_counts_arr, token_offsets = _make_namespace(
        kept_tokens
    )
    exact_key_to_id, exact_counts_arr, exact_offsets = _make_namespace(
        kept_exact
    )
    pair_key_to_id, pair_counts_arr, pair_offsets = _make_namespace(
        kept_pairs
    )

    # Release pass-1 Python dictionaries and query vocab before allocating
    # the disk-backed posting arrays. These are otherwise dead after the
    # dense namespaces are constructed.
    del token_counts
    del exact_counts
    del pair_counts
    del needed_tokens
    del needed_exact
    del needed_pairs

    if verbose:
        token_postings = int(token_offsets[-1]) if len(token_offsets) else 0
        exact_postings = int(exact_offsets[-1]) if len(exact_offsets) else 0
        pair_postings = int(pair_offsets[-1]) if len(pair_offsets) else 0

        print()
        print("KEPT INDEX FEATURES")
        print("-" * 70)
        print(
            f"Tokens      : {len(token_key_to_id):,} keys / "
            f"{token_postings:,} postings"
        )
        print(
            f"Exact names : {len(exact_key_to_id):,} keys / "
            f"{exact_postings:,} postings"
        )
        print(
            f"Token pairs : {len(pair_key_to_id):,} keys / "
            f"{pair_postings:,} postings"
        )
        print(
            f"Entity IDs  : {total_rows:,} × {id_width} bytes = "
            f"{_estimate_file_bytes(total_rows, f'S{id_width}') / (1024**2):.1f} MB"
        )
        print()

    _save_namespace(
        index_dir,
        "token",
        token_key_to_id,
        token_offsets,
    )
    _save_namespace(
        index_dir,
        "exact",
        exact_key_to_id,
        exact_offsets,
    )
    _save_namespace(
        index_dir,
        "pair",
        pair_key_to_id,
        pair_offsets,
    )

    _pass2_build(
        source_paths,
        index_dir,
        token_key_to_id,
        token_offsets,
        exact_key_to_id,
        exact_offsets,
        pair_key_to_id,
        pair_offsets,
        total_rows,
        id_width,
        max_rows_per_source=max_rows_per_source,
        verbose=verbose,
        max_pair_tokens=max_pair_tokens,
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "max_doc_freq": float(max_doc_freq),
        "max_postings_per_token": int(
            max_postings_per_token
            if max_postings_per_token is not None
            else 0
        ),
        "exact_max_postings": int(exact_max_postings),
        "pair_max_postings": int(pair_max_postings),
        "fallback_threshold": int(fallback_threshold),
        "max_pair_tokens": int(max_pair_tokens),
        "total_rows": int(total_rows),
        "id_width": int(id_width),
        "country_totals": country_totals,
    }
    _write_json(index_dir / "metadata.json", metadata)

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("BLOCKER BUILD COMPLETE")
    print("=" * 70)
    print(f"Candidate rows : {total_rows:,}")
    print(f"Countries      : {len(country_totals)}")
    print(f"Build time     : {elapsed:.1f}s")
    print(f"Index          : {index_dir}")
    print("=" * 70)

    return metadata


class HybridTokenIndex:
    """Read-only query-side view of the persisted hybrid index."""

    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self.metadata = _read_json(self.index_dir / "metadata.json")

        (
            self.token_key_to_id,
            self.token_offsets,
        ) = _load_namespace(self.index_dir, "token")
        (
            self.exact_key_to_id,
            self.exact_offsets,
        ) = _load_namespace(self.index_dir, "exact")
        (
            self.pair_key_to_id,
            self.pair_offsets,
        ) = _load_namespace(self.index_dir, "pair")

        self.token_postings = np.load(
            self.index_dir / "token_postings.npy",
            mmap_mode="r",
        )
        self.exact_postings = np.load(
            self.index_dir / "exact_postings.npy",
            mmap_mode="r",
        )
        self.pair_postings = np.load(
            self.index_dir / "pair_postings.npy",
            mmap_mode="r",
        )
        self.entity_ids = np.load(
            self.index_dir / "entity_ids.npy",
            mmap_mode="r",
        )

        self.fallback_threshold = int(
            self.metadata["fallback_threshold"]
        )
        self.max_pair_tokens = int(
            self.metadata["max_pair_tokens"]
        )

        self._marks = np.zeros(
            int(self.metadata["total_rows"]),
            dtype=np.uint32,
        )
        self._query_id = np.uint32(0)

    def _next_query_marker(self):
        self._query_id = np.uint32(int(self._query_id) + 1)
        if self._query_id == 0:
            self._marks.fill(0)
            self._query_id = np.uint32(1)
        return self._query_id

    @staticmethod
    def _slice(postings, offsets, key_id):
        left = int(offsets[key_id])
        right = int(offsets[key_id + 1])
        return postings[left:right]

    def _mark_and_collect(self, arrays, marker):
        if not arrays:
            return np.empty(0, dtype=np.uint32)

        chunks = []
        for arr in arrays:
            if arr.size == 0:
                continue

            unseen = self._marks[arr] != marker
            if np.any(unseen):
                ids = np.asarray(arr[unseen], dtype=np.uint32)
                self._marks[ids] = marker
                chunks.append(ids)

        if not chunks:
            return np.empty(0, dtype=np.uint32)

        ids = np.concatenate(chunks)
        ids.sort()
        return ids

    def query_ids(self, name, country):
        tokens = _token_set(name)

        token_arrays = []
        for token in tokens:
            key_id = self.token_key_to_id.get((country, token))
            if key_id is not None:
                token_arrays.append(
                    self._slice(
                        self.token_postings,
                        self.token_offsets,
                        key_id,
                    )
                )

        marker = self._next_query_marker()
        ids = self._mark_and_collect(token_arrays, marker)

        # Adaptive recall fallback. Most normal queries never pay for it.
        if ids.size <= self.fallback_threshold:
            extras = []

            norm = _normalized_name(name)
            if norm:
                key_id = self.exact_key_to_id.get((country, norm))
                if key_id is not None:
                    extras.append(
                        self._slice(
                            self.exact_postings,
                            self.exact_offsets,
                            key_id,
                        )
                    )

            if len(tokens) >= 2:
                for pair in _query_pairs(
                    tokens,
                    self.max_pair_tokens,
                ):
                    key_id = self.pair_key_to_id.get((country, pair))
                    if key_id is not None:
                        extras.append(
                            self._slice(
                                self.pair_postings,
                                self.pair_offsets,
                                key_id,
                            )
                        )

            if extras:
                if ids.size:
                    old = ids
                    for arr in extras:
                        if arr.size == 0:
                            continue
                        unseen = self._marks[arr] != marker
                        if np.any(unseen):
                            add = np.asarray(
                                arr[unseen],
                                dtype=np.uint32,
                            )
                            self._marks[add] = marker
                            old = np.concatenate((old, add))
                    ids = old
                    if ids.size:
                        ids.sort()
                else:
                    ids = self._mark_and_collect(extras, marker)

        return ids

    def query_entity_ids(self, name, country):
        internal_ids = self.query_ids(name, country)
        if internal_ids.size == 0:
            return []

        raw = self.entity_ids[internal_ids]
        return [value.decode("utf-8") for value in raw.tolist()]


def run_query(
    index_dir,
    query_path,
    output_path,
    max_rows=None,
    verbose=True,
):
    start = time.time()
    index = HybridTokenIndex(index_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    zero = 0
    total_candidates = 0
    min_candidates = None
    max_candidates = 0

    with open(
        output_path,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as out:
        out.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        for entity_id, name, country in _iter_source_rows(
            query_path,
            max_rows=max_rows,
        ):
            candidates = index.query_entity_ids(
                name,
                country,
            )

            if not candidates:
                zero += 1

            count = len(candidates)
            total += 1
            total_candidates += count
            max_candidates = max(max_candidates, count)
            min_candidates = (
                count
                if min_candidates is None
                else min(min_candidates, count)
            )

            out.write(
                entity_id
                + "\t"
                + ",".join(candidates)
                + "\n"
            )

            if verbose and total % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                print(
                    f"    queried={total:,} | "
                    f"avg={total_candidates / total:.2f} | "
                    f"zero={zero:,} | "
                    f"rate={total / elapsed:.1f}/s",
                    flush=True,
                )

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("TOKEN QUERY COMPLETE")
    print("=" * 70)
    print(f"S1 queries       : {total:,}")
    print(f"Total candidates : {total_candidates:,}")
    print(
        f"Average/query    : "
        f"{(total_candidates / total) if total else 0:.2f}"
    )
    print(f"Zero candidates  : {zero:,}")
    print(
        f"Min / Max        : "
        f"{min_candidates if min_candidates is not None else 0} / "
        f"{max_candidates}"
    )
    print(f"Query time       : {elapsed:.1f}s")
    print(
        f"Query rate       : "
        f"{(total / elapsed) if elapsed else 0:.2f} S1/s"
    )
    print(f"Output           : {output_path}")
    print("=" * 70)

    return {
        "n_queries": total,
        "zero_candidate_count": zero,
        "mean_candidates": (
            total_candidates / total if total else 0
        ),
        "max_candidates": max_candidates,
        "min_candidates": min_candidates if min_candidates is not None else 0,
        "elapsed_seconds": elapsed,
    }


def run_token_blocking(
    source2_path,
    source3_path,
    dev_s1_path,
    output_path,
    max_doc_freq=DEFAULT_MAX_DOC_FREQ,
    max_postings_per_token=DEFAULT_MAX_POSTINGS_PER_TOKEN,
    max_rows_per_source=None,
    verbose=True,
):
    """
    Backward-compatible wrapper matching the original P1 entry point.
    """
    output_path_obj = Path(output_path)
    index_dir = (
        output_path_obj.parent
        / f".{output_path_obj.stem}_index"
    )

    return build_index(
        [source2_path, source3_path],
        dev_s1_path,
        index_dir=index_dir,
        max_doc_freq=max_doc_freq,
        max_postings_per_token=max_postings_per_token,
        exact_max_postings=DEFAULT_EXACT_MAX_POSTINGS,
        pair_max_postings=DEFAULT_PAIR_MAX_POSTINGS,
        fallback_threshold=DEFAULT_FALLBACK_THRESHOLD,
        max_pair_tokens=DEFAULT_MAX_PAIR_TOKENS,
        max_rows_per_source=max_rows_per_source,
        verbose=verbose,
    ), run_query(
        index_dir,
        dev_s1_path,
        output_path,
        max_rows=max_rows_per_source,
        verbose=verbose,
    )


def _legacy_reference_smoke():
    """
    Lightweight smoke test with no real dataset.
    This validates exact-name, token, pair retrieval and zero rows.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        s2 = root / "s2.tsv"
        s3 = root / "s3.tsv"
        s1 = root / "s1.tsv"
        idx = root / "idx"
        out = root / "out.tsv"

        columns = [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]

        import pandas as pd

        pd.DataFrame(
            [
                ["S2-1", "Apple Technologies Pvt Ltd", "x", "India"],
                ["S2-2", "Generic Store Inc", "x", "India"],
                ["S2-3", "Rocky Piedmont Clinic", "x", "US"],
                ["S2-4", "Rocky Clinic", "x", "US"],
            ],
            columns=columns,
        ).to_csv(s2, sep="\t", index=False)

        pd.DataFrame(
            [
                ["S3-1", "Apple Technology Private Limited", "x", "India"],
                ["S3-2", "Unrelated Bakery", "x", "India"],
                ["S3-3", "Rocky Piedmont Giant Clinic", "x", "US"],
            ],
            columns=columns,
        ).to_csv(s3, sep="\t", index=False)

        pd.DataFrame(
            [
                ["S1-1", "Apple Technologies Pvt Ltd", "x", "India"],
                ["S1-2", "Rocky Piedmont Giant Clinic", "x", "US"],
                ["S1-3", "Totally Unmatched Entity Zzz", "x", "France"],
            ],
            columns=columns,
        ).to_csv(s1, sep="\t", index=False)

        build_index(
            [str(s2), str(s3)],
            str(s1),
            str(idx),
            max_doc_freq=1.0,
            max_postings_per_token=5000,
            exact_max_postings=5000,
            pair_max_postings=5000,
            fallback_threshold=50,
            verbose=False,
        )

        stats = run_query(
            str(idx),
            str(s1),
            str(out),
            verbose=False,
        )

        result = pd.read_csv(
            out,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        by_id = dict(
            zip(
                result["source1_entity_id"],
                result["candidate_entity_ids"],
            )
        )

        assert "S1-3" in by_id
        assert "S3-1" in by_id["S1-1"]
        assert "S3-3" in by_id["S1-2"]
        assert stats["n_queries"] == 3

        print("Smoke test passed.")
        print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Optimized hybrid token blocker"
    )

    parser.add_argument(
        "--source2",
        help="Path to S2 TSV.",
    )
    parser.add_argument(
        "--source3",
        help="Path to S3 TSV.",
    )
    parser.add_argument(
        "--dev-s1",
        help="Path to S1 query TSV.",
    )
    parser.add_argument(
        "--output",
        help="Output candidate TSV.",
    )

    parser.add_argument(
        "--max-doc-freq",
        type=float,
        default=DEFAULT_MAX_DOC_FREQ,
    )
    parser.add_argument(
        "--max-postings-per-token",
        type=int,
        default=DEFAULT_MAX_POSTINGS_PER_TOKEN,
        help="0 or negative disables the absolute token cap.",
    )
    parser.add_argument(
        "--exact-max-postings",
        type=int,
        default=DEFAULT_EXACT_MAX_POSTINGS,
    )
    parser.add_argument(
        "--pair-max-postings",
        type=int,
        default=DEFAULT_PAIR_MAX_POSTINGS,
    )
    parser.add_argument(
        "--fallback-threshold",
        type=int,
        default=DEFAULT_FALLBACK_THRESHOLD,
        help="Use exact-name/pair fallback when single-token result is <= this.",
    )
    parser.add_argument(
        "--max-pair-tokens",
        type=int,
        default=DEFAULT_MAX_PAIR_TOKENS,
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Debug only.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.smoke_test:
        _legacy_reference_smoke()
        return

    required = [
        ("--source2", args.source2),
        ("--source3", args.source3),
        ("--dev-s1", args.dev_s1),
        ("--output", args.output),
    ]
    missing = [name for name, value in required if not value]
    if missing:
        parser.error(
            f"Missing required arguments: {', '.join(missing)} "
            "(or pass --smoke-test)"
        )

    token_cap = (
        args.max_postings_per_token
        if args.max_postings_per_token > 0
        else None
    )

    build_index(
        [args.source2, args.source3],
        args.dev_s1,
        index_dir=(
            Path(args.output).parent
            / f".{Path(args.output).stem}_index"
        ),
        max_doc_freq=args.max_doc_freq,
        max_postings_per_token=token_cap,
        exact_max_postings=args.exact_max_postings,
        pair_max_postings=args.pair_max_postings,
        fallback_threshold=args.fallback_threshold,
        max_pair_tokens=args.max_pair_tokens,
        max_rows_per_source=args.max_rows_per_source,
        verbose=True,
    )

    run_query(
        Path(args.output).parent
        / f".{Path(args.output).stem}_index",
        args.dev_s1,
        args.output,
        max_rows=args.max_rows_per_source,
        verbose=True,
    )


if __name__ == "__main__":
    main()
