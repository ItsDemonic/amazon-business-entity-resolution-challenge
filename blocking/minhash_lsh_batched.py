"""

blocking/minhash_lsh.py



Disk-backed Character n-gram MinHash / LSH blocker.



This version is designed for:

    - 10M+ candidate records

    - bounded RAM during index construction

    - fast repeated querying of an existing index

    - parallel S1 querying

    - streaming candidate output



IMPORTANT:

The existing LSH band index (built via --build) produced by the previous

implementation is compatible with this query implementation. Do NOT

rebuild it.



WHAT CHANGED IN THIS REVISION (read this before running):



The previous query path re-did expensive one-time setup on EVERY shard

task instead of once per worker process:

    - QueryCache.open() (cheap - memory-mapped band files)

    - EntityIDReader(...).open() -> _build_offsets(), which does a

      SEQUENTIAL SCAN of the entire entity_ids.bin file (all \~10M+

      S2/S3 entity IDs) just to build a random-access offset table.



With query_chunk_size=2000, a 2.2M-row S1 query implied \~1,100 shard

tasks, each redoing that \~10M-record scan from scratch: \~11 billion

redundant loop iterations just to rebuild the same lookup table

repeatedly. That is the dominant cost that made this too slow for

millions of records.



Fix, in two parts:



  1. Worker setup now happens exactly once per worker process, via

     ProcessPoolExecutor(initializer=_worker_init, ...) - not once per

     shard. Cost is now paid `--workers` times total (e.g. 12), not

     `n_shards` times (was \~1,100+ for the full train set).



  2. entity_ids.bin's length-prefixed format required a full sequential

     scan (_build_offsets) to enable random access at all, and even then

     did one disk seek per candidate ID at query time. This revision

     adds a ONE-TIME conversion to `entity_ids.npy`: a fixed-width numpy

     byte-string array. It is created once (cached to disk next to your

     existing entity_ids.bin - your LSH bands are untouched), then

     loaded via np.load(mmap_mode="r") - near-instant, no scanning - and

     candidate IDs are resolved via a single vectorized numpy fancy-index

     operation instead of one seek+read per candidate.



First run after upgrading will print a one-time conversion message

("Converting entity_ids.bin -> entity_ids.npy"). Every run after that

is instant for this part.



Typical workflow (unchanged):



    # Build once, if needed (do NOT re-run this - you already have it):

    python -m blocking.minhash_lsh --build



    # Prepare query acceleration cache once (do NOT re-run this either):

    python -m blocking.minhash_lsh --prepare-cache



    # Query (this is the part that was slow - now fixed):

    python -m blocking.minhash_lsh --query \\

        --s1_dev common/dev_s1.tsv \\

        --output blocking/candidates_lsh.tsv \\

        --workers 12



Defaults:

    n_gram    = 3

    threshold = 0.3

    num_perm  = 64

    seed      = 42



The default query behavior is recall-preserving relative to the existing

LSH index:

    min_band_hits = 1

    max_candidates = None

"""



import argparse

import hashlib

import heapq

import os

import shutil

import struct

import time

from concurrent.futures import ProcessPoolExecutor

from pathlib import Path

from typing import Dict, Iterable, List, Set, Tuple



import numpy as np

import pandas as pd

from datasketch import MinHash



from common.normalize import normalize_name, normalize_address





# ============================================================

# Defaults

# ============================================================



DEFAULT_N_GRAM = 3

DEFAULT_THRESHOLD = 0.3

DEFAULT_NUM_PERM = 64

DEFAULT_SEED = 42



DEFAULT_INDEX_DIR = "blocking/minhash_index"

DEFAULT_OUTPUT = "blocking/candidates_lsh.tsv"



DEFAULT_CHUNK_SIZE = 25_000

DEFAULT_PROGRESS_EVERY = 100_000



# Query acceleration.

DEFAULT_WORKERS = max(1, min(12, os.cpu_count() or 1))

# Setup cost is now paid once per worker, not once per shard, so we can

# afford (and want) larger shards than before - this reduces task

# dispatch/IPC overhead relative to actual work done per task.

DEFAULT_QUERY_CHUNK_SIZE = 5_000



RAW_DTYPE = np.dtype(

    [

        ("bucket", "<u8"),

        ("entity", "<u8"),

    ]

)





# ============================================================

# Character n-grams

# ============================================================



def get_char_ngrams(text: str, n: int) -> Set[str]:

    if not text:

        return set()



    if len(text) < n:

        return {text}



    return {

        text[i:i + n]

        for i in range(len(text) - n + 1)

    }





def prepare_record_shingles(

    name: str,

    address: str,

    n: int,

) -> Set[str]:

    norm_name = normalize_name(str(name))

    norm_addr = normalize_address(str(address))

    combined = f"{norm_name} {norm_addr}".strip()

    return get_char_ngrams(combined, n)





# ============================================================

# MinHash

# ============================================================



def create_minhash(

    shingles: Iterable[str],

    num_perm: int,

    seed: int = DEFAULT_SEED,

) -> MinHash:

    mh = MinHash(num_perm=num_perm, seed=seed)



    for shingle in shingles:

        mh.update(shingle.encode("utf-8"))



    return mh





# ============================================================

# LSH parameter selection

# ============================================================



def false_positive_probability(

    threshold: float,

    bands: int,

    rows: int,

) -> float:

    xs = np.linspace(0.0, threshold, 4001)



    probabilities = (

        1.0

        - np.power(

            1.0 - np.power(xs, rows),

            bands,

        )

    )



    return float(np.trapezoid(probabilities, xs))





def false_negative_probability(

    threshold: float,

    bands: int,

    rows: int,

) -> float:

    xs = np.linspace(threshold, 1.0, 4001)



    probabilities = np.power(

        1.0 - np.power(xs, rows),

        bands,

    )



    return float(np.trapezoid(probabilities, xs))





def find_lsh_parameters(

    threshold: float,

    num_perm: int,

) -> Tuple[int, int]:

    best_error = float("inf")

    best_params = None



    for bands in range(1, num_perm + 1):

        max_rows = num_perm // bands



        for rows in range(1, max_rows + 1):

            fp = false_positive_probability(

                threshold,

                bands,

                rows,

            )



            fn = false_negative_probability(

                threshold,

                bands,

                rows,

            )



            error = 0.5 * fp + 0.5 * fn



            if error < best_error:

                best_error = error

                best_params = (bands, rows)



    if best_params is None:

        raise RuntimeError(

            "Unable to determine LSH parameters."

        )



    return best_params





# ============================================================

# Band hashing

# ============================================================



def hash_band(values: np.ndarray) -> np.uint64:

    digest = hashlib.blake2b(

        values.tobytes(),

        digest_size=8,

    ).digest()



    return np.frombuffer(

        digest,

        dtype=np.uint64,

    )[0]





def get_band_hashes(

    mh: MinHash,

    bands: int,

    rows: int,

) -> List[np.uint64]:

    values = np.asarray(mh.hashvalues)

    result = []



    for band in range(bands):

        start = band * rows

        end = start + rows



        if end > len(values):

            break



        result.append(

            hash_band(values[start:end])

        )



    return result





def bucket_for_band(

    country: str,

    band_hash: np.uint64,

    band_number: int,

) -> int:

    payload = (

        country.encode("utf-8")

        + b"\x00"

        + struct.pack("<Q", int(band_hash))

        + struct.pack("<I", band_number)

    )



    digest = hashlib.blake2b(

        payload,

        digest_size=8,

    ).digest()



    return struct.unpack("<Q", digest)[0]





# ============================================================

# Entity ID storage (build-time format, unchanged)

# ============================================================



def write_entity_id(fh, entity_id: str) -> None:

    encoded = str(entity_id).encode("utf-8")



    fh.write(

        struct.pack("<I", len(encoded))

    )



    fh.write(encoded)





def read_entity_ids(path: Path) -> List[str]:

    """

    Sequential scan of the length-prefixed entity_ids.bin file.



    This is intentionally only called from convert_entity_ids_to_npy()

    below, which runs it ONCE (cached to disk afterward) - never per

    shard, never per worker, never per query.

    """

    result = []



    with open(path, "rb") as fh:

        while True:

            length_bytes = fh.read(4)



            if not length_bytes:

                break



            length = struct.unpack(

                "<I",

                length_bytes,

            )[0]



            value = fh.read(length).decode("utf-8")

            result.append(value)



    return result





def convert_entity_ids_to_npy(index_dir: str, force: bool = False) -> Path:

    """

    One-time conversion of entity_ids.bin (length-prefixed, requires an

    O(n) sequential scan for any random access) into entity_ids.npy: a

    fixed-width numpy byte-string array that can be memory-mapped and

    randomly indexed in O(1), with zero per-run scanning cost.



    Cached to disk - runs once per index, ever, regardless of how many

    queries or how many workers you use afterward. Safe to call at the

    start of every query run: it's a no-op (a single file-exists check)

    once the sidecar already exists.

    """

    index_path = Path(index_dir)

    bin_path = index_path / "entity_ids.bin"

    npy_path = index_path / "entity_ids.npy"



    if npy_path.exists() and not force:

        return npy_path



    if not bin_path.exists():

        raise FileNotFoundError(f"Missing {bin_path}")



    print(

        f"Converting {bin_path.name} -> {npy_path.name} "

        f"(one-time, cached afterward)...",

        flush=True,

    )

    start = time.time()



    ids = read_entity_ids(bin_path)

    max_len = max((len(s) for s in ids), default=1)

    arr = np.array(ids, dtype=f"S{max_len}")

    del ids



    np.save(npy_path, arr)



    elapsed = time.time() - start

    print(

        f"  converted {len(arr):,} entity IDs in {elapsed:.1f}s "

        f"(max id length: {max_len} bytes)",

        flush=True,

    )



    return npy_path





# ============================================================

# Disk-backed index (build-time logic, unchanged)

# ============================================================



class DiskMinHashLSH:



    def __init__(

        self,

        index_dir: str,

        bands: int,

        rows: int,

    ):

        self.index_dir = Path(index_dir)

        self.bands = bands

        self.rows = rows



        self.raw_dir = self.index_dir / "raw"

        self.band_dir = self.index_dir / "bands"

        self.entity_map_path = (

            self.index_dir / "entity_ids.bin"

        )



    def initialize(self):

        if self.index_dir.exists():

            shutil.rmtree(self.index_dir)



        self.raw_dir.mkdir(parents=True)

        self.band_dir.mkdir(parents=True)



    def finalize_band(self, band_number: int):

        raw_path = (

            self.raw_dir

            / f"band_{band_number:03d}.bin"

        )



        index_path = (

            self.band_dir

            / f"band_{band_number:03d}.npy"

        )



        if not raw_path.exists():

            return



        file_size = raw_path.stat().st_size



        if file_size == 0:

            np.save(

                index_path,

                np.empty(0, dtype=RAW_DTYPE),

            )

            return



        count = file_size // RAW_DTYPE.itemsize



        print(

            f"    loading band {band_number}: "

            f"{count:,} records",

            flush=True,

        )



        raw = np.memmap(

            raw_path,

            dtype=RAW_DTYPE,

            mode="r",

            shape=(count,),

        )



        chunk_size = 2_000_000

        sorted_chunks = []



        for start in range(0, count, chunk_size):

            end = min(start + chunk_size, count)



            chunk = np.array(raw[start:end])

            chunk.sort(order=["bucket", "entity"])



            chunk_path = (

                self.raw_dir

                / f"band_{band_number:03d}_"

                f"sorted_{start}.bin"

            )



            chunk.tofile(chunk_path)

            sorted_chunks.append(chunk_path)



        del raw



        arrays = []

        positions = []

        heap = []



        for chunk_number, path in enumerate(sorted_chunks):

            arr = np.fromfile(

                path,

                dtype=RAW_DTYPE,

            )



            arrays.append(arr)

            positions.append(0)



            if len(arr):

                first = arr[0]



                heapq.heappush(

                    heap,

                    (

                        int(first["bucket"]),

                        int(first["entity"]),

                        chunk_number,

                    ),

                )



        merged = np.empty(

            count,

            dtype=RAW_DTYPE,

        )



        output_position = 0



        while heap:

            bucket, entity, chunk_number = heapq.heappop(heap)



            merged[output_position] = (

                bucket,

                entity,

            )

            output_position += 1



            positions[chunk_number] += 1

            pos = positions[chunk_number]

            arr = arrays[chunk_number]



            if pos < len(arr):

                value = arr[pos]



                heapq.heappush(

                    heap,

                    (

                        int(value["bucket"]),

                        int(value["entity"]),

                        chunk_number,

                    ),

                )



        np.save(index_path, merged)



        for path in sorted_chunks:

            try:

                path.unlink()

            except OSError:

                pass



        try:

            raw_path.unlink()

        except OSError:

            pass



        print(

            f"    finalized band {band_number}",

            flush=True,

        )





# ============================================================

# Query acceleration cache (unchanged)

# ============================================================



class QueryCache:



    """

    For each sorted band, store:



        keys   = unique bucket values

        offsets = [start0, start1, ..., end]



    This changes a query from two searchsorted operations over

    10.3M bucket entries to two searchsorted operations over

    the number of UNIQUE buckets.



    The original band .npy files remain untouched and memory-mapped.

    """



    def __init__(

        self,

        index_dir: str,

        bands: int,

    ):

        self.index_dir = Path(index_dir)

        self.bands = bands



        self.cache_dir = (

            self.index_dir / "query_cache"

        )



        self.band_arrays = []

        self.keys = []

        self.offsets = []



    def _paths(self, band: int):

        return (

            self.cache_dir

            / f"band_{band:03d}_keys.npy",

            self.cache_dir

            / f"band_{band:03d}_offsets.npy",

        )



    def cache_exists(self) -> bool:

        for band in range(self.bands):

            key_path, offset_path = self._paths(band)



            if not key_path.exists() or not offset_path.exists():

                return False



        return True



    def prepare(self):

        self.cache_dir.mkdir(

            parents=True,

            exist_ok=True,

        )



        print(

            "Preparing query acceleration cache...",

            flush=True,

        )



        for band in range(self.bands):

            band_path = (

                self.index_dir

                / "bands"

                / f"band_{band:03d}.npy"

            )



            if not band_path.exists():

                raise FileNotFoundError(

                    f"Missing band index: {band_path}"

                )



            key_path, offset_path = self._paths(band)



            if key_path.exists() and offset_path.exists():

                print(

                    f"  cache {band + 1}/{self.bands}: "

                    f"already exists",

                    flush=True,

                )

                continue



            arr = np.load(

                band_path,

                mmap_mode="r",

            )



            buckets = arr["bucket"]

            count = len(buckets)



            if count == 0:

                keys = np.empty(

                    0,

                    dtype=np.uint64,

                )

                offsets = np.array(

                    [0],

                    dtype=np.uint64,

                )

            else:

                changes = (

                    np.flatnonzero(

                        buckets[1:] != buckets[:-1]

                    ).astype(np.int64)

                    + 1

                )



                starts = np.concatenate(

                    [

                        np.array([0], dtype=np.int64),

                        changes,

                    ]

                )



                keys = np.asarray(

                    buckets[starts],

                    dtype=np.uint64,

                )



                offsets = np.concatenate(

                    [

                        starts,

                        np.array(

                            [count],

                            dtype=np.int64,

                        ),

                    ]

                ).astype(np.uint64)



            np.save(key_path, keys)

            np.save(offset_path, offsets)



            print(

                f"  cache {band + 1}/{self.bands}: "

                f"{len(keys):,} unique buckets / "

                f"{count:,} records",

                flush=True,

            )



            del arr



        print(

            "Query cache preparation complete.",

            flush=True,

        )



    def open(self):

        if not self.cache_exists():

            raise RuntimeError(

                "Query cache is missing. Run "

                "--prepare-cache first."

            )



        for band in range(self.bands):

            band_path = (

                self.index_dir

                / "bands"

                / f"band_{band:03d}.npy"

            )



            key_path, offset_path = self._paths(band)



            self.band_arrays.append(

                np.load(

                    band_path,

                    mmap_mode="r",

                )

            )



            self.keys.append(

                np.load(

                    key_path,

                    mmap_mode="r",

                )

            )



            self.offsets.append(

                np.load(

                    offset_path,

                    mmap_mode="r",

                )

            )



    def lookup(
        self,
        band: int,
        bucket: int,
    ) -> np.ndarray:
        """Look up one bucket."""
        keys = self.keys[band]

        if keys.size == 0:
            return np.empty(0, dtype=np.uint64)

        pos = np.searchsorted(keys, bucket)

        if pos == keys.size or keys[pos] != bucket:
            return np.empty(0, dtype=np.uint64)

        offsets = self.offsets[band]
        left = offsets[pos]
        right = offsets[pos + 1]

        return self.band_arrays[band]["entity"][left:right]

    def lookup_many(
        self,
        band: int,
        buckets: np.ndarray,
    ) -> List[Tuple[int, np.ndarray]]:
        """Look up many buckets with one vectorized searchsorted."""
        buckets = np.asarray(buckets, dtype=np.uint64)

        if buckets.size == 0:
            return []

        keys = self.keys[band]

        if keys.size == 0:
            return []

        positions = np.searchsorted(keys, buckets)
        valid = positions < keys.size

        if not np.any(valid):
            return []

        query_indices = np.flatnonzero(valid)
        valid_positions = positions[valid]

        matched = keys[valid_positions] == buckets[valid]

        if not np.any(matched):
            return []

        query_indices = query_indices[matched]
        valid_positions = valid_positions[matched]

        offsets = self.offsets[band]
        arr = self.band_arrays[band]

        result = []
        for query_index, pos in zip(query_indices, valid_positions):
            left = offsets[pos]
            right = offsets[pos + 1]
            result.append(
                (
                    int(query_index),
                    arr["entity"][left:right],
                )
            )

        return result


def _rank_and_filter_candidates(
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


def query_internal_ids(
    cache: QueryCache,
    country: str,
    band_hashes: List[np.uint64],
    min_band_hits: int = 1,
    max_candidates: int = 0,
) -> np.ndarray:
    """Single-record compatibility wrapper."""
    return query_internal_ids_batch(
        cache,
        [country],
        [band_hashes],
        min_band_hits,
        max_candidates,
    )[0]


def query_internal_ids_batch(
    cache: QueryCache,
    countries: List[str],
    band_hashes_batch: List[List[np.uint64]],
    min_band_hits: int = 1,
    max_candidates: int = 0,
) -> List[np.ndarray]:
    """
    Query all records in a shard using one vectorized search per band.

    The bucket hashing is unchanged, so the existing on-disk LSH index is
    compatible. Candidate filtering is still done independently per query.
    """
    n = len(countries)
    empty = np.empty(0, dtype=np.uint64)

    if n == 0:
        return []

    per_query_parts: List[List[np.ndarray]] = [
        [] for _ in range(n)
    ]

    active_indices = [
        i
        for i, hashes in enumerate(band_hashes_batch)
        if hashes
    ]

    if not active_indices:
        return [empty.copy() for _ in range(n)]

    max_bands = max(
        len(band_hashes_batch[i])
        for i in active_indices
    )

    for band_number in range(max_bands):
        query_indices: List[int] = []
        buckets = np.empty(
            len(active_indices),
            dtype=np.uint64,
        )
        bucket_count = 0

        for query_index in active_indices:
            hashes = band_hashes_batch[query_index]
            if band_number >= len(hashes):
                continue

            buckets[bucket_count] = bucket_for_band(
                countries[query_index],
                hashes[band_number],
                band_number,
            )
            query_indices.append(query_index)
            bucket_count += 1

        if bucket_count == 0:
            continue

        matches = cache.lookup_many(
            band_number,
            buckets[:bucket_count],
        )

        for local_index, values in matches:
            query_index = query_indices[local_index]
            if values.size:
                per_query_parts[query_index].append(values)

    results: List[np.ndarray] = []

    for query_index in range(n):
        parts = per_query_parts[query_index]

        if not parts:
            results.append(empty.copy())
            continue

        raw = np.concatenate(parts)
        results.append(
            _rank_and_filter_candidates(
                raw,
                min_band_hits,
                max_candidates,
            )
        )

    return results


def _worker_init(

    index_dir: str,

    entity_ids_npy_path: str,

    bands: int,

    rows: int,

    n_gram: int,

    num_perm: int,

    seed: int,

    min_band_hits: int,

    max_candidates: int,

):

    global _WORKER_CACHE, _WORKER_ENTITY_IDS

    global _WORKER_N_GRAM, _WORKER_NUM_PERM, _WORKER_SEED

    global _WORKER_BANDS, _WORKER_ROWS

    global _WORKER_MIN_BAND_HITS, _WORKER_MAX_CANDIDATES



    _WORKER_N_GRAM = n_gram

    _WORKER_NUM_PERM = num_perm

    _WORKER_SEED = seed

    _WORKER_BANDS = bands

    _WORKER_ROWS = rows

    _WORKER_MIN_BAND_HITS = min_band_hits

    _WORKER_MAX_CANDIDATES = max_candidates



    _WORKER_CACHE = QueryCache(index_dir, bands)

    _WORKER_CACHE.open()



    # Memory-mapped: near-instant, no O(n) scan, and the OS shares the

    # underlying physical pages across every worker process that opens

    # this same read-only file - no per-process duplication of the

    # \~10M-entry ID array in RAM.

    _WORKER_ENTITY_IDS = np.load(entity_ids_npy_path, mmap_mode="r")





def _process_shard(records: List[Tuple], shard_path: str, profile: bool = False):
    """Process a shard using batched/vectorized LSH lookups."""
    total_candidates = 0
    zero_candidates = 0

    t_minhash = 0.0
    t_lookup = 0.0
    t_resolve = 0.0

    n_records = len(records)
    countries = [str(record[3]) for record in records]

    band_hashes_batch: List[List[np.uint64]] = [
        [] for _ in range(n_records)
    ]

    if profile:
        t0 = time.perf_counter()

    for i, record in enumerate(records):
        business_name = str(record[1])
        business_address = str(record[2])

        shingles = prepare_record_shingles(
            business_name,
            business_address,
            _WORKER_N_GRAM,
        )

        if shingles:
            mh = create_minhash(
                shingles,
                _WORKER_NUM_PERM,
                _WORKER_SEED,
            )

            band_hashes_batch[i] = get_band_hashes(
                mh,
                _WORKER_BANDS,
                _WORKER_ROWS,
            )

    if profile:
        t_minhash = time.perf_counter() - t0

    if profile:
        t1 = time.perf_counter()

    all_internal_ids = query_internal_ids_batch(
        _WORKER_CACHE,
        countries,
        band_hashes_batch,
        _WORKER_MIN_BAND_HITS,
        _WORKER_MAX_CANDIDATES,
    )

    if profile:
        t_lookup = time.perf_counter() - t1

    with open(
        shard_path,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as fh:
        write_output_header(fh)

        for i, record in enumerate(records):
            s1_id = str(record[0])
            internal_ids = all_internal_ids[i]

            if internal_ids.size:
                if profile:
                    t2 = time.perf_counter()

                raw = _WORKER_ENTITY_IDS[internal_ids]
                candidate_ids = [
                    b.decode("utf-8")
                    for b in raw.tolist()
                ]

                if profile:
                    t_resolve += time.perf_counter() - t2
            else:
                candidate_ids = []

            write_output_row(
                fh,
                s1_id,
                candidate_ids,
            )

            count = len(candidate_ids)
            total_candidates += count

            if count == 0:
                zero_candidates += 1

    if profile:
        n = max(n_records, 1)
        print(
            f"    [profile] per-record avg (ms): "
            f"minhash+shingle={1000 * t_minhash / n:.3f}"
            f" | lsh_lookup={1000 * t_lookup / n:.3f}"
            f" | resolve_ids={1000 * t_resolve / n:.3f}"
            f" | total={1000 * (t_minhash + t_lookup + t_resolve) / n:.3f}",
            flush=True,
        )

    return (
        str(shard_path),
        len(records),
        total_candidates,
        zero_candidates,
    )


def write_output_header(fh):

    fh.write(

        "source1_entity_id\tcandidate_entity_ids\n"

    )





def write_output_row(

    fh,

    s1_id: str,

    candidate_ids: List[str],

):

    fh.write(

        s1_id

        + "\t"

        + ",".join(candidate_ids)

        + "\n"

    )





# ============================================================

# Metadata

# ============================================================



def load_metadata(

    index_dir: str,

) -> Dict[str, str]:

    metadata = {}



    path = (

        Path(index_dir)

        / "metadata.txt"

    )



    with open(

        path,

        "r",

        encoding="utf-8",

    ) as fh:

        for line in fh:

            key, value = (

                line.strip()

                .split("=", 1)

            )

            metadata[key] = value



    return metadata





# ============================================================

# Parallel query (rewritten)

# ============================================================



def query_index(

    index_dir: str,

    s1_path: str,

    output_path: str,

    workers: int,

    min_band_hits: int = 1,

    max_candidates: int = 0,

    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,

    profile: bool = False,

):

    metadata = load_metadata(index_dir)



    n_gram = int(metadata["n_gram"])

    threshold = float(metadata["threshold"])

    num_perm = int(metadata["num_perm"])

    seed = int(metadata["seed"])

    bands = int(metadata["bands"])

    rows = int(metadata["rows"])



    print()

    print("=" * 70)

    print("PARALLEL MINHASH LSH QUERY")

    print("=" * 70)



    print(f"n-gram          : {n_gram}")

    print(f"threshold       : {threshold}")

    print(f"num_perm        : {num_perm}")

    print(f"bands           : {bands}")

    print(f"rows/band       : {rows}")

    print(f"workers         : {workers}")

    print(f"min_band_hits   : {min_band_hits}")

    print(

        f"max_candidates  : "

        f"{max_candidates if max_candidates > 0 else 'none'}"

    )

    print(f"chunk size      : {query_chunk_size}")

    print()



    # Make sure the LSH cache exists before launching workers.

    cache_check = QueryCache(index_dir, bands)



    if not cache_check.cache_exists():

        raise RuntimeError(

            "Query acceleration cache is missing.\n"

            "Run:\n"

            "  python -m blocking.minhash_lsh "

            "--prepare-cache"

        )



    # One-time conversion (cached to disk); cheap no-op on every run

    # after the first.

    entity_ids_npy_path = convert_entity_ids_to_npy(index_dir)



    columns = [

        "entity_id",

        "business_name",

        "business_address",

        "country",

    ]



    output_path_obj = Path(output_path)

    output_path_obj.parent.mkdir(

        parents=True,

        exist_ok=True,

    )



    shard_dir = (

        output_path_obj.parent

        / f".{output_path_obj.stem}_shards"

    )



    if shard_dir.exists():

        shutil.rmtree(shard_dir)



    shard_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    start_time = time.time()



    shard_paths = []

    total_queries = 0

    total_candidates = 0

    total_zero = 0



    print(

        f"Launching {workers} worker process(es) - on Windows this "

        f"means a cold interpreter start + numpy/pandas/datasketch "

        f"import per worker before the first shard can run. If this "

        f"line is the last thing you see for more than \~1-2 minutes, "

        f"check Task Manager for python.exe CPU usage before assuming "

        f"a hang.",

        flush=True,

    )



    with ProcessPoolExecutor(

        max_workers=workers,

        initializer=_worker_init,

        initargs=(

            index_dir,

            str(entity_ids_npy_path),

            bands,

            rows,

            n_gram,

            num_perm,

            seed,

            min_band_hits,

            max_candidates,

        ),

    ) as executor:



        futures = []



        shard_number = 0



        def _report(finished):

            nonlocal total_queries, total_candidates, total_zero



            (

                _,

                n_queries,

                n_candidates,

                n_zero,

            ) = finished



            total_queries += n_queries

            total_candidates += n_candidates

            total_zero += n_zero



            elapsed = time.time() - start_time

            rate = (

                total_queries / elapsed

                if elapsed > 0

                else 0

            )

            average = (

                total_candidates / total_queries

                if total_queries

                else 0

            )



            print(

                f"  queried={total_queries:,}"

                f" | avg={average:.2f}"

                f" | rate={rate:.2f}/s"

                f" | zero={total_zero:,}"

                f" | elapsed={elapsed:.1f}s",

                flush=True,

            )



        for chunk in pd.read_csv(

            s1_path,

            sep="\t",

            dtype=str,

            keep_default_na=False,

            usecols=columns,

            chunksize=query_chunk_size,

        ):

            records = list(

                chunk.itertuples(

                    index=False,

                    name=None,

                )

            )



            shard_path = (

                shard_dir

                / f"shard_{shard_number:06d}.tsv"

            )



            future = executor.submit(

                _process_shard,

                records,

                str(shard_path),

                profile,

            )



            futures.append(future)

            shard_paths.append(shard_path)

            shard_number += 1



            print(

                f"  submitted shard {shard_number} "

                f"({len(records):,} records) - "

                f"{len(futures)} in flight",

                flush=True,

            )



            # Keep only a bounded number of in-flight batches so we

            # never queue the entire dataset's worth of futures at once.

            # Report on EVERY completed shard (not just when a bound is

            # hit) so a small dev file still gives live feedback instead

            # of going silent until the whole run finishes.

            if len(futures) >= workers * 2:

                _report(futures.pop(0).result())



        while futures:

            _report(futures.pop(0).result())



    elapsed = time.time() - start_time



    average = (

        total_candidates / total_queries

        if total_queries

        else 0

    )



    print()

    print("=" * 70)

    print("QUERY SHARDS COMPLETE")

    print("=" * 70)



    print(f"S1 queries       : {total_queries:,}")

    print(f"Total candidates : {total_candidates:,}")

    print(f"Average/query    : {average:.2f}")

    print(f"Zero candidates  : {total_zero:,}")

    print(f"Query time       : {elapsed:.1f}s")

    print(

        f"Query rate       : "

        f"{total_queries / elapsed:.2f} S1/s"

    )



    # Combine shards in original order.

    print()

    print(

        f"Combining {len(shard_paths):,} output shards...",

        flush=True,

    )



    with open(

        output_path,

        "w",

        encoding="utf-8",

        buffering=1024 * 1024,

    ) as out:



        write_output_header(out)



        for shard_path in shard_paths:

            with open(

                shard_path,

                "r",

                encoding="utf-8",

                buffering=1024 * 1024,

            ) as fh:



                next(fh, None)  # skip shard header



                for line in fh:

                    out.write(line)



    shutil.rmtree(shard_dir)



    print(

        f"Final output: {output_path}",

        flush=True,

    )

    print("Done.", flush=True)





# ============================================================

# Prepare cache (unchanged)

# ============================================================



def prepare_query_cache(

    index_dir: str,

):

    metadata = load_metadata(index_dir)



    bands = int(metadata["bands"])



    cache = QueryCache(

        index_dir,

        bands,

    )



    cache.prepare()





# ============================================================

# Build index (unchanged, except it now also produces the .npy

# sidecar immediately so a fresh build never needs the one-time

# conversion message on its first query)

# ============================================================



def build_index(

    s2_path: str,

    s3_path: str,

    index_dir: str,

    n_gram: int,

    threshold: float,

    num_perm: int,

    seed: int,

):

    bands, rows = find_lsh_parameters(

        threshold,

        num_perm,

    )



    print()

    print("=" * 70)

    print("DISK-BACKED MINHASH LSH")

    print("=" * 70)



    print(f"n-gram       : {n_gram}")

    print(f"threshold    : {threshold}")

    print(f"num_perm     : {num_perm}")

    print(f"bands        : {bands}")

    print(f"rows/band    : {rows}")

    print(f"index        : {index_dir}")

    print("=" * 70)



    index = DiskMinHashLSH(

        index_dir,

        bands,

        rows,

    )



    index.initialize()



    handles = {}



    for band in range(bands):

        path = (

            index.raw_dir

            / f"band_{band:03d}.bin"

        )



        handles[band] = open(

            path,

            "ab",

            buffering=1024 * 1024,

        )



    entity_map = open(

        index.entity_map_path,

        "wb",

        buffering=1024 * 1024,

    )



    internal_id = 0

    total_records = 0

    start_time = time.time()



    for source_number, source_path in [

        (2, s2_path),

        (3, s3_path),

    ]:



        print()

        print(

            f"Processing S{source_number}: "

            f"{source_path}",

            flush=True,

        )



        source_records = 0



        for chunk in pd.read_csv(

            source_path,

            sep="\t",

            dtype=str,

            keep_default_na=False,

            usecols=[

                "entity_id",

                "business_name",

                "business_address",

                "country",

            ],

            chunksize=DEFAULT_CHUNK_SIZE,

        ):



            for record in chunk.itertuples(

                index=False

            ):

                entity_id = str(record.entity_id)

                country = str(record.country)



                shingles = prepare_record_shingles(

                    record.business_name,

                    record.business_address,

                    n_gram,

                )



                if not shingles:

                    continue



                mh = create_minhash(

                    shingles,

                    num_perm,

                    seed,

                )



                band_hashes = get_band_hashes(

                    mh,

                    bands,

                    rows,

                )



                write_entity_id(

                    entity_map,

                    entity_id,

                )



                for band_number, band_hash in enumerate(

                    band_hashes

                ):

                    bucket = bucket_for_band(

                        country,

                        band_hash,

                        band_number,

                    )



                    handles[band_number].write(

                        struct.pack(

                            "<QQ",

                            bucket,

                            internal_id,

                        )

                    )



                internal_id += 1

                source_records += 1

                total_records += 1



                if (

                    total_records

                    % DEFAULT_PROGRESS_EVERY

                    == 0

                ):

                    elapsed = time.time() - start_time



                    print(

                        f"  processed={total_records:,}"

                        f" | elapsed={elapsed:.1f}s",

                        flush=True,

                    )



        print(

            f"Finished S{source_number}: "

            f"{source_records:,} records",

            flush=True,

        )



    entity_map.close()



    for fh in handles.values():

        fh.close()



    print()

    print(

        f"Total records indexed: "

        f"{total_records:,}"

    )



    print()

    print(

        "Sorting LSH bands...",

        flush=True,

    )



    for band_number in range(bands):

        print(

            f"  Band {band_number + 1}/{bands}",

            flush=True,

        )



        index.finalize_band(

            band_number

        )



    metadata_path = (

        Path(index_dir)

        / "metadata.txt"

    )



    with open(

        metadata_path,

        "w",

        encoding="utf-8",

    ) as fh:

        fh.write(f"n_gram={n_gram}\n")

        fh.write(f"threshold={threshold}\n")

        fh.write(f"num_perm={num_perm}\n")

        fh.write(f"seed={seed}\n")

        fh.write(f"bands={bands}\n")

        fh.write(f"rows={rows}\n")

        fh.write(f"records={total_records}\n")



    # Produce the fast-access sidecar immediately so the first query

    # run never has to pay the one-time conversion cost separately.

    convert_entity_ids_to_npy(index_dir, force=True)



    elapsed = time.time() - start_time



    print()

    print("=" * 70)

    print("BUILD COMPLETE")

    print("=" * 70)



    print(f"Records : {total_records:,}")

    print(

        f"Time    : {elapsed / 60:.2f} minutes"

    )





# ============================================================

# CLI (unchanged interface)

# ============================================================



def main():

    parser = argparse.ArgumentParser(

        description=(

            "Disk-backed MinHash LSH blocker."

        )

    )



    parser.add_argument(

        "--build",

        action="store_true",

        help="Build the MinHash LSH index.",

    )



    parser.add_argument(

        "--prepare-cache",

        action="store_true",

        help=(

            "Prepare persistent query acceleration "

            "cache for an existing index."

        ),

    )



    parser.add_argument(

        "--query",

        action="store_true",

        help="Query an existing MinHash LSH index.",

    )



    parser.add_argument(

        "--s2",

        default=(

            "dataset/train/"

            "train_source2.tsv"

        ),

        help="Path to S2 TSV.",

    )



    parser.add_argument(

        "--s3",

        default=(

            "dataset/train/"

            "train_source3.tsv"

        ),

        help="Path to S3 TSV.",

    )



    parser.add_argument(

        "--s1_dev",

        default="common/dev_s1.tsv",

        help=(

            "Path to S1 TSV. Despite the name, "

            "this may point to the full train/test S1."

        ),

    )



    parser.add_argument(

        "--index",

        default=DEFAULT_INDEX_DIR,

        help="MinHash index directory.",

    )



    parser.add_argument(

        "--output",

        default=DEFAULT_OUTPUT,

        help="Candidate output TSV.",

    )



    parser.add_argument(

        "--n_gram",

        type=int,

        default=DEFAULT_N_GRAM,

        help="Character n-gram size.",

    )



    parser.add_argument(

        "--threshold",

        type=float,

        default=DEFAULT_THRESHOLD,

        help="LSH threshold.",

    )



    parser.add_argument(

        "--num_perm",

        type=int,

        default=DEFAULT_NUM_PERM,

        help="Number of MinHash permutations.",

    )



    parser.add_argument(

        "--seed",

        type=int,

        default=DEFAULT_SEED,

        help="MinHash seed.",

    )



    parser.add_argument(

        "--workers",

        type=int,

        default=DEFAULT_WORKERS,

        help=(

            "Number of parallel query processes. "

            "Start around 12 and benchmark."

        ),

    )



    parser.add_argument(

        "--min-band-hits",

        type=int,

        default=1,

        help=(

            "Minimum number of LSH bands in which a candidate "

            "must collide. Default 1 preserves baseline recall."

        ),

    )



    parser.add_argument(

        "--max-candidates",

        type=int,

        default=0,

        help=(

            "Optional candidate cap ranked by band collision "

            "count. 0 means unlimited."

        ),

    )



    parser.add_argument(

        "--query-chunk-size",

        type=int,

        default=DEFAULT_QUERY_CHUNK_SIZE,

        help="Number of S1 records submitted per worker batch.",

    )



    parser.add_argument(

        "--profile",

        action="store_true",

        help=(

            "Print a per-shard timing breakdown (MinHash/shingling vs "

            "LSH band lookup vs candidate-ID resolution) to diagnose "

            "where per-record time is actually going, instead of "

            "guessing."

        ),

    )



    args = parser.parse_args()



    actions = [

        args.build,

        args.prepare_cache,

        args.query,

    ]



    if not any(actions):

        parser.error(

            "Specify --build, --prepare-cache, and/or --query."

        )



    if args.workers < 1:

        parser.error("--workers must be >= 1.")



    if args.min_band_hits < 1:

        parser.error("--min-band-hits must be >= 1.")



    if args.max_candidates < 0:

        parser.error("--max-candidates must be >= 0.")



    if args.query_chunk_size < 1:

        parser.error("--query-chunk-size must be >= 1.")



    if args.build:

        build_index(

            s2_path=args.s2,

            s3_path=args.s3,

            index_dir=args.index,

            n_gram=args.n_gram,

            threshold=args.threshold,

            num_perm=args.num_perm,

            seed=args.seed,

        )



    if args.prepare_cache:

        prepare_query_cache(

            args.index

        )



    if args.query:

        query_index(

            index_dir=args.index,

            s1_path=args.s1_dev,

            output_path=args.output,

            workers=args.workers,

            min_band_hits=args.min_band_hits,

            max_candidates=args.max_candidates,

            query_chunk_size=args.query_chunk_size,

            profile=args.profile,

        )





if __name__ == "__main__":

    main()