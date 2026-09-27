"""
FAST / RESUMABLE TEST MINHASH-LSH BUILD

Run from project root while your P1+P3 blocker keeps running:

    python blocking/fast_minhash_test_build.py --workers 12

This script:
- resumes blocking/minhash_test_index if it was interrupted;
- uses MinHash.bulk() instead of calling MinHash.update() for every n-gram;
- parallelizes S2/S3 indexing across CPU worker processes;
- preserves the existing n-gram / threshold / 64-permutation / seed settings;
- finalizes the same band layout (21 bands x 3 rows) and query cache.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasketch import MinHash
from common.normalize import normalize_name, normalize_address
from blocking.minhash_lsh import (
    find_lsh_parameters,
    bucket_for_band,
    hash_band,
    QueryCache,
    convert_entity_ids_to_npy,
)

RAW_DTYPE = np.dtype([("bucket", "<u8"), ("entity", "<u8")])

N_GRAM = 3
THRESHOLD = 0.3
NUM_PERM = 64
SEED = 42
BANDS, ROWS = find_lsh_parameters(THRESHOLD, NUM_PERM)


def record_shingles(name: str, address: str) -> set[str]:
    nn = normalize_name(name)
    na = normalize_address(address)
    combined = f"{nn} {na}".strip()
    if not combined:
        return set()
    if len(combined) < N_GRAM:
        return {combined}
    return {
        combined[i:i + N_GRAM]
        for i in range(len(combined) - N_GRAM + 1)
    }


def split_ranges(start: int, end: int, n: int) -> List[Tuple[int, int]]:
    if end <= start:
        return []
    span = end - start
    out = []
    for i in range(n):
        a = start + (span * i) // n
        b = start + (span * (i + 1)) // n
        if b > a:
            out.append((a, b))
    return out


def write_entity_batch(fh, ids: List[bytes]) -> None:
    buf = bytearray()
    for eid in ids:
        buf += struct.pack("<I", len(eid))
        buf += eid
    fh.write(buf)


def compute_bulk(shingle_batch):
    try:
        return MinHash.bulk(
            shingle_batch,
            num_perm=NUM_PERM,
            seed=SEED,
        )
    except AttributeError:
        # Older datasketch fallback.
        return list(
            MinHash.generator(
                shingle_batch,
                num_perm=NUM_PERM,
                seed=SEED,
            )
        )


def worker(task):
    """
    One independent byte range of one TSV.
    Returns task-local raw files; the parent assigns global IDs.
    """
    source_num, task_id, path, start, end, temp_root, batch_size = task

    task_dir = Path(temp_root) / f"task_{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    entity_path = task_dir / "entity_ids.bin"
    band_paths = [
        task_dir / f"band_{b:03d}.bin"
        for b in range(BANDS)
    ]

    efh = open(entity_path, "wb", buffering=8 * 1024 * 1024)
    band_fhs = [
        open(p, "wb", buffering=8 * 1024 * 1024)
        for p in band_paths
    ]

    batch_shingles = []
    batch_countries = []
    batch_ids = []
    local_id = 0

    def flush():
        nonlocal local_id
        if not batch_shingles:
            return

        mhs = compute_bulk(batch_shingles)
        _n = len(mhs)

        write_entity_batch(efh, batch_ids)

        for band_no in range(BANDS):
            lo = band_no * ROWS
            hi = lo + ROWS
            out = np.empty(_n, dtype=RAW_DTYPE)

            for i, mh in enumerate(mhs):
                bh = hash_band(mh.hashvalues[lo:hi])
                out[i] = (
                    bucket_for_band(
                        batch_countries[i],
                        bh,
                        band_no,
                    ),
                    local_id + i,
                )

            out.tofile(band_fhs[band_no])

        local_id += _n
        batch_shingles.clear()
        batch_countries.clear()
        batch_ids.clear()

    try:
        with open(path, "rb", buffering=8 * 1024 * 1024) as fh:
            fh.seek(start)

            if start > 0:
                fh.readline()
            else:
                fh.readline()  # header

            while True:
                pos = fh.tell()
                if pos >= end:
                    break

                line = fh.readline()
                if not line:
                    break

                parts = line.rstrip(b"\r\n").split(b"\t", 3)
                if len(parts) != 4:
                    continue

                eid_b, name_b, addr_b, country_b = parts

                name = name_b.decode("utf-8", errors="replace")
                addr = addr_b.decode("utf-8", errors="replace")
                country = country_b.decode("utf-8", errors="replace")

                sh = record_shingles(name, addr)
                if not sh:
                    continue

                batch_shingles.append(
                    [s.encode("utf-8") for s in sh]
                )
                batch_countries.append(country)
                batch_ids.append(eid_b)

                if len(batch_shingles) >= batch_size:
                    flush()

        flush()
    finally:
        efh.close()
        for h in band_fhs:
            h.close()

    return (
        source_num,
        task_id,
        start,
        end,
        local_id,
        str(entity_path),
        [str(p) for p in band_paths],
    )


def entity_count_and_last(path: Path):
    if not path.exists():
        return 0, None, 0

    count = 0
    last = None
    good_end = 0

    with open(path, "rb", buffering=8 * 1024 * 1024) as fh:
        while True:
            lb = fh.read(4)
            if len(lb) != 4:
                break
            ln = struct.unpack("<I", lb)[0]
            data = fh.read(ln)
            if len(data) != ln:
                break
            count += 1
            last = data
            good_end = fh.tell()

    return count, last, good_end


def truncate_to_n_ids(path: Path, n: int):
    if not path.exists():
        return

    if n <= 0:
        with open(path, "wb"):
            pass
        return

    pos = 0
    count = 0
    with open(path, "rb", buffering=8 * 1024 * 1024) as fh:
        while count < n:
            lb = fh.read(4)
            if len(lb) != 4:
                break
            ln = struct.unpack("<I", lb)[0]
            data = fh.read(ln)
            if len(data) != ln:
                break
            pos = fh.tell()
            count += 1

    with open(path, "r+b") as fh:
        fh.truncate(pos)


def repair_partial(index_dir: Path):
    raw_dir = index_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "bands").mkdir(parents=True, exist_ok=True)
    (index_dir / "query_cache").mkdir(parents=True, exist_ok=True)

    raw_paths = [
        raw_dir / f"band_{b:03d}.bin"
        for b in range(BANDS)
    ]

    counts = [
        p.stat().st_size // RAW_DTYPE.itemsize
        for p in raw_paths
        if p.exists()
    ]

    raw_count = min(counts) if counts else 0

    for p in raw_paths:
        if p.exists():
            with open(p, "r+b") as fh:
                fh.truncate(raw_count * RAW_DTYPE.itemsize)

    entity_path = index_dir / "entity_ids.bin"
    entity_count, last_id, _ = entity_count_and_last(entity_path)

    count = min(raw_count, entity_count)

    for p in raw_paths:
        if p.exists():
            with open(p, "r+b") as fh:
                fh.truncate(count * RAW_DTYPE.itemsize)

    if entity_path.exists():
        truncate_to_n_ids(entity_path, count)

    return count, last_id


def find_after_id(tsv_path: Path, target_id: bytes):
    if target_id is None:
        return 0

    with open(tsv_path, "rb", buffering=8 * 1024 * 1024) as fh:
        fh.readline()
        while True:
            line = fh.readline()
            if not line:
                return None
            if line.split(b"\t", 1)[0] == target_id:
                return fh.tell()


def append_task_outputs(index_dir: Path, results, base_id: int):
    raw_dir = index_dir / "raw"
    entity_out = index_dir / "entity_ids.bin"

    current = base_id

    ordered = sorted(results, key=lambda r: (r[0], r[2]))

    with open(entity_out, "ab", buffering=8 * 1024 * 1024) as efh:
        for (
            _source,
            _task_id,
            _start,
            _end,
            count,
            entity_path,
            band_paths,
        ) in ordered:

            with open(
                entity_path,
                "rb",
                buffering=8 * 1024 * 1024,
            ) as src:
                shutil.copyfileobj(
                    src,
                    efh,
                    length=8 * 1024 * 1024,
                )

            for b in range(BANDS):
                src_path = Path(band_paths[b])
                arr = np.fromfile(
                    src_path,
                    dtype=RAW_DTYPE,
                )
                if arr.size:
                    arr["entity"] += np.uint64(current)
                    with open(
                        raw_dir / f"band_{b:03d}.bin",
                        "ab",
                        buffering=8 * 1024 * 1024,
                    ) as dst:
                        arr.tofile(dst)

            current += count

    return current


def finalize(index_dir: Path, total_records: int):
    raw_dir = index_dir / "raw"
    band_dir = index_dir / "bands"

    for b in range(BANDS):
        raw_path = raw_dir / f"band_{b:03d}.bin"
        band_path = band_dir / f"band_{b:03d}.npy"

        actual = raw_path.stat().st_size // RAW_DTYPE.itemsize
        if actual != total_records:
            raise RuntimeError(
                f"Band {b}: expected {total_records:,}, got {actual:,}"
            )

        print(
            f"[sort] band {b+1}/{BANDS} | {actual:,} records",
            flush=True,
        )

        arr = np.fromfile(raw_path, dtype=RAW_DTYPE)
        arr.sort(order=["bucket", "entity"])
        np.save(band_path, arr)

        del arr
        raw_path.unlink()

    meta = index_dir / "metadata.txt"
    meta.write_text(
        "\n".join(
            [
                f"n_gram={N_GRAM}",
                f"threshold={THRESHOLD}",
                f"num_perm={NUM_PERM}",
                f"seed={SEED}",
                f"bands={BANDS}",
                f"rows={ROWS}",
                f"records={total_records}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index",
        default="blocking/minhash_test_index",
    )
    ap.add_argument(
        "--s2",
        default="dataset/test/test_source2.tsv",
    )
    ap.add_argument(
        "--s3",
        default="dataset/test/test_source3.tsv",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )
    args = ap.parse_args()

    index = Path(args.index).resolve()
    s2 = Path(args.s2).resolve()
    s3 = Path(args.s3).resolve()

    print("=" * 70, flush=True)
    print("FAST / RESUMABLE TEST MINHASH LSH", flush=True)
    print("=" * 70, flush=True)
    print(f"Index  : {index}", flush=True)
    print(f"Bands  : {BANDS} x {ROWS}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Batch  : {args.batch_size}", flush=True)

    index.mkdir(parents=True, exist_ok=True)

    existing, last_id = repair_partial(index)

    if existing:
        print(
            f"Existing partial records retained: {existing:,}",
            flush=True,
        )
        print(
            "Last ID:",
            last_id.decode("utf-8", errors="replace") if last_id else "none",
            flush=True,
        )
    else:
        print("No usable partial build; starting fresh.", flush=True)

    # Determine exact resume byte.
    if existing == 0:
        s2_start = 0
        s3_start = 0
        resume_source = 2
    elif last_id and last_id.startswith(b"S3-"):
        s2_start = s2.stat().st_size
        s3_start = find_after_id(s3, last_id)
        if s3_start is None:
            raise RuntimeError("Could not locate last S3 ID.")
        resume_source = 3
    else:
        s2_start = find_after_id(s2, last_id)
        if s2_start is None:
            raise RuntimeError("Could not locate last S2 ID.")
        s3_start = 0
        resume_source = 2

    temp = index / ".fast_worker_tmp"
    if temp.exists():
        shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True, exist_ok=True)

    tasks = []
    task_id = 0
    ranges_per_source = max(1, args.workers * 2)

    if resume_source == 2:
        for a, b in split_ranges(
            s2_start,
            s2.stat().st_size,
            ranges_per_source,
        ):
            tasks.append(
                (
                    2, task_id, str(s2), a, b,
                    str(temp), args.batch_size,
                )
            )
            task_id += 1

        for a, b in split_ranges(
            0,
            s3.stat().st_size,
            ranges_per_source,
        ):
            tasks.append(
                (
                    3, task_id, str(s3), a, b,
                    str(temp), args.batch_size,
                )
            )
            task_id += 1
    else:
        for a, b in split_ranges(
            s3_start,
            s3.stat().st_size,
            ranges_per_source,
        ):
            tasks.append(
                (
                    3, task_id, str(s3), a, b,
                    str(temp), args.batch_size,
                )
            )
            task_id += 1

    if not tasks:
        print("No remaining records to process.", flush=True)
        results = []
    else:
        print(
            f"Launching {args.workers} workers over {len(tasks)} ranges...",
            flush=True,
        )

        results = []
        started = time.time()

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(worker, t)
                for t in tasks
            ]

            for done, fut in enumerate(
                as_completed(futures),
                start=1,
            ):
                r = fut.result()
                results.append(r)
                print(
                    f"range {done}/{len(futures)} complete | "
                    f"valid={r[4]:,} | elapsed={(time.time()-started)/60:.1f}m",
                    flush=True,
                )

        new_count = sum(r[4] for r in results)

        append_task_outputs(
            index,
            results,
            existing,
        )

        print(
            f"New records: {new_count:,}",
            flush=True,
        )

    raw0 = index / "raw" / "band_000.bin"
    total_records = raw0.stat().st_size // RAW_DTYPE.itemsize

    print(
        f"Total records before sorting: {total_records:,}",
        flush=True,
    )

    finalize(index, total_records)

    print("Preparing query cache...", flush=True)
    QueryCache(str(index), BANDS).prepare()

    print("Creating entity_ids.npy...", flush=True)
    convert_entity_ids_to_npy(str(index), force=True)

    shutil.rmtree(temp, ignore_errors=True)

    elapsed = time.time() - (
        started if tasks else time.time()
    )

    print()
    print("=" * 70)
    print("BUILD COMPLETE", flush=True)
    print("=" * 70)
    print(f"Index   : {index}", flush=True)
    print(f"Records : {total_records:,}", flush=True)
    print(f"Layout  : {BANDS} bands x {ROWS} rows", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
