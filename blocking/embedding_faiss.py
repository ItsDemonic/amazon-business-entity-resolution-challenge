"""
Streaming embedding-based candidate generation with FAISS.

Designed for large S2/S3 candidate pools (10M+ rows) and million-scale S1
queries without materializing the full candidate embeddings or query
embeddings in RAM.

Main improvements over the original implementation:
- Streams S2/S3 from TSV instead of loading/concatenating all rows.
- Builds one persistent FAISS index per country.
- Uses IVF-PQ for large countries to keep the index compact.
- Uses a bounded training sample instead of re-encoding the same records.
- Encodes with larger GPU batches and automatic CUDA OOM backoff.
- Uses half precision on CUDA when requested.
- Avoids Python dict conversion for every embedding batch.
- Stores candidate entity IDs in fixed-width memory-mapped arrays.
- Streams S1 queries and output in bounded chunks.
- Separates BUILD and QUERY so the expensive candidate index is built once
  and reused for train/test/dev queries.
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from common.normalize import normalize_address, normalize_name


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

DEFAULT_INDEX_DIR = "blocking/embedding_index"
DEFAULT_OUTPUT = "blocking/candidates_embed.tsv"

DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_BATCH_SIZE = 512
DEFAULT_QUERY_CHUNK_SIZE = 2_048

# IVF-PQ defaults.
DEFAULT_NLIST = 2048
DEFAULT_PQ_M = 32
DEFAULT_NPROBE = 32

# Training 100k vectors is large enough for a 2048-list coarse quantizer
# while remaining small compared with a 10M+ candidate pool.
DEFAULT_MAX_TRAINING_VECTORS = 100_000

# Small country buckets are cheaper and more accurate with exact search.
DEFAULT_FLAT_THRESHOLD = 100_000

# Entity IDs in this challenge are expected to be comfortably below this.
# The code validates this instead of silently truncating.
DEFAULT_ID_WIDTH = 64


def _import_faiss():
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required for embedding blocking. "
            "Install a compatible FAISS package first."
        ) from exc


def load_encoder(model_name=DEFAULT_MODEL, device=None, use_half=True):
    """Load SentenceTransformer and configure the CUDA path safely."""
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedding blocking. "
            "Install it with: python -m pip install sentence-transformers"
        ) from exc

    torch = None
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        pass

    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    encoder = sentence_transformers.SentenceTransformer(
        model_name,
        device=device,
    )

    # MiniLM uses a short context window. Do not increase it; longer sequence
    # lengths only make inference slower. This leaves the model's own limit
    # intact when it is already <= 128.
    try:
        current_max = int(encoder.max_seq_length)
        if current_max > 128:
            encoder.max_seq_length = 128
    except Exception:
        pass

    if torch is not None and str(device).startswith("cuda"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        if use_half:
            try:
                encoder.half()
            except Exception:
                # Some library versions/models may not expose a usable half()
                # path. Falling back to FP32 keeps correctness intact.
                pass

    try:
        encoder.eval()
    except Exception:
        pass

    print(f"Encoder device: {device}", flush=True)
    print(f"Encoder half precision requested: {use_half}", flush=True)
    return encoder


def prepare_texts(df):
    """Create embedding text directly from DataFrame columns."""
    names = df["business_name"].astype(str).tolist()
    addresses = df["business_address"].astype(str).tolist()

    texts = []
    append = texts.append

    for name, address in zip(names, addresses):
        normalized_name = normalize_name(name)
        normalized_address = normalize_address(address)
        append(f"{normalized_name} {normalized_address}".strip())

    return texts


def encode_texts(encoder, texts, batch_size=DEFAULT_BATCH_SIZE):
    """
    Encode normalized texts as contiguous float32 vectors.

    On CUDA, automatically backs off the batch size if an out-of-memory error
    occurs. This makes large-batch runs much safer on laptop GPUs.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    current_batch = max(1, int(batch_size))

    while True:
        try:
            embeddings = encoder.encode(
                texts,
                batch_size=current_batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return np.ascontiguousarray(
                embeddings,
                dtype=np.float32,
            )

        except RuntimeError as exc:
            message = str(exc).lower()

            is_cuda_oom = (
                "out of memory" in message
                and "cuda" in message
            )

            if not is_cuda_oom or current_batch <= 32:
                raise

            current_batch //= 2

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            print(
                f"CUDA OOM; retrying embedding batch_size={current_batch}",
                flush=True,
            )


def benchmark_encoder(encoder, rows, batch_size=DEFAULT_BATCH_SIZE):
    """Measure encoder throughput on a small sample."""
    if isinstance(rows, pd.DataFrame):
        texts = prepare_texts(rows)
    else:
        texts = list(rows)

    if not texts:
        raise ValueError("At least one record is required for a benchmark.")

    start = time.perf_counter()
    embeddings = encode_texts(
        encoder,
        texts,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - start

    return {
        "records": len(texts),
        "seconds": elapsed,
        "records_per_second": (
            len(texts) / elapsed if elapsed else float("inf")
        ),
        "dimensions": int(embeddings.shape[1]),
    }


def iter_source_chunks(source_paths, chunk_size):
    """Stream source TSVs without materializing them."""
    usecols = REQUIRED_COLUMNS

    for path in source_paths:
        print(f"Reading candidate source: {path}", flush=True)

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=usecols,
            chunksize=chunk_size,
        ):
            yield chunk


def load_query_sample(path):
    """Load one S1 query file into a bounded DataFrame when appropriate."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=REQUIRED_COLUMNS,
    )

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    return df


def stream_country_stats(
    source_paths,
    chunk_size,
    max_training_vectors,
):
    """
    First pass over S2/S3.

    Counts every candidate row and stores only a bounded deterministic
    training sample from the start of each country bucket.
    """
    country_counts = OrderedDict()
    training_samples = OrderedDict()

    for chunk in iter_source_chunks(source_paths, chunk_size):
        for country, group in chunk.groupby(
            "country",
            sort=False,
            dropna=False,
        ):
            country = str(country)
            count = len(group)

            country_counts[country] = (
                country_counts.get(country, 0) + count
            )

            existing = training_samples.get(country)

            if existing is None:
                existing = []
                training_samples[country] = existing

            remaining = max_training_vectors - len(existing)
            if remaining <= 0:
                continue

            sample = group.iloc[:remaining]

            sample_texts = prepare_texts(sample)
            existing.extend(sample_texts)

    return country_counts, training_samples


def country_key(country):
    """Stable filesystem-safe country key."""
    digest = hashlib.sha1(
        country.encode("utf-8")
    ).hexdigest()

    return digest[:16]


def make_index(
    training_vectors,
    candidate_count,
    nlist,
    pq_m,
    nprobe,
    flat_threshold,
):
    """Create and train an appropriate FAISS index."""
    faiss = _import_faiss()

    if training_vectors.ndim != 2 or training_vectors.shape[0] == 0:
        raise ValueError("Training vectors must be a non-empty 2D array.")

    dimension = int(training_vectors.shape[1])

    if candidate_count <= flat_threshold:
        index = faiss.IndexFlatIP(dimension)
        return index, "flat", 1, 1

    actual_m = min(int(pq_m), dimension)

    while actual_m > 1 and dimension % actual_m:
        actual_m -= 1

    if dimension % actual_m:
        raise ValueError(
            f"Embedding dimension {dimension} is incompatible "
            f"with pq_m={pq_m}."
        )

    training_count = training_vectors.shape[0]

    # Keep enough training vectors per coarse centroid. If the requested
    # nlist is too aggressive for the available sample, reduce it rather
    # than making FAISS train on an undersized sample.
    max_trainable_lists = max(
        1,
        training_count // 40,
    )

    actual_nlist = min(
        int(nlist),
        max(1, candidate_count),
        max_trainable_lists,
    )

    if actual_nlist <= 1:
        index = faiss.IndexFlatIP(dimension)
        return index, "flat", 1, 1

    quantizer = faiss.IndexFlatIP(dimension)

    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        actual_nlist,
        actual_m,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )

    training_vectors = np.ascontiguousarray(
        training_vectors,
        dtype=np.float32,
    )

    print(
        f"  Training IVF-PQ: nlist={actual_nlist}, "
        f"pq_m={actual_m}, training_vectors={training_count:,}",
        flush=True,
    )

    index.train(training_vectors)
    index.nprobe = min(int(nprobe), actual_nlist)

    # Precomputed lookup tables can speed CPU IVF-PQ search at the cost of
    # a modest amount of RAM. FAISS versions exposing this API support it.
    try:
        index.use_precomputed_table = 1
        index.precompute_table()
    except Exception:
        pass

    return index, "ivf_pq", actual_nlist, actual_m


def _write_metadata(index_dir, metadata):
    path = Path(index_dir) / "metadata.json"

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            metadata,
            fh,
            indent=2,
            ensure_ascii=False,
        )


def _read_metadata(index_dir):
    path = Path(index_dir) / "metadata.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing embedding index metadata: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as fh:
        return json.load(fh)


def _allocate_id_store(
    path,
    count,
    id_width,
):
    return np.memmap(
        path,
        dtype=f"S{id_width}",
        mode="w+",
        shape=(count,),
    )


def _open_id_store(
    path,
    count,
    id_width,
):
    return np.memmap(
        path,
        dtype=f"S{id_width}",
        mode="r",
        shape=(count,),
    )


def _validate_ids(ids, id_width):
    max_bytes = 0

    for value in ids:
        encoded = str(value).encode("utf-8")
        max_bytes = max(max_bytes, len(encoded))

        if len(encoded) > id_width:
            raise ValueError(
                f"Entity ID {value!r} needs {len(encoded)} bytes, "
                f"but --id-width={id_width}. Increase --id-width."
            )

    return max_bytes


def build_embedding_index(
    source2,
    source3,
    index_dir,
    encoder,
    metadata_model=DEFAULT_MODEL,
    chunk_size=DEFAULT_CHUNK_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    nlist=DEFAULT_NLIST,
    pq_m=DEFAULT_PQ_M,
    nprobe=DEFAULT_NPROBE,
    max_training_vectors=DEFAULT_MAX_TRAINING_VECTORS,
    flat_threshold=DEFAULT_FLAT_THRESHOLD,
    id_width=DEFAULT_ID_WIDTH,
):
    """
    Build a persistent, country-partitioned FAISS index.

    Two streaming passes are used:
      1. Count candidates and retain bounded training samples.
      2. Encode/add every candidate once and store aligned entity IDs.
    """
    source_paths = [source2, source3]
    index_dir = Path(index_dir)

    if index_dir.exists():
        print(
            f"Removing existing embedding index: {index_dir}",
            flush=True,
        )
        shutil.rmtree(index_dir)

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("PASS 1: COUNT + TRAINING SAMPLES")
    print("=" * 70)

    start = time.time()

    country_counts, training_samples = stream_country_stats(
        source_paths,
        chunk_size,
        max_training_vectors,
    )

    print()
    print("Candidate counts by country:", flush=True)
    for country, count in country_counts.items():
        print(
            f"  {country!r}: {count:,} candidates",
            flush=True,
        )

    print(
        f"Pass 1 time: {time.time() - start:.1f}s",
        flush=True,
    )

    # Build empty/trained per-country indexes before the streaming add pass.
    states = OrderedDict()
    metadata_countries = OrderedDict()

    print()
    print("=" * 70)
    print("TRAINING COUNTRY INDEXES")
    print("=" * 70)

    for country, candidate_count in country_counts.items():
        sample_texts = training_samples.get(country, [])

        if not sample_texts:
            raise RuntimeError(
                f"No training sample available for country {country!r}."
            )

        print(
            f"Country {country!r}: "
            f"{candidate_count:,} candidates / "
            f"{len(sample_texts):,} training vectors",
            flush=True,
        )

        training_vectors = encode_texts(
            encoder,
            sample_texts,
            batch_size=batch_size,
        )

        index, index_type, actual_nlist, actual_m = make_index(
            training_vectors,
            candidate_count,
            nlist,
            pq_m,
            nprobe,
            flat_threshold,
        )

        country_digest = country_key(country)

        index_path = index_dir / f"index_{country_digest}.faiss"
        ids_path = index_dir / f"ids_{country_digest}.bin"

        id_store = _allocate_id_store(
            ids_path,
            candidate_count,
            id_width,
        )

        states[country] = {
            "index": index,
            "id_store": id_store,
            "next_id": 0,
            "count": candidate_count,
            "index_path": index_path,
            "ids_path": ids_path,
            "index_type": index_type,
            "nlist": actual_nlist,
            "pq_m": actual_m,
            "nprobe": (
                int(index.nprobe)
                if hasattr(index, "nprobe")
                else 1
            ),
        }

        metadata_countries[country] = {
            "digest": country_digest,
            "count": candidate_count,
            "index_type": index_type,
            "nlist": actual_nlist,
            "pq_m": actual_m,
            "nprobe": states[country]["nprobe"],
            "index_file": index_path.name,
            "ids_file": ids_path.name,
        }

        del training_vectors

    # Release training text memory before the large add pass.
    training_samples.clear()

    print()
    print("=" * 70)
    print("PASS 2: STREAM + ENCODE + ADD")
    print("=" * 70)

    total_records = 0
    encoded_records = 0
    add_start = time.time()

    for chunk in iter_source_chunks(source_paths, chunk_size):
        for country, group in chunk.groupby(
            "country",
            sort=False,
            dropna=False,
        ):
            country = str(country)

            state = states.get(country)

            if state is None:
                continue

            texts = prepare_texts(group)

            vectors = encode_texts(
                encoder,
                texts,
                batch_size=batch_size,
            )

            ids = group["entity_id"].astype(str).tolist()

            _validate_ids(ids, id_width)

            start_pos = state["next_id"]
            end_pos = start_pos + len(ids)

            state["id_store"][start_pos:end_pos] = np.asarray(
                ids,
                dtype=f"S{id_width}",
            )

            state["index"].add(vectors)

            state["next_id"] = end_pos
            encoded_records += len(ids)

        total_records += len(chunk)

        if encoded_records and encoded_records % 100_000 < len(chunk):
            elapsed = time.time() - add_start
            rate = encoded_records / elapsed if elapsed else 0

            print(
                f"  encoded={encoded_records:,} "
                f"| rate={rate:.1f}/s",
                flush=True,
            )

    print(
        f"Encoded candidate records: {encoded_records:,}",
        flush=True,
    )

    for country, state in states.items():
        if state["next_id"] != state["count"]:
            raise RuntimeError(
                f"Country {country!r}: expected "
                f"{state['count']:,} IDs, stored {state['next_id']:,}."
            )

        state["id_store"].flush()

        faiss = _import_faiss()

        faiss.write_index(
            state["index"],
            str(state["index_path"]),
        )

        del state["index"]
        del state["id_store"]

    metadata = {
        "format_version": 2,
        "model": metadata_model,
        "id_width": id_width,
        "dimension": None,
        "countries": metadata_countries,
    }

    # Get the dimension from one persisted index.
    faiss = _import_faiss()

    for info in metadata_countries.values():
        loaded = faiss.read_index(
            str(index_dir / info["index_file"])
        )
        metadata["dimension"] = int(loaded.d)
        break

    _write_metadata(
        index_dir,
        metadata,
    )

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("EMBEDDING INDEX BUILD COMPLETE")
    print("=" * 70)
    print(f"Candidates indexed : {encoded_records:,}")
    print(f"Countries           : {len(metadata_countries)}")
    print(f"Build time           : {elapsed / 60:.2f} min")
    print(f"Index directory      : {index_dir}")
    print(flush=True)

    return metadata


class LoadedCountryIndex:
    """Persistent country index + memory-mapped entity IDs."""

    def __init__(
        self,
        country,
        index,
        id_store,
        info,
    ):
        self.country = country
        self.index = index
        self.id_store = id_store
        self.info = info


def load_embedding_indexes(index_dir):
    """Load all country indexes and ID stores once."""
    faiss = _import_faiss()
    index_dir = Path(index_dir)
    metadata = _read_metadata(index_dir)
    id_width = int(metadata["id_width"])

    countries = OrderedDict()

    for country, info in metadata["countries"].items():
        index_path = index_dir / info["index_file"]
        ids_path = index_dir / info["ids_file"]

        index = faiss.read_index(str(index_path))

        if hasattr(index, "nprobe"):
            index.nprobe = min(
                128,
                int(index.nlist),
            )

        id_store = _open_id_store(
            ids_path,
            int(info["count"]),
            id_width,
        )

        countries[country] = LoadedCountryIndex(
            country,
            index,
            id_store,
            info,
        )

    return metadata, countries


def _decode_id(value):
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8")
    return bytes(value).rstrip(b"\x00").decode("utf-8")


def search_country(
    country_index,
    query_vectors,
    top_k,
):
    """Search one country index and map positions back to entity IDs."""
    ntotal = int(country_index.index.ntotal)

    if ntotal == 0:
        return [[] for _ in range(len(query_vectors))]

    k = min(
        int(top_k),
        ntotal,
    )

    distances, positions = country_index.index.search(
        np.ascontiguousarray(
            query_vectors,
            dtype=np.float32,
        ),
        k,
    )

    results = []

    # FAISS already returns results sorted by descending similarity.
    # We retain that order to avoid an expensive Python sort for every row.
    for row_positions in positions:
        valid = row_positions >= 0

        if not np.any(valid):
            results.append([])
            continue

        raw_ids = country_index.id_store[
            row_positions[valid]
        ]

        results.append(
            [
                _decode_id(value)
                for value in raw_ids
            ]
        )

    return results


def query_embedding_index(
    index_dir,
    source1,
    output_path,
    encoder,
    top_k=200,
    chunk_size=DEFAULT_QUERY_CHUNK_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """Stream S1 queries through a persisted embedding index."""
    metadata, country_indexes = load_embedding_indexes(
        index_dir
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_queries = 0
    total_candidates = 0
    total_zero = 0

    start = time.time()

    print()
    print("=" * 70)
    print("STREAMING EMBEDDING QUERY")
    print("=" * 70)
    print(f"Index directory : {index_dir}")
    print(f"Top-k           : {top_k}")
    print(f"Query chunk     : {chunk_size}")
    print()

    with open(
        output_path,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as out:

        out.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        for chunk in pd.read_csv(
            source1,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=REQUIRED_COLUMNS,
            chunksize=chunk_size,
        ):
            # Use an explicit local row-position column instead of pandas
            # index labels.  This is robust even when a chunk carries an
            # inherited/global index.
            chunk = chunk.reset_index(drop=True)
            chunk["_local_position"] = np.arange(
                len(chunk),
                dtype=np.int64,
            )

            # Preserve S1 input order inside each bounded chunk.
            results = [None] * len(chunk)

            for country, group in chunk.groupby(
                "country",
                sort=False,
                dropna=False,
            ):
                country = str(country)

                country_index = country_indexes.get(country)
                positions = group["_local_position"].to_numpy(
                    dtype=np.int64,
                )

                if country_index is None:
                    for position in positions:
                        results[int(position)] = []
                    continue

                query_group = group[REQUIRED_COLUMNS]
                texts = prepare_texts(query_group)

                vectors = encode_texts(
                    encoder,
                    texts,
                    batch_size=batch_size,
                )

                country_results = search_country(
                    country_index,
                    vectors,
                    top_k,
                )

                if len(country_results) != len(positions):
                    raise RuntimeError(
                        f"FAISS returned {len(country_results)} result rows "
                        f"for {len(positions)} queries in country {country!r}."
                    )

                for position, candidates in zip(
                    positions,
                    country_results,
                ):
                    results[int(position)] = candidates

            for row, candidates in zip(
                chunk[REQUIRED_COLUMNS].itertuples(
                    index=False,
                    name=None,
                ),
                results,
            ):
                if candidates is None:
                    candidates = []

                s1_id = str(row[0])

                out.write(
                    s1_id
                    + "\t"
                    + ",".join(candidates)
                    + "\n"
                )

                total_queries += 1
                total_candidates += len(candidates)

                if not candidates:
                    total_zero += 1

            elapsed = time.time() - start
            rate = (
                total_queries / elapsed
                if elapsed
                else 0
            )

            print(
                f"  queried={total_queries:,} "
                f"| avg={total_candidates / total_queries:.2f} "
                f"| rate={rate:.1f}/s "
                f"| zero={total_zero:,}",
                flush=True,
            )

    elapsed = time.time() - start
    average = (
        total_candidates / total_queries
        if total_queries
        else 0
    )

    print()
    print("=" * 70)
    print("EMBEDDING QUERY COMPLETE")
    print("=" * 70)
    print(f"S1 queries       : {total_queries:,}")
    print(f"Total candidates : {total_candidates:,}")
    print(f"Average/query    : {average:.2f}")
    print(f"Zero candidates  : {total_zero:,}")
    print(f"Query time       : {elapsed:.1f}s")
    print(
        f"Query rate       : "
        f"{total_queries / elapsed:.2f} S1/s"
        if elapsed
        else "Query rate       : 0 S1/s"
    )
    print(f"Output            : {output_path}")
    print()


def build_and_query(args):
    encoder = load_encoder(
        model_name=args.model,
        device=args.device,
        use_half=not args.no_half,
    )

    build_embedding_index(
        source2=args.source2,
        source3=args.source3,
        index_dir=args.index_dir,
        encoder=encoder,
        metadata_model=args.model,
        chunk_size=args.candidate_chunk_size,
        batch_size=args.batch_size,
        nlist=args.nlist,
        pq_m=args.pq_m,
        nprobe=args.nprobe,
        max_training_vectors=args.max_training_vectors,
        flat_threshold=args.flat_threshold,
        id_width=args.id_width,
    )

    query_embedding_index(
        index_dir=args.index_dir,
        source1=args.source1,
        output_path=args.output,
        encoder=encoder,
        top_k=args.top_k,
        chunk_size=args.query_chunk_size,
        batch_size=args.batch_size,
    )


def build_only(args):
    encoder = load_encoder(
        model_name=args.model,
        device=args.device,
        use_half=not args.no_half,
    )

    build_embedding_index(
        source2=args.source2,
        source3=args.source3,
        index_dir=args.index_dir,
        encoder=encoder,
        metadata_model=args.model,
        chunk_size=args.candidate_chunk_size,
        batch_size=args.batch_size,
        nlist=args.nlist,
        pq_m=args.pq_m,
        nprobe=args.nprobe,
        max_training_vectors=args.max_training_vectors,
        flat_threshold=args.flat_threshold,
        id_width=args.id_width,
    )


def query_only(args):
    encoder = load_encoder(
        model_name=args.model,
        device=args.device,
        use_half=not args.no_half,
    )

    query_embedding_index(
        index_dir=args.index_dir,
        source1=args.source1,
        output_path=args.output,
        encoder=encoder,
        top_k=args.top_k,
        chunk_size=args.query_chunk_size,
        batch_size=args.batch_size,
    )


def benchmark_only(args):
    encoder = load_encoder(
        model_name=args.model,
        device=args.device,
        use_half=not args.no_half,
    )

    sample = load_query_sample(args.source1).head(
        args.benchmark_rows
    )

    print(
        benchmark_encoder(
            encoder,
            sample,
            batch_size=args.batch_size,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    action = parser.add_mutually_exclusive_group()

    action.add_argument(
        "--build-index",
        action="store_true",
        help="Build the persistent country-partitioned embedding index.",
    )

    action.add_argument(
        "--query",
        action="store_true",
        help="Query an existing embedding index.",
    )

    action.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Benchmark encoder throughput only.",
    )

    parser.add_argument(
        "--source1",
        required=True,
        help="S1 TSV used for query or encoder benchmark.",
    )

    parser.add_argument(
        "--source2",
        help="Full S2 candidate TSV, required for --build-index.",
    )

    parser.add_argument(
        "--source3",
        help="Full S3 candidate TSV, required for --build-index.",
    )

    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Persistent embedding index directory.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Candidate output TSV.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device, e.g. cuda or cpu. Default: auto.",
    )

    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Do not use half precision on CUDA.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="SentenceTransformer batch size.",
    )

    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows streamed from S2/S3 at once.",
    )

    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=DEFAULT_QUERY_CHUNK_SIZE,
        help="Rows streamed from S1 at once.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Candidates returned per S1 query.",
    )

    parser.add_argument(
        "--nlist",
        type=int,
        default=DEFAULT_NLIST,
        help="Maximum IVF coarse clusters.",
    )

    parser.add_argument(
        "--pq-m",
        type=int,
        default=DEFAULT_PQ_M,
        help="Number of PQ subquantizers.",
    )

    parser.add_argument(
        "--nprobe",
        type=int,
        default=DEFAULT_NPROBE,
        help="IVF clusters searched per query.",
    )

    parser.add_argument(
        "--max-training-vectors",
        type=int,
        default=DEFAULT_MAX_TRAINING_VECTORS,
        help="Maximum embedding vectors used to train each country index.",
    )

    parser.add_argument(
        "--flat-threshold",
        type=int,
        default=DEFAULT_FLAT_THRESHOLD,
        help="Country sizes at or below this use exact IndexFlatIP.",
    )

    parser.add_argument(
        "--id-width",
        type=int,
        default=DEFAULT_ID_WIDTH,
        help="Fixed byte width reserved for each entity ID.",
    )

    parser.add_argument(
        "--benchmark-rows",
        type=int,
        default=1000,
        help="Rows used by --benchmark-only.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")

    if args.candidate_chunk_size < 1:
        parser.error("--candidate-chunk-size must be >= 1.")

    if args.query_chunk_size < 1:
        parser.error("--query-chunk-size must be >= 1.")

    if args.top_k < 1:
        parser.error("--top-k must be >= 1.")

    if args.nlist < 1:
        parser.error("--nlist must be >= 1.")

    if args.pq_m < 1:
        parser.error("--pq-m must be >= 1.")

    if args.nprobe < 1:
        parser.error("--nprobe must be >= 1.")

    if args.max_training_vectors < 1:
        parser.error("--max-training-vectors must be >= 1.")

    if args.flat_threshold < 0:
        parser.error("--flat-threshold must be >= 0.")

    if args.id_width < 1:
        parser.error("--id-width must be >= 1.")

    if args.build_index or not (
        args.query
        or args.benchmark_only
    ):
        if not args.source2:
            parser.error("--source2 is required for index building.")

        if not args.source3:
            parser.error("--source3 is required for index building.")

        build_only(args)

        if args.build_index:
            return

        # Default behavior: build once, then immediately query.
        query_only(args)
        return

    if args.query:
        query_only(args)
        return

    if args.benchmark_only:
        benchmark_only(args)
        return


if __name__ == "__main__":
    main()
