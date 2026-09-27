"""
High-signal pairwise similarity features.

The feature layer is deliberately split into:
- normalized-string features
- token features
- numeric/address-structure features
- blocker-rank features

All normalization/tokenization comes from common.normalize.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rapidfuzz import fuzz

from common.normalize import (
    address_tokens,
    name_tokens,
    normalize_address,
    normalize_name,
)


FEATURE_NAMES = [
    "name_levenshtein",
    "address_levenshtein",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "address_token_set_ratio",
    "address_token_sort_ratio",
    "name_partial_ratio",
    "address_partial_ratio",
    "name_wratio",
    "address_wratio",
    "name_jaccard",
    "address_jaccard",
    "name_token_overlap",
    "address_token_overlap",
    "name_exact",
    "address_exact",
    "name_length_ratio",
    "address_length_ratio",
    "name_digit_jaccard",
    "address_digit_jaccard",
    "address_number_match",
    "address_contains",
    "name_first_char_match",
    "country_match",
    "address_missing_query",
    "address_missing_candidate",
    "blocker_rank_score",
    "blocker_rank_log",
    "blocker_source_count",
    "blocker_p1_hit",
    "blocker_p2_hit",
    "blocker_p3_hit",
]


_DIGIT_RE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    name: str
    address: str
    name_tokens: tuple[str, ...]
    address_tokens: tuple[str, ...]
    country: str
    name_digits: frozenset[str]
    address_digits: frozenset[str]


def prepare_record(
    name: str,
    address: str,
    country: str,
) -> PreparedRecord:
    nn = normalize_name(name)
    na = normalize_address(address)
    nt = tuple(name_tokens(name))
    at = tuple(address_tokens(address))

    return PreparedRecord(
        name=nn,
        address=na,
        name_tokens=nt,
        address_tokens=at,
        country=(country or "").strip().casefold(),
        name_digits=frozenset(_DIGIT_RE.findall(nn)),
        address_digits=frozenset(_DIGIT_RE.findall(na)),
    )


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa = set(a)
    bb = set(b)
    if not aa and not bb:
        return 1.0
    union = aa | bb
    return len(aa & bb) / len(union) if union else 0.0


def _overlap(a: Iterable[str], b: Iterable[str]) -> float:
    aa = set(a)
    bb = set(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / min(len(aa), len(bb))


def _safe_ratio(a: str, b: str, scorer) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(scorer(a, b)) / 100.0


def _length_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def pair_features_prepared(
    query: PreparedRecord,
    candidate: PreparedRecord,
    *,
    blocker_rank: int = 0,
    source_count: int = 0,
    p1_hit: int = 0,
    p2_hit: int = 0,
    p3_hit: int = 0,
) -> np.ndarray:
    qn, cn = query.name, candidate.name
    qa, ca = query.address, candidate.address

    rank_score = 1.0 / (60.0 + float(blocker_rank)) if blocker_rank > 0 else 0.0
    rank_log = 1.0 / np.log2(2.0 + float(blocker_rank)) if blocker_rank > 0 else 0.0

    return np.asarray(
        [
            _safe_ratio(qn, cn, fuzz.ratio),
            _safe_ratio(qa, ca, fuzz.ratio),
            _safe_ratio(
                " ".join(query.name_tokens),
                " ".join(candidate.name_tokens),
                fuzz.token_set_ratio,
            ),
            _safe_ratio(
                " ".join(query.name_tokens),
                " ".join(candidate.name_tokens),
                fuzz.token_sort_ratio,
            ),
            _safe_ratio(
                " ".join(query.address_tokens),
                " ".join(candidate.address_tokens),
                fuzz.token_set_ratio,
            ),
            _safe_ratio(
                " ".join(query.address_tokens),
                " ".join(candidate.address_tokens),
                fuzz.token_sort_ratio,
            ),
            _safe_ratio(qn, cn, fuzz.partial_ratio),
            _safe_ratio(qa, ca, fuzz.partial_ratio),
            _safe_ratio(qn, cn, fuzz.WRatio),
            _safe_ratio(qa, ca, fuzz.WRatio),
            _jaccard(query.name_tokens, candidate.name_tokens),
            _jaccard(query.address_tokens, candidate.address_tokens),
            _overlap(query.name_tokens, candidate.name_tokens),
            _overlap(query.address_tokens, candidate.address_tokens),
            float(bool(qn) and bool(cn) and qn == cn),
            float(bool(qa) and bool(ca) and qa == ca),
            _length_ratio(qn, cn),
            _length_ratio(qa, ca),
            _jaccard(query.name_digits, candidate.name_digits),
            _jaccard(query.address_digits, candidate.address_digits),
            float(
                bool(query.address_digits)
                and bool(candidate.address_digits)
                and bool(query.address_digits & candidate.address_digits)
            ),
            float(
                bool(qa)
                and bool(ca)
                and (qa in ca or ca in qa)
            ),
            float(bool(qn) and bool(cn) and qn[0] == cn[0]),
            float(query.country == candidate.country),
            float(not qa),
            float(not ca),
            rank_score,
            rank_log,
            float(source_count),
            float(p1_hit),
            float(p2_hit),
            float(p3_hit),
        ],
        dtype=np.float32,
    )


def pair_features(
    query_name: str,
    query_address: str,
    query_country: str,
    candidate_name: str,
    candidate_address: str,
    candidate_country: str,
    *,
    blocker_rank: int = 0,
    source_count: int = 0,
    p1_hit: int = 0,
    p2_hit: int = 0,
    p3_hit: int = 0,
) -> np.ndarray:
    return pair_features_prepared(
        prepare_record(query_name, query_address, query_country),
        prepare_record(candidate_name, candidate_address, candidate_country),
        blocker_rank=blocker_rank,
        source_count=source_count,
        p1_hit=p1_hit,
        p2_hit=p2_hit,
        p3_hit=p3_hit,
    )


def compute_similarity_features(*args, **kwargs) -> dict[str, float]:
    values = pair_features(*args, **kwargs)
    return dict(zip(FEATURE_NAMES, values.tolist()))
