# common/normalize.py

import re
import unicodedata


def normalize_text(text):
    """
    General normalization shared by all blocking/matching methods.

    - Convert missing values to ""
    - Unicode normalize
    - Lowercase
    - Normalize punctuation/separators to spaces
    - Collapse repeated whitespace
    """
    if text is None:
        return ""

    text = str(text)

    if not text:
        return ""

    # Unicode canonicalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation/symbols with spaces.
    # Keep letters and numbers from all scripts.
    text = "".join(
        ch if (ch.isalnum() or ch.isspace()) else " "
        for ch in text
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(name):
    """
    Normalize a business name.
    """
    return normalize_text(name)


def normalize_address(address):
    """
    Normalize a business address.
    """
    return normalize_text(address)


def tokenize(text):
    """
    Tokenize already-normalized text.
    """
    if not text:
        return []

    return text.split()


def name_tokens(name):
    """
    Normalize + tokenize a business name.
    """
    return tokenize(normalize_name(name))


def address_tokens(address):
    """
    Normalize + tokenize a business address.
    """
    return tokenize(normalize_address(address))


def normalized_record(business_name, business_address):
    """
    Convenience helper used by blocking methods.
    """
    name = normalize_name(business_name)
    address = normalize_address(business_address)

    return {
        "name": name,
        "address": address,
        "name_tokens": tokenize(name),
        "address_tokens": tokenize(address),
    }