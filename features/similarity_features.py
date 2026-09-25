from common.normalize import normalize_name
from common.normalize import normalize_address
from common.normalize import name_tokens
from common.normalize import address_tokens
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_similarity(tokens_a, tokens_b):
    """
    jaccard similarity between two token collections
    """

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)

def token_overlap(tokens_a, tokens_b):
    """
    fraction of tokens in a that also appear in b
    """

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)

def country_match(country_a, country_b):
    """
    1 if countries match, 0 otherwise
    """

    if not country_a or not country_b:
        return 0.0
    return float(country_a == country_b)

def missing_address(address):
    """
    1 if address is empty/missing, 0 otherwise
    """
    normalized = normalize_address(address)

    return float(not normalized)

def levenshtein_similarity(text_a, text_b):
    """
    Normalized edit similarity between 0 and 1
    """

    text_a = str(text_a or "")
    text_b = str(text_b or "")

    if not text_a and not text_b:
        return 1.0
    
    if not text_a or not text_b:
        return 0.0
    
    return ratio(text_a, text_b) / 100.0

def name_levenshtein(name_a, name_b):
    return levenshtein_similarity(
        normalize_name(name_a),
        normalize_name(name_b)
    )

def address_levenshtein(address_a, address_b):
    return levenshtein_similarity(
        normalize_address(address_a),
        normalize_address(address_b)
    )

def name_jaccard(name_a, name_b):
    return jaccard_similarity(
        name_tokens(name_a),
        name_tokens(name_b)
    )

def address_jaccard(address_a, address_b):
    return jaccard_similarity(
        address_tokens(address_a),
        address_tokens(address_b)
    )

def name_token_overlap(name_a, name_b):
    return token_overlap(
        name_tokens(name_a),
        name_tokens(name_b)
    )

def address_token_overlap(address_a, address_b):
    return token_overlap(
        address_tokens(address_a),
        address_tokens(address_b),
    )

class TfidfSimilarity:
    """
    Reusable TF-IDF representation for candidate similarity
    """

    def __init__(self, analyzer="word", ngram_range=(1, 2), min_df=2):
        
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df)

        self.matrix = None

    def fit(self, texts):
        """
        Fit TF-IDF on a corpus
        """

        self.matrix = self.vectorizer.fit_transform(texts)

        return self
    
    def transform(self, texts):
        """
        Transforms texts using fitted TF-IDF vocabulary
        """

        if self.matrix is None:
            raise RuntimeError("TF-IDF similarity must be fitted first")
        
        return self.vectorizer.transform(texts)
    
    def similarity(self, text_a, text_b):
        """
        Cosine similarity between 2 texts
        """

        vectors = self.vectorizer.transform([text_a, text_b])

        return float(cosine_similarity(vectors[0], vectors[1])[0, 0])

def compute_similarity_features(source_record, candidate_record):
    """
    Compute pairwise features for one S1 record
    and one S2/S3 candidate.
    """

    name_a = source_record["business_name"]
    name_b = candidate_record["business_name"]

    address_a = source_record["business_address"]
    address_b = candidate_record["business_address"]

    country_a = source_record["country"]
    country_b = candidate_record["country"]

    return {
        "name_levenshtein": name_levenshtein(name_a, name_b),

        "address_levenshtein": address_levenshtein(address_a, address_b),

        "name_jaccard": name_jaccard(name_a, name_b),

        "address_jaccard": address_jaccard(address_a, address_b),

        "name_token_overlap": name_token_overlap(name_a, name_b),

        "address_token_overlap": address_token_overlap(address_a, address_b),

        "country_match": country_match(country_a, country_b,),

        "address_missing_a": missing_address(address_a),

        "address_missing_b": missing_address(address_b),
    }

if __name__ == "__main__":

    record_a = {
        "business_name": "Amazon Technologies Pvt Ltd",
        "business_address": "123 MG Road Bangalore",
        "country": "India",
    }

    record_b = {
        "business_name": "Amazon Technology Private Limited",
        "business_address": "123 MG Road Bengaluru",
        "country": "India",
    }

    name_corpus = [
    "Amazon Technologies Pvt Ltd",
    "Amazon Technology Private Limited",
    "Amazon Fresh Store",
    "Flipkart Internet Private Limited",
    "Reliance Industries Limited",
    ]

    address_corpus = [
        "123 MG Road Bangalore",
        "123 MG Roag Bengaluru",
        "45 MG Road Bangalore",
        "221B Baker Street Delhi",
        "10 Park Street Kolkata",
    ]

    # Create TF-IDF objects
    name_tfidf = TfidfSimilarity()
    address_tfidf = TfidfSimilarity()

    # Fit them on a small example corpus
    name_tfidf.fit(name_corpus)

    address_tfidf.fit(address_corpus)

    # Calculate TF-IDF cosine similarities
    name_tfidf_score = name_tfidf.similarity(
        record_a["business_name"],
        record_b["business_name"],
    )

    address_tfidf_score = address_tfidf.similarity(
        record_a["business_address"],
        record_b["business_address"],
    )

    # Calculate all the other features
    features = compute_similarity_features(
        record_a,
        record_b,
    )

    # Add TF-IDF features
    features["name_tfidf_cosine"] = name_tfidf_score
    features["address_tfidf_cosine"] = address_tfidf_score

    # Print everything
    for name, value in features.items():
        print(f"{name}: {value:.4f}")
