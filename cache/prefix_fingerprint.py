import hashlib


def normalize(text: str) -> str:
    """Whitespace-normalize prompts prior to fingerprinting."""
    return " ".join(text.strip().split())


def rolling_hash(text: str, n: int = 5) -> bytes:
    """Compute a rolling n-gram hash for prefix cache fingerprinting."""
    grams = [text[i : i + n] for i in range(max(0, len(text) - n + 1))]
    digest = hashlib.blake2b(digest_size=16)
    for gram in grams:
        digest.update(gram.encode())
    return digest.digest()
