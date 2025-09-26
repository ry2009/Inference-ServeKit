from __future__ import annotations

import hashlib
import os
from typing import Tuple

from nacl import signing


def _load_signing_key() -> signing.SigningKey:
    key_hex = os.getenv("PRIMERL_VERIFIER_SK")
    if key_hex:
        key_bytes = bytes.fromhex(key_hex)
        return signing.SigningKey(key_bytes)
    # deterministic dev key; do not use in production
    seed = hashlib.blake2b(b"primerl-dev-key", digest_size=32).digest()
    return signing.SigningKey(seed)


def merkle_and_sign(trace: dict, reward: float) -> Tuple[str, str]:
    trace_digest = hashlib.blake2b(repr(trace).encode(), digest_size=32).digest()
    reward_digest = hashlib.blake2b(f"{reward:.6f}".encode(), digest_size=32).digest()
    root = hashlib.blake2b(trace_digest + reward_digest, digest_size=32).hexdigest()
    key = _load_signing_key()
    signature = key.sign(bytes.fromhex(root)).signature.hex()
    return root, signature
