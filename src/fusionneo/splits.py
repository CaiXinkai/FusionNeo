from __future__ import annotations

import hashlib


def stable_group_split(group: str, seed: int = 42) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "validation"
    return "test"

