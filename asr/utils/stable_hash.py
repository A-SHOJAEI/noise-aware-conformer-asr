from __future__ import annotations

import hashlib


def stable_int_from_str(s: str, modulo: int = 2**31 - 1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], byteorder="big", signed=False)
    return int(v % modulo)

