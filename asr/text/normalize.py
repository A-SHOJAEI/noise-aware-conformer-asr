from __future__ import annotations

import re


_allowed = re.compile(r"[^a-z' ]+")
_spaces = re.compile(r"\\s+")


def normalize_librispeech(text: str) -> str:
    # LibriSpeech transcripts are uppercase with punctuation; normalize consistently.
    t = text.strip().lower()
    t = t.replace("-", " ")
    t = _allowed.sub(" ", t)
    t = _spaces.sub(" ", t).strip()
    return t

