from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CharTokenizer:
    # CTC vocabulary:
    # 0: blank
    # 1: space token "|"
    # 2..: letters + apostrophe
    blank: str = "<blk>"
    space_token: str = "|"

    def __post_init__(self) -> None:
        pass

    @property
    def vocab(self) -> List[str]:
        letters = list("abcdefghijklmnopqrstuvwxyz")
        extra = ["'"]
        return [self.blank, self.space_token] + letters + extra

    @property
    def token_to_id(self) -> Dict[str, int]:
        return {t: i for i, t in enumerate(self.vocab)}

    @property
    def id_to_token(self) -> Dict[int, str]:
        return {i: t for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        text = text.strip()
        ids: List[int] = []
        m = self.token_to_id
        for ch in text:
            if ch == " ":
                ids.append(m[self.space_token])
            elif ch in m:
                ids.append(m[ch])
            else:
                # Skip unsupported chars; normalization should have removed them.
                continue
        return ids

    def decode_ctc(self, ids: List[int]) -> str:
        # ids already collapsed/filtered.
        t = []
        inv = self.id_to_token
        for i in ids:
            tok = inv.get(int(i), "")
            if tok == self.space_token:
                t.append(" ")
            elif tok and tok != self.blank:
                t.append(tok)
        return "".join(t).strip()

