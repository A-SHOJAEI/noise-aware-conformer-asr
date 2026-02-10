from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def git_commit_hash(repo_root: str | Path = ".") -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None

