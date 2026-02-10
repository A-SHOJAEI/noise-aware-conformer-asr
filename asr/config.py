from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_scalar(s: str) -> Any:
    sl = s.strip().lower()
    if sl in {"true", "false"}:
        return sl == "true"
    if sl in {"null", "none"}:
        return None
    try:
        if "." in sl or "e" in sl:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _set_by_dotted_key(d: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_yaml_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data)} at {path}")

    if overrides:
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Override must look like key=value, got: {ov}")
            k, v = ov.split("=", 1)
            _set_by_dotted_key(data, k.strip(), _parse_scalar(v))
    return data


def save_yaml_config(path: str | Path, cfg: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def require(cfg: Dict[str, Any], key: str) -> Any:
    v = get(cfg, key, default=None)
    if v is None:
        raise KeyError(f"Missing required config key: {key}")
    return v

