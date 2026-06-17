"""On-disk cache for LLM calls, keyed by a hash of the full request.

Every paid call is cached so re-running eval costs nothing after the first pass.
The key covers model + messages + generation params, so changing any of them
yields a fresh entry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def cache_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = Path(self._path(key))
        return json.loads(p.read_text()) if p.exists() else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        Path(self._path(key)).write_text(json.dumps(value, ensure_ascii=False, indent=2))
