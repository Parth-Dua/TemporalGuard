"""JSONL read/write helpers. ``iter_jsonl`` streams so large files (the bench
chunk sidecar can be millions of lines) never load fully into memory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
