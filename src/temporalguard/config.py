"""Config + environment helpers shared across the project."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(rel: str) -> Path:
    """Resolve a path relative to the repo root (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(rel_or_abs_path: str) -> Dict[str, Any]:
    with open(repo_path(rel_or_abs_path)) as f:
        return yaml.safe_load(f) or {}


def load_configs() -> Dict[str, Dict[str, Any]]:
    """Load all project configs into one dict keyed by name."""
    return {
        "app": load_yaml("configs/app.yaml"),
        "retrieval": load_yaml("configs/retrieval.yaml"),
        "prompts": load_yaml("configs/prompts.yaml"),
        "source_authority": load_yaml("configs/source_authority.yaml"),
    }


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ (no overwrite)."""
    p = repo_path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
