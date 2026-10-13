"""knowledge.yaml loading.

Separate from kavach.config because it is live-reloadable: the chunk size and the
retrieval knobs are exactly the things you want to change between rehearsals
without restarting the gateway, in the same way prototypes.yaml is for the
router.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

from ..config import get_settings

_config: dict[str, Any] | None = None
_lock = threading.Lock()


def config_path() -> Path:
    return get_settings().config_dir / "knowledge.yaml"


def load_knowledge_config(*, force: bool = False) -> dict[str, Any]:
    global _config
    if _config is None or force:
        with _lock:
            if _config is None or force:
                path = config_path()
                if not path.exists():
                    raise FileNotFoundError(f"config file missing: {path}")
                with path.open("r", encoding="utf-8") as fh:
                    _config = yaml.safe_load(fh) or {}
    return _config


def reload_knowledge_config() -> dict[str, Any]:
    return load_knowledge_config(force=True)


def reset_knowledge_config() -> None:
    """Tests only."""
    global _config
    _config = None
