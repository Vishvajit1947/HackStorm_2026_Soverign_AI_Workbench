"""Config loading. Everything is a local file; nothing here may resolve a hostname."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    # services/kavach/config.py -> repo root
    return Path(__file__).resolve().parents[2]


# Settings that describe THIS MACHINE, read once from a file beside the repo.
#
# Where the weights are, where llama-server is, which backend to use: three
# facts that do not change between launches and that the operator should not
# have to remember. Passing them as environment variables meant every command
# that forgot one silently ran against an empty models folder and reported
# fourteen models as "file not on this PC" - which is a true statement about a
# misconfiguration and a very confusing one to read.
#
# Deliberately not python-dotenv: this is twelve lines, and the offline install
# already has enough wheels to stage.
#
# The real environment always WINS over the file, so a one-off
# `KAVACH_PROFILE=LOW ...` still works and CI is unaffected by a developer's
# local file.
ENV_FILE = "kavach.env"


def _load_env_file(root: Path) -> None:
    path = root / ENV_FILE
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Only KAVACH_* keys, so a stray line in this file can never reach
        # PATH, PYTHONPATH or anything else the process depends on.
        if not key.startswith("KAVACH_"):
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Profile:
    name: str
    description: str
    vram_gb: float
    reserve_gb: float
    cpu_only: bool
    context: int
    pinned: list[str]
    available: list[str]
    swap_slots: int = 1
    ram_gb: float = 0.0          # CPU-only profiles budget against system RAM

    @property
    def usable_vram_gb(self) -> float:
        """What the warden is allowed to hand out, after the fragmentation reserve.

        On a CPU-only profile there is no VRAM at all, so the budget comes from
        ram_gb instead. Returning 0.0 here would mean nothing could ever be
        loaded, which would break the one profile whose job is to keep working
        when the GPU does not.
        """
        if self.cpu_only:
            return max(0.0, self.ram_gb - self.reserve_gb)
        return max(0.0, self.vram_gb - self.reserve_gb)


@dataclass
class Settings:
    profile_name: str
    backend: str
    var_dir: Path
    model_dir: Path
    config_dir: Path
    enforce_digests: bool
    host: str
    port: int
    profile: Profile
    policy: dict[str, Any] = field(default_factory=dict)
    # Where llama-server lives, when it is not on PATH. The warden spawns one
    # process per resident model, so a binary it cannot find is the difference
    # between real tokens and none at all - and "not on PATH" is the single
    # most common reason a correctly staged box still answers from the mock.
    llama_bin_dir: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.var_dir / "kavach.db"

    @property
    def workspace_root(self) -> Path:
        return self.var_dir / "workspaces"

    def ensure_dirs(self) -> None:
        for p in (self.var_dir, self.workspace_root, self.model_dir):
            p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = _repo_root()
    _load_env_file(root)
    config_dir = Path(_env("KAVACH_CONFIG_DIR", str(root / "config")))

    profile_name = _env("KAVACH_PROFILE", "MID").upper()
    profiles = _load_yaml(config_dir / "profiles.yaml")
    if profile_name not in profiles:
        raise ValueError(
            f"unknown profile {profile_name!r}; have {sorted(profiles)}"
        )
    raw = profiles[profile_name]
    profile = Profile(
        name=profile_name,
        description=raw.get("description", ""),
        vram_gb=float(raw["vram_gb"]),
        reserve_gb=float(raw.get("reserve_gb", 0.0)),
        cpu_only=bool(raw.get("cpu_only", False)),
        context=int(raw.get("context", 8192)),
        pinned=list(raw.get("pinned", [])),
        available=list(raw.get("available", [])),
        swap_slots=int(raw.get("swap_slots", 1)),
        ram_gb=float(raw.get("ram_gb", 0.0)),
    )

    settings = Settings(
        profile_name=profile_name,
        backend=_env("KAVACH_BACKEND", "mock").lower(),
        var_dir=Path(_env("KAVACH_VAR_DIR", str(root / "var"))).resolve(),
        model_dir=Path(_env("KAVACH_MODEL_DIR", str(root / "models"))).resolve(),
        config_dir=config_dir,
        enforce_digests=_env_bool("KAVACH_ENFORCE_DIGESTS", True),
        host=_env("KAVACH_HOST", "127.0.0.1"),
        port=int(_env("KAVACH_PORT", "8080")),
        profile=profile,
        policy=_load_yaml(config_dir / "policy.yaml"),
        llama_bin_dir=(
            Path(raw_bin).resolve() if (raw_bin := _env("KAVACH_LLAMA_BIN_DIR", ""))
            else None
        ),
    )
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    """Tests and `POST /api/models/reload` need to re-read config from disk."""
    get_settings.cache_clear()
