"""Adding a model to the box, without a redesign (SS 26117, L1).

The brief asks that new open-weight models be addable later "without redesigning
the system, since this space is moving fast". The registry has always been able
to do that from a YAML entry; what was missing was the path an operator can
actually walk: get a file onto the machine and have it become a routing
candidate, with the same provenance rules as every model already there.

FOUR THINGS THIS DOES, AND WHY EACH ONE IS NOT OPTIONAL

  IT READS THE FILE BEFORE BELIEVING IT. `inspect_gguf` parses the header
  offline - magic, version, and the metadata keys llama.cpp itself reads. A
  file that is not a GGUF is refused here rather than at the first request,
  and the architecture tag is reported because that is what decides whether
  this build can load it at all. That check is not hypothetical: the staged
  bge-m3 embedder in this repo is tagged `xlmr`, which mainline llama.cpp does
  not implement, and it was discovered by a failing load rather than by
  looking.

  IT REGISTERS IN BOTH PLACES. A model is gated TWICE: `config/models.yaml`
  makes it exist, and a profile's `available:` list decides whether the router
  may consider it. Writing only the first produces a model that is present,
  verified, and never chosen, with no error anywhere. So `register` does both
  or neither.

  IT EDITS CONFIG BY LINE, NEVER BY ROUND-TRIP. `yaml.safe_dump` on models.yaml
  destroys every comment in it - including the one saying the quality numbers
  are unmeasured placeholders, which is the single comment this project cannot
  afford to lose. Same rule as `scripts/stage_model.py:write_digests`.

  IT RECORDS WHAT IT GUESSED. VRAM and speed are ESTIMATED from the file, and
  the quality numbers are seeded, and the entry it writes says so in the file
  itself. An estimate that looks like a measurement is the failure this whole
  build is organised against.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types, from the format spec. Only the ones that can
# appear in the keys below are decoded; anything else is skipped by length.
(_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64,
 _F64) = range(13)

_FIXED = {
    _U8: ("<B", 1), _I8: ("<b", 1), _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4), _F32: ("<f", 4), _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

# A header this large is not a header. Real ones are tens of kilobytes; the cap
# stops a malformed or hostile file from walking us through gigabytes of
# "metadata" one field at a time.
MAX_HEADER_BYTES = 32 * 1024 * 1024
MAX_KV_PAIRS = 4096

# Architectures this build's llama.cpp is known to load. Anything else is
# reported, not refused - a newer llama.cpp may well handle it, and refusing a
# model because a list in this file is out of date would be exactly the kind of
# hardcoding the brief asks us to avoid.
KNOWN_ARCHITECTURES = {
    "llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "phi2", "phi3",
    "gemma", "gemma2", "gemma3", "mistral", "mixtral", "starcoder2",
    "stablelm", "bert", "nomic-bert", "gpt2", "falcon", "deepseek2",
    "command-r", "olmo", "granite", "internlm2", "minicpm", "orion",
}


class StagingError(RuntimeError):
    """The file cannot be staged. Never a half-written manifest entry."""


# ------------------------------------------------------------------ GGUF

class _Reader:
    def __init__(self, handle, limit: int):
        self.handle = handle
        self.limit = limit
        self.read = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.read + count > self.limit:
            raise StagingError(
                "the GGUF header is larger than this reader will follow; the "
                "file is malformed or not a model"
            )
        data = self.handle.read(count)
        if len(data) != count:
            raise StagingError("the file ends in the middle of its header")
        self.read += count
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > 1 << 20:
            raise StagingError("a metadata string in the header is implausibly long")
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, kind: int) -> Any:
        if kind in _FIXED:
            fmt, size = _FIXED[kind]
            return struct.unpack(fmt, self.take(size))[0]
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            inner = self.u32()
            count = self.u64()
            if count > 1 << 22:
                raise StagingError("a metadata array in the header is implausibly long")
            # Arrays are the tokeniser vocabulary in practice. Skipped by
            # length rather than decoded - nothing here needs 150k tokens, and
            # materialising them would cost more than reading the whole header.
            if inner in _FIXED:
                self.take(_FIXED[inner][1] * count)
                return f"<{count} values>"
            if inner == _STRING:
                for _ in range(count):
                    self.string()
                return f"<{count} strings>"
            raise StagingError(f"unsupported array element type {inner}")
        raise StagingError(f"unsupported metadata value type {kind}")


@dataclass
class GgufInfo:
    """What the file says about itself. `warnings` is the honest half."""

    path: Path
    size_bytes: int
    version: int
    tensor_count: int
    architecture: str = ""
    name: str = ""
    context_length: int = 0
    embedding_length: int = 0
    quantisation: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024 ** 3), 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.path.name,
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "gguf_version": self.version,
            "tensors": self.tensor_count,
            "architecture": self.architecture,
            "name": self.name,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "quantisation": self.quantisation,
            "warnings": self.warnings,
        }


def inspect_gguf(path: Path) -> GgufInfo:
    """Read a GGUF header. Raises StagingError if this is not one.

    Deliberately strict about the magic and lenient about everything after it:
    a file that says GGUF and then disagrees with us about a metadata key is
    still probably a model, and the fields we could not read come back empty
    with a warning rather than taking the staging down.
    """
    if not path.is_file():
        raise StagingError(f"{path.name} is not a file")

    size = path.stat().st_size
    if size < 32:
        raise StagingError(f"{path.name} is too small to be a model file")

    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != GGUF_MAGIC:
            raise StagingError(
                f"{path.name} is not a GGUF model file (it starts with "
                f"{magic!r}, not {GGUF_MAGIC!r}). This box loads GGUF weights - "
                "convert or download the .gguf build of the model."
            )
        reader = _Reader(handle, min(MAX_HEADER_BYTES, size))
        version = reader.u32()
        tensor_count = reader.u64()
        kv_count = reader.u64()

        info = GgufInfo(path=path, size_bytes=size, version=version,
                        tensor_count=tensor_count)

        if version not in (2, 3):
            info.warnings.append(
                f"this file is GGUF v{version}; this build has been used with "
                "v2 and v3, so loading it is untested here"
            )
        if kv_count > MAX_KV_PAIRS:
            raise StagingError(
                f"{path.name} declares {kv_count} metadata entries, which is "
                "not a plausible model header"
            )

        metadata: dict[str, Any] = {}
        try:
            for _ in range(kv_count):
                key = reader.string()
                kind = reader.u32()
                metadata[key] = reader.value(kind)
        except StagingError as exc:
            # The fields we already have are still worth reporting. A header we
            # could only half read is a warning, not a reason to refuse a file
            # whose first 4 bytes were right.
            info.warnings.append(f"the header could only be read in part: {exc}")

    info.architecture = str(metadata.get("general.architecture", "") or "")
    info.name = str(metadata.get("general.name", "") or "")
    arch = info.architecture
    for key in (f"{arch}.context_length", "llama.context_length"):
        if isinstance(metadata.get(key), int):
            info.context_length = int(metadata[key])
            break
    for key in (f"{arch}.embedding_length", "llama.embedding_length"):
        if isinstance(metadata.get(key), int):
            info.embedding_length = int(metadata[key])
            break

    quant = re.search(r"(Q\d[_A-Za-z0-9]*|F16|BF16|F32)", path.name, re.IGNORECASE)
    info.quantisation = quant.group(1).upper() if quant else ""

    if arch and arch not in KNOWN_ARCHITECTURES:
        # The bge-m3 lesson, made preventable. Reported and not refused: a
        # newer llama.cpp may implement it, and a hardcoded list in this file
        # deciding what the box may run is the opposite of what the brief asks.
        info.warnings.append(
            f"this file declares architecture '{arch}', which this build's "
            "llama.cpp may not implement - if it refuses to load with "
            f"\"unknown model architecture: '{arch}'\", that is why, and the "
            "fix is a GGUF converted with a supported architecture tag"
        )
    if not arch:
        info.warnings.append(
            "the file does not declare an architecture; it may be a projector "
            "or an adapter rather than a model"
        )
    return info


# -------------------------------------------------------------- proposing

_SLUG = re.compile(r"[^a-z0-9]+")

# Everything a generation model needs to be routable. An embedder is detected
# from its own metadata rather than from its filename.
GENERATION_CAPS = ["text", "tools"]
EMBEDDING_CAPS = ["embedding"]

# The seed. A new model has no measured quality, and zero would eliminate it
# from every routing decision while looking like a score. A flat 0.5 makes it a
# candidate that loses to a tuned model and wins against nothing - which is the
# honest position for a model nobody has evaluated yet. `quality_measured:
# false` is what the router surfaces on every decision.
SEED_QUALITY = 0.5

TASK_CLASSES = [
    "agentic_multi", "code_generation", "doc_summary", "kb_qa",
    "numerical_calc", "official_drafting", "visual_doc",
]


def propose_id(path: Path, taken: set[str]) -> str:
    stem = _SLUG.sub("-", path.stem.lower()).strip("-") or "model"
    candidate = stem[:48]
    if candidate not in taken:
        return candidate
    for n in range(2, 100):
        alt = f"{candidate}-{n}"
        if alt not in taken:
            return alt
    raise StagingError("could not find a free id for this file")


def propose_entry(info: GgufInfo, model_id: str, *, license_name: str = "unknown",
                  sha256: str | None = None) -> dict[str, Any]:
    """The manifest entry for a freshly staged file.

    Every number here is derived from the file or is an explicit seed, and the
    YAML written from it says which is which.
    """
    is_embedder = bool(
        info.embedding_length
        and info.architecture in ("bert", "nomic-bert")
    ) or "embed" in info.path.name.lower()

    # Weights plus a working allowance. A quantised model needs roughly its own
    # size on the card, and then some for the KV cache and the compute buffers;
    # 15% + 0.35 GB lands close for the 1-8B range this box is built for. It is
    # an ESTIMATE and is written into the file as one - the warden refuses a
    # model that does not fit, so an estimate that is too low fails loudly at
    # admission rather than quietly at inference.
    vram = round(info.size_gb * 1.15 + (0.1 if is_embedder else 0.35), 2)

    entry: dict[str, Any] = {
        "id": model_id,
        "file": info.path.name,
        "sha256": sha256,
        "license": license_name or "unknown",
        "tier": 0 if is_embedder else 2,
        "capabilities": EMBEDDING_CAPS if is_embedder else GENERATION_CAPS,
        "context": info.context_length or (512 if is_embedder else 4096),
        "vram_gb": max(vram, 0.05),
        "load_ms": int(max(200, info.size_gb * 900)),
    }
    if is_embedder:
        entry["role"] = "embedding"
        if info.embedding_length:
            entry["dim"] = info.embedding_length
    else:
        entry["tok_per_s"] = 20.0
        entry["quality"] = dict.fromkeys(TASK_CLASSES, SEED_QUALITY)
        entry["quality_measured"] = False
    return entry


# ----------------------------------------------------------- writing config

def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    text = str(value)
    return text if re.fullmatch(r"[A-Za-z0-9_.\-/]+", text) else f'"{text}"'


def render_entry(entry: dict[str, Any], notes: list[str]) -> str:
    """One manifest entry as text, with the provenance of its numbers in it."""
    lines = ["", *[f"# {note}" for note in notes]]
    first = True
    for key, value in entry.items():
        if key == "quality" and isinstance(value, dict):
            lines.append("  quality:")
            for cls, score in sorted(value.items()):
                lines.append(f"    {cls}: {score}")
            continue
        prefix = "- " if first else "  "
        lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
        first = False
    return "\n".join(lines) + "\n"


def append_model(manifest_path: Path, entry: dict[str, Any], notes: list[str]) -> None:
    """Append an entry to models.yaml without touching a byte of the rest.

    Same rule as scripts/stage_model.py: a yaml.safe_dump round-trip destroys
    every comment in this file, and the comments are load-bearing.
    """
    text = manifest_path.read_text(encoding="utf-8")
    if re.search(rf"^\s*-?\s*id:\s*{re.escape(entry['id'])}\s*$", text, re.MULTILINE):
        raise StagingError(f"a model with id {entry['id']!r} is already in the manifest")
    if not text.endswith("\n"):
        text += "\n"
    manifest_path.write_text(text + render_entry(entry, notes), encoding="utf-8")


_AVAILABLE_INLINE = re.compile(r"^(\s*available:\s*\[)(.*?)(\]\s*)$")


def add_to_profile(profiles_path: Path, profile_name: str, model_id: str) -> bool:
    """Add the id to one profile's `available:` list. Returns False if already there.

    THE SECOND GATE. `config/models.yaml` makes a model exist;
    `registry.available(profile.available)` decides whether the router may ever
    consider it. A model added to the manifest alone is verified, listed, and
    never chosen - with no error anywhere to say why.
    """
    lines = profiles_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    in_profile = False
    done = False

    for line in lines:
        stripped = line.rstrip("\n")
        if re.fullmatch(rf"{re.escape(profile_name)}:\s*", stripped):
            in_profile = True
            out.append(line)
            continue
        # A new top-level key ends the profile block.
        if in_profile and stripped and not stripped[0].isspace():
            in_profile = False

        if in_profile and not done:
            inline = _AVAILABLE_INLINE.match(stripped)
            if inline:
                current = [m.strip() for m in inline.group(2).split(",") if m.strip()]
                if model_id in current:
                    return False
                current.append(model_id)
                out.append(f"{inline.group(1)}{', '.join(current)}{inline.group(3)}\n")
                done = True
                continue
            if re.fullmatch(r"\s*available:\s*", stripped):
                out.append(line)
                indent = len(line) - len(line.lstrip())
                out.append(" " * (indent + 2) + f"- {model_id}\n")
                done = True
                continue
        out.append(line)

    if not done:
        raise StagingError(
            f"profile {profile_name!r} has no `available:` list to add to; add "
            f"{model_id!r} to it by hand in config/profiles.yaml"
        )
    profiles_path.write_text("".join(out), encoding="utf-8")
    return True


# ------------------------------------------------------------------ scan

def unregistered_files(model_dir: Path, known_files: set[str]) -> list[dict[str, Any]]:
    """Every .gguf on the box that no manifest entry points at.

    This is the practical path for a real model: a 5 GB file is copied into the
    folder with the operating system, not pushed through a browser, and this is
    what turns that copy into something the box can be told about.
    """
    if not model_dir.is_dir():
        return []
    found = []
    for path in sorted(model_dir.glob("*.gguf")):
        if path.name in known_files:
            continue
        found.append({
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "size_gb": round(path.stat().st_size / (1024 ** 3), 2),
        })
    return found
