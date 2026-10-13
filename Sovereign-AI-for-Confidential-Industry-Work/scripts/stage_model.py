"""Record the SHA-256 of staged weight files into config/models.yaml.

This is the deliberate act of approval that model provenance (§02.4) rests on. It
is separate from the runtime on purpose: the runtime only ever *checks* digests,
it never writes one. An operator staging weights runs this once, reviews the diff,
and commits it.

    python scripts/stage_model.py                 # report what is staged
    python scripts/stage_model.py --write         # record digests for staged files
    python scripts/stage_model.py --write --only qwen3-8b-q4

Refuses to overwrite a digest that is already recorded and now disagrees - that is
either a corrupted download or a substituted file, and it needs a human, not a
flag. Use --force only when you have decided the new file is the approved one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

import yaml

from kavach.config import get_settings
from kavach.runtime.registry import sha256_file

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def write_digests(manifest_path: Path, digests: dict[str, str]) -> None:
    """Rewrite only the `sha256:` lines, leaving the rest of the file byte-identical.

    A yaml.safe_dump round-trip would silently destroy every comment in
    models.yaml - including the warning that the `quality` numbers are unmeasured
    placeholders, which is the one comment nobody can afford to lose. So this
    walks the raw lines and edits in place.
    """
    lines = manifest_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    current: str | None = None
    pending = dict(digests)

    for line in lines:
        stripped = line.lstrip("- ").strip()
        if stripped.startswith("id:"):
            current = stripped.split("id:", 1)[1].strip()

        if (
            current in pending
            and line.strip().startswith("sha256:")
        ):
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}sha256: {pending.pop(current)}{newline}")
            continue

        out.append(line)

    if pending:
        raise RuntimeError(
            "no `sha256:` line found for: "
            + ", ".join(sorted(pending))
            + " - add `sha256: null` to those entries first"
        )

    manifest_path.write_text("".join(out), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="record weight digests into models.yaml")
    parser.add_argument("--write", action="store_true", help="write digests into the manifest")
    parser.add_argument("--only", default=None, help="limit to one model id")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite a recorded digest that now disagrees (needs a human decision)",
    )
    args = parser.parse_args()

    settings = get_settings()
    manifest_path = settings.config_dir / "models.yaml"
    model_dir = settings.model_dir

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or []

    print(f"{BOLD}Staging report{RESET}")
    print(f"{DIM}  manifest {manifest_path}{RESET}")
    print(f"{DIM}  weights  {model_dir}{RESET}\n")

    staged = missing = recorded = conflicts = 0
    to_write: dict[str, str] = {}

    for entry in raw:
        model_id = entry["id"]
        if args.only and model_id != args.only:
            continue

        path = model_dir / entry["file"]
        if not path.exists():
            print(f"  {YELLOW}not staged {RESET}{model_id:<24} {DIM}{entry['file']}{RESET}")
            missing += 1
            continue

        staged += 1
        digest = sha256_file(path)
        existing = entry.get("sha256")
        size_gb = path.stat().st_size / (1024**3)

        if existing and existing.lower() == digest.lower():
            print(f"  {GREEN}verified   {RESET}{model_id:<24} {DIM}{digest[:16]}...{RESET}")
            recorded += 1
            continue

        if existing and not args.force:
            print(f"  {RED}CONFLICT   {RESET}{model_id:<24}")
            print(f"    {DIM}approved {existing}{RESET}")
            print(f"    {DIM}on disk  {digest}{RESET}")
            print(f"    {RED}the file is not the one that was approved. "
                  f"Re-stage it, or re-run with --force if you have decided "
                  f"this file is correct.{RESET}")
            conflicts += 1
            continue

        if args.write:
            to_write[model_id] = digest
            verb = f"{GREEN}recorded  {RESET}" if not existing else f"{YELLOW}REPLACED  {RESET}"
            print(f"  {verb} {model_id:<24} {DIM}{digest[:16]}... ({size_gb:.1f} GB){RESET}")
        else:
            print(f"  {YELLOW}would add  {RESET}{model_id:<24} "
                  f"{DIM}{digest[:16]}... ({size_gb:.1f} GB){RESET}")

    if to_write:
        write_digests(manifest_path, to_write)
        print(f"\n{GREEN}manifest updated{RESET} - review the diff before committing")

    print(
        f"\n{staged} staged, {recorded} already verified, {missing} not on this box"
        + (f", {RED}{conflicts} CONFLICTS{RESET}" if conflicts else "")
    )

    if conflicts:
        return 2
    if not args.write and staged > recorded:
        print(f"{DIM}re-run with --write to record them{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
