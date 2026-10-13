"""offline_check.py - prove this box has nothing to phone home with (§09).

    .venv\\Scripts\\python.exe scripts\\offline_check.py

"Air-gapped" is the product's central claim, and the L0 canary proves the
*network* is closed. This script proves something different and just as
load-bearing: that nothing we ship would try in the first place.

Four checks, each of which can fail the build:

    1. THE UI BUNDLE      no absolute URL, anywhere. Not a CDN script, not a web
                          font, not a favicon, not a tracking pixel, not an
                          `@import url(https://...)` three levels into a
                          stylesheet. This is the check that matters, because a
                          single `<script src="https://cdn...">` turns a
                          sovereign workbench into a blank panel on a plant
                          network - and into a request log on someone else's
                          server if the network is only half closed.

    2. THE CSP            index.html must carry a Content-Security-Policy that
                          confines it to 'self'. Belt and braces on top of check
                          1: an asset added later that we failed to catch is
                          refused by the browser rather than fetched.

    3. DEPENDENCIES       every declared dependency imports from the local
                          environment right now. An offline install that needs
                          one wheel from PyPI is not an offline install, and the
                          moment you find out is on the venue's wifi.

    4. SOURCE HOSTNAMES   report every absolute URL and hostname in the Python
                          source, each of which must be a known, justified
                          exception (the RFC 5737 canary target, the private
                          llama.cpp endpoint) rather than something that crept in.

Check 4 reports rather than fails, because the exceptions are real and a check
that cries wolf gets suppressed. Checks 1-3 fail.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "services" / "kavach" / "ui"
SRC_DIR = ROOT / "services" / "kavach"

GREEN, RED, AMBER, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

# Anything that would make the browser open a connection to a host: an absolute
# URL, or a protocol-relative one.
#
# The host is REQUIRED. Matching a bare `https://` finds every mention of the
# string in a comment and reports a hostname of "", which is how a checker earns
# the reputation that gets it skipped. A `//` must be followed immediately by
# something host-shaped, so an ordinary JS comment does not trip it.
_HOST = (
    r"(?:localhost"
    r"|\d{1,3}(?:\.\d{1,3}){3}"
    r"|[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9][a-z0-9\-]*)+)"
)
EXTERNAL = re.compile(rf"(?i)(?:(?:https?|ftp|wss?):)?//(?P<host>{_HOST})(?::\d+)?")

# `url(...)` in CSS that is not relative or a data: URI.
CSS_URL = re.compile(r"""url\(\s*['"]?(?!data:)([^'")]+)['"]?\s*\)""", re.IGNORECASE)

# XML namespace NAMES. `document.createElementNS(ns, "svg")` requires this exact
# string, and the browser never dereferences it - it is an identifier that
# happens to be spelled like a URL. Nothing is fetched, on any network, ever.
#
# Exempted as whole literals rather than by host, so `https://www.w3.org/TR/...`
# - a real page on a real server - is still a failure. Widening this is then a
# deliberate edit to this list, not an accident in a regex.
NAMESPACE_URIS = (
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/1999/xlink",
    "http://www.w3.org/1999/xhtml",
)


def _inside_namespace_literal(line: str, start: int, end: int) -> bool:
    """Is this match part of one of the exempt namespace names, exactly?"""
    for uri in NAMESPACE_URIS:
        offset = 0
        while (found := line.find(uri, offset)) != -1:
            if found <= start and end <= found + len(uri):
                return True
            offset = found + 1
    return False

# Known, deliberate, documented. Each of these is a decision with a comment on it.
JUSTIFIED = {
    "203.0.113.1": "RFC 5737 documentation address - the canary's own target",
    "127.0.0.1": "loopback - never leaves the box, and the kernel cannot route it out",
    "localhost": "loopback - the CORS allowance for a vite dev server, dev only",
    "0.0.0.0": "a bind address, not a destination - nothing is dialled",
    # The sandbox egress probe deliberately TRIES to reach the network so the
    # failure can be shown. A checker that flagged it would be flagging the
    # demonstration of the property it is checking for.
    "1.1.1.1": "the code.exec egress probe's target - it is meant to fail",
    "claude.com": "attribution string in a generated commit body",
}


def _colour(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)


# ---------------------------------------------------------------- 1. the bundle


def check_ui_bundle() -> list[str]:
    rule("1. THE UI BUNDLE - no absolute URL, anywhere")
    failures: list[str] = []

    if not UI_DIR.exists():
        print(_colour(f"  no UI bundle at {UI_DIR}", AMBER))
        print(_colour("  the API is still usable; there is just no panel", DIM))
        return failures

    assets = sorted(p for p in UI_DIR.rglob("*") if p.is_file())
    print()
    for path in assets:
        text = path.read_text(encoding="utf-8", errors="replace")
        hits: list[str] = []

        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in EXTERNAL.finditer(line):
                if _inside_namespace_literal(line, match.start(), match.end()):
                    continue
                hits.append(f"line {line_no}: {match.group(0)}")
            if path.suffix == ".css":
                for match in CSS_URL.finditer(line):
                    ref = match.group(1)
                    if ref.startswith(("http", "//")):
                        hits.append(f"line {line_no}: css url({ref})")

        # Loopback is not exempt here. A bundle hard-coding http://127.0.0.1:8420
        # breaks the moment it is served on the demo box's real address or a
        # different port - and it is served same-origin precisely so it never
        # needs to name a host at all.
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        if hits:
            print(_colour(f"  FAIL  {rel}", RED))
            for hit in hits:
                print(_colour(f"          {hit}", RED))
            failures.extend(f"{rel} {h}" for h in hits)
        else:
            print(_colour(f"  ok    {rel}", GREEN)
                  + _colour(f"  ({len(text)} bytes)", DIM))

    print()
    if failures:
        print(_colour("  This bundle would make a network request.", RED))
        print(_colour("  On a closed plant network that is a blank panel; on a "
                      "half-open one it is a request log on someone else's "
                      "server.", DIM))
    else:
        print(_colour(f"  {len(assets)} asset(s), zero absolute URLs", GREEN))
    return failures


# ---------------------------------------------------------------- 2. the CSP


def check_csp() -> list[str]:
    rule("2. CONTENT-SECURITY-POLICY - the browser enforces it too")
    index = UI_DIR / "index.html"
    if not index.exists():
        print(_colour("  no index.html - skipped", DIM))
        return []

    text = index.read_text(encoding="utf-8")
    # The delimiter is captured and back-referenced. A CSP is full of 'self', so
    # a value delimited by " legitimately contains ' - and a naive ["'] ... ["']
    # pair truncates the policy at "default-src" and then reports every directive
    # as missing, which is exactly the false alarm that gets a check disabled.
    match = re.search(
        r"""http-equiv=(["'])Content-Security-Policy\1[^>]*"""
        r"""content=(["'])(?P<policy>.*?)\2""",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        print(_colour("  FAIL  index.html has no Content-Security-Policy", RED))
        return ["index.html: no CSP"]

    policy = " ".join(match.group("policy").split())
    print()
    print(f"  {policy}")
    print()

    failures = []
    for directive in ("default-src 'self'", "connect-src 'self'", "script-src 'self'"):
        if directive in policy:
            print(_colour(f"  ok    {directive}", GREEN))
        else:
            print(_colour(f"  FAIL  missing {directive}", RED))
            failures.append(f"CSP missing {directive}")
    return failures


# ---------------------------------------------------------------- 3. deps


def _declared_dependencies() -> list[tuple[str, str]]:
    """(distribution, import name) for everything the wheel needs at runtime."""
    return [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("pyyaml", "yaml"),
        ("httpx", "httpx"),
        ("numpy", "numpy"),
        ("sse-starlette", "sse_starlette"),
        ("sympy", "sympy"),
        # the `knowledge` extra - Phase 6 does not work without these
        ("pymupdf", "pymupdf"),
        ("rank-bm25", "rank_bm25"),
    ]


def check_dependencies() -> list[str]:
    rule("3. DEPENDENCIES - every one of them is already on this box")
    failures: list[str] = []
    print()
    for dist, module in _declared_dependencies():
        try:
            importlib.import_module(module)
        except ImportError as exc:
            print(_colour(f"  MISSING  {dist:<16} import {module}: {exc}", RED))
            failures.append(f"{dist} is not installed")
        else:
            print(_colour(f"  ok       {dist:<16} import {module}", GREEN))

    print()
    if failures:
        print(_colour("  An install that needs one wheel from PyPI is not an "
                      "offline install,", RED))
        print(_colour("  and the venue's wifi is where you find out.", DIM))
        print(_colour("  Stage a wheelhouse: pip download -d wheels -e \".[knowledge]\"", DIM))
        print(_colour("  then install with:  pip install --no-index --find-links wheels ...", DIM))
    return failures


# ---------------------------------------------------------------- 4. source


def check_source_hostnames() -> None:
    rule("4. HOSTNAMES IN THE SOURCE - each one had better be deliberate")
    found: dict[str, list[str]] = {}

    for path in sorted(SRC_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in EXTERNAL.finditer(line):
                where = f"{path.relative_to(ROOT).as_posix()}:{line_no}"
                found.setdefault(match.group(0), []).append(where)

    print()
    if not found:
        print(_colour("  no absolute URL anywhere in the Python source", GREEN))
    for url, places in sorted(found.items()):
        host = EXTERNAL.match(url).group("host")
        note = JUSTIFIED.get(host)
        tone = DIM if note else AMBER
        print(_colour(f"  {url}", tone))
        print(_colour(f"      {note or 'REVIEW THIS - not on the justified list'}",
                      DIM if note else AMBER))
        for place in places[:3]:
            print(_colour(f"      {place}", DIM))

    # Reported, never failed: the canary MUST name an address to attempt egress
    # to, and a check that fires on its own subject gets suppressed within a week.
    print()
    print(_colour("  Reported, not failed - the canary target is the point of the "
                  "canary.", DIM))


# ---------------------------------------------------------------- main


def main() -> int:
    print(_colour("Kavach offline check", BOLD))
    print(_colour("Does anything we ship try to reach a network?", DIM))

    failures: list[str] = []
    failures += check_ui_bundle()
    failures += check_csp()
    failures += check_dependencies()
    check_source_hostnames()

    rule("VERDICT")
    print()
    if failures:
        print(_colour(f"  {len(failures)} problem(s):", RED))
        for failure in failures:
            print(_colour(f"    {failure}", RED))
        print()
        print(_colour("  This box would attempt an external request. Fix before "
                      "the demo.", RED))
        return 1

    print(_colour("  Nothing shipped here can reach a network.", GREEN))
    print()
    print(_colour("  This is the STATIC half of the sovereignty claim. The other "
                  "half is", DIM))
    print(_colour("  enforcement, which only the canary on a machine with "
                  "nftables can show:", DIM))
    print(_colour("    sudo nft -f deploy/nftables.conf && kavach canary", DIM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
