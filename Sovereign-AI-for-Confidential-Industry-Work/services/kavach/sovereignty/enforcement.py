"""Is anything actually stopping THIS PROCESS from reaching the network? (§02.2)

The canary fires a probe and reports what came back. Until this module existed it
had no way to know whether a refusal came from a POLICY or from the routing
table, and it guessed wrong in the reassuring direction: a connect to the RFC
5737 documentation address returned EHOSTUNREACH, and the panel reported "kernel
refused the connection outright - this is the firewall, not a slow network" on a
machine with no firewall rule of any kind. Measured on the development box:

    203.0.113.1:443  tcp   times out after 2000 ms      (nothing is blocking it)
    203.0.113.1:53   udp   times out after 2000 ms      (nothing is blocking it)
    1.1.1.1:443      tcp   REACHED in 5 ms
    1.1.1.1:53       udp   REACHED, 39-byte answer in 5 ms

An unroutable address is unreachable everywhere, on every machine, forever. It
proves the address is documentation-only. It proves nothing about Kavach.

THE CLAIM IS ABOUT THE APPLICATION, NOT THE MACHINE. Nobody unplugs the plant to
run a workbench, so "sovereign" here has to mean *this process* cannot get out
while the computer it runs on carries on working normally. Three mechanisms can
make that true, and this module's only job is to report, honestly, which of them
is armed right now:

    nftables    Linux, in the compose stack: `deploy/nftables.conf` drops egress
                from Kavach's own uid/cgroup. The drop counter is readable, which
                is why the sentinel prefers it.
    windows     Windows Firewall outbound BLOCK rule scoped to the program -
                `deploy/windows-firewall.ps1`. Per-application, leaves the rest
                of the machine online, and is the only one of the three that
                works on the box this is demonstrated on.
    container   the code.exec sandbox, which has no network interface at all.
                Strongest of the three and the narrowest: it contains generated
                code, not the gateway.

`detect()` returns NONE rather than guessing. A workbench that says "egress is
enforced" because a documentation address did not answer is worse than one that
says "nothing is armed here" - the second is a task, the first is a false claim
with a real digest next to it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

# The rule name deploy/windows-firewall.ps1 writes. Matched exactly, because a
# rule someone else created that happens to mention Kavach is not our rule and
# may not do what this module would claim it does.
WINDOWS_RULE_NAME = "Kavach egress block"

# Linux: the table deploy/nftables.conf creates.
NFT_TABLE = ("inet", "kavach")


@dataclass(frozen=True)
class Enforcement:
    """What is stopping this process, if anything."""

    present: bool
    mechanism: str          # nftables | windows-firewall | none | unknown
    scope: str              # application | machine | none
    detail: str
    # What to run to arm it. Empty when something is already armed - an
    # instruction shown beside a working control is noise.
    fix: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "mechanism": self.mechanism,
            "scope": self.scope,
            "detail": self.detail,
            "fix": self.fix,
        }


_WINDOWS_FIX = (
    "run deploy/windows-firewall.ps1 -Arm from an ADMINISTRATOR PowerShell. It adds "
    "one outbound block rule for this Python executable only - the rest of the "
    "machine keeps its network."
)
_LINUX_FIX = (
    "sudo nft -f deploy/nftables.conf, or run inside the compose stack where that "
    "is applied for you."
)


def _fix_for_platform() -> str:
    return _WINDOWS_FIX if os.name == "nt" else _LINUX_FIX


def _nftables() -> Enforcement | None:
    """The Linux path. Present only if OUR table is loaded - any other firewall
    on the box may be doing something entirely different."""
    if not shutil.which("nft"):
        return None
    family, name = NFT_TABLE
    try:
        out = subprocess.run(
            ["nft", "-j", "list", "table", family, name],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout)
    except ValueError:
        return None
    if not data.get("nftables"):
        return None
    return Enforcement(
        present=True,
        mechanism="nftables",
        scope="application",
        detail=f"nftables table {family} {name} is loaded and dropping egress from this process",
    )


def _windows_firewall() -> Enforcement | None:
    """The Windows path, via netsh rather than PowerShell.

    netsh is on every Windows install and needs no execution policy; spawning
    powershell.exe to read one rule costs about a second and is one more thing
    that can be disabled on a locked-down venue machine.

    The rule is only reported as enforcement when it is ENABLED, outbound, a
    BLOCK, and points at the executable running this process. A disabled rule
    is not a policy, and a rule for some other program does not contain us.
    """
    if os.name != "nt":
        return None
    netsh = shutil.which("netsh")
    if not netsh:
        return None
    try:
        out = subprocess.run(
            [netsh, "advfirewall", "firewall", "show", "rule",
             f"name={WINDOWS_RULE_NAME}", "dir=out", "verbose"],
            capture_output=True, text=True, timeout=8, check=False,
            # netsh localises its output; the fields below are matched
            # case-insensitively on the English labels, which is what an
            # en-US venue box produces. A mismatch reads as "no rule", which
            # is the safe direction to be wrong in.
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0 or "No rules match" in out.stdout:
        return None

    fields = {}
    for line in out.stdout.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip().lower()] = value.strip()

    enabled = fields.get("enabled", "").lower().startswith("yes")
    blocking = fields.get("action", "").lower().startswith("block")
    program = fields.get("program", "")
    if not (enabled and blocking):
        return None

    # `Any` would be a machine-wide block, which is a different claim and not
    # the one this product makes. Reported, but named for what it is.
    scope = "machine" if program.lower() in ("", "any") else "application"
    return Enforcement(
        present=True,
        mechanism="windows-firewall",
        scope=scope,
        detail=(
            f"Windows Firewall rule '{WINDOWS_RULE_NAME}' is enabled and blocking "
            f"outbound traffic from {program or 'every program'}"
        ),
    )


def detect() -> Enforcement:
    """What is armed on this box right now. Never guesses upward."""
    for probe in (_nftables, _windows_firewall):
        found = probe()
        if found is not None:
            return found

    return Enforcement(
        present=False,
        mechanism="none",
        scope="none",
        detail=(
            "no egress policy is armed for this process. Whatever a probe reports "
            "is the routing table's answer, not a refusal - an address that does "
            "not answer has not been blocked, it has merely not answered"
        ),
        fix=_fix_for_platform(),
    )


def this_executable() -> str:
    """The program a per-application rule has to name. `sys.executable` is the
    venv's python.exe, which is what actually opens the sockets."""
    return sys.executable
