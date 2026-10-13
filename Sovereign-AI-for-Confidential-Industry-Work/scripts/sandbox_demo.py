"""Run generated code that tries to reach the network, and watch it fail.

    python scripts/sandbox_demo.py

The tool-plane counterpart of the L0 canary. Rather than asserting that generated
code cannot exfiltrate, this runs some that tries.

Needs Docker and the sandbox image:
    docker compose build sandbox-runner

Without them it says so and exits non-zero, rather than running on the host and
presenting the result as if it proved something.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

from kavach.tools import get_executor
from kavach.tools.impl.sandbox import (
    EGRESS_PROBE,
    SANDBOX_IMAGE,
    docker_available,
    image_available,
)
from kavach.tools.workspace import Workspace

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

FILESYSTEM_PROBE = """\
import os
print("root filesystem is read-only?")
try:
    open("/etc/kavach-probe", "w").write("x")
    print("  WROTE TO / - the rootfs is NOT read-only")
except OSError as exc:
    print(f"  blocked: {type(exc).__name__}: {exc}")

print("can I see the host's files?")
print(f"  /work contains: {os.listdir('/work')}")

print("who am I?")
print(f"  uid={os.getuid()} gid={os.getgid()}")
"""


def banner(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")
    print(DIM + "-" * len(text) + RESET)


def main() -> int:
    docker = docker_available()
    image = image_available() if docker else False

    banner("0. Is the sandbox actually available?")
    print(f"  docker daemon   {GREEN + 'yes' + RESET if docker else RED + 'NO' + RESET}")
    print(f"  image           "
          f"{GREEN + 'present' + RESET if image else RED + 'MISSING' + RESET}"
          f"  {DIM}{SANDBOX_IMAGE}{RESET}")

    if not (docker and image):
        print(f"\n{YELLOW}Cannot run the demo.{RESET} code.exec will refuse rather than")
        print("silently running generated code on this machine.")
        print(f"\n{DIM}  docker compose build sandbox-runner{RESET}")
        return 1

    executor = get_executor()
    ws = Workspace(Path(tempfile.mkdtemp(prefix="kavach-sandbox-demo-")) / "run")
    (ws.root / "input.txt").write_text("a file the run owns", encoding="utf-8")

    banner("1. Ordinary work runs fine")
    result = executor.invoke(
        "code.exec",
        {"code": "import numpy as np\nprint('sum:', int(np.sum([1, 2, 3, 4])))"},
        ws,
    )
    print(f"  {result.data['stdout'].strip()}")
    print(f"  {DIM}exit {result.data['exit_code']} in "
          f"{result.data['elapsed_ms']:.0f} ms{RESET}")

    banner("2. The same sandbox, trying to reach the network")
    result = executor.invoke("code.exec", {"code": EGRESS_PROBE}, ws)
    stdout = result.data["stdout"]
    for line in stdout.rstrip().splitlines():
        colour = RED if "REACHED" in line or "CONNECTED" in line else GREEN
        marker = colour if ("blocked" in line or "REACHED" in line
                            or "CONNECTED" in line) else DIM
        print(f"  {marker}{line}{RESET}")

    reached = "REACHED THE NETWORK" in stdout or "SOCKET CONNECTED" in stdout

    banner("3. And trying to escape the filesystem")
    result = executor.invoke("code.exec", {"code": FILESYSTEM_PROBE}, ws)
    for line in result.data["stdout"].rstrip().splitlines():
        marker = RED if "NOT read-only" in line else (
            GREEN if "blocked" in line else DIM
        )
        print(f"  {marker}{line}{RESET}")

    print()
    if reached:
        print(f"{RED}{BOLD}SANDBOX BREACH{RESET} - generated code reached the network.")
        print("Do not present this build.")
        return 1

    print(f"{BOLD}Generated code cannot exfiltrate even in principle.{RESET}")
    print(
        f"{DIM}The container has no network interface at all (--network=none), so\n"
        f"there is no rule to misconfigure. The seccomp profile additionally removes\n"
        f"socket() from the syscall table, so two independent controls would both\n"
        f"have to fail before anything got out.{RESET}"
    )
    print(f"{DIM}workspace: {ws.root}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
