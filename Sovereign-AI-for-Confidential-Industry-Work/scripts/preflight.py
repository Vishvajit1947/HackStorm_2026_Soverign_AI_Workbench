"""Check the box BEFORE the gateway starts, and say what is there.

The workbench has four external things it can want: weight files, a
`llama-server` binary, Docker, and the sandbox image. Any of them can be
absent, each absence costs a different capability, and until this existed the
only way to find out was to start the gateway and discover it four minutes
later in the middle of a task.

TWO RULES, AND THEY ARE THE WHOLE POINT.

  ONLY A REAL BLOCKER STOPS THE START. Exactly one thing here is fatal - a
  config this process cannot read at all, because then there is no gateway to
  report anything with. Everything else is a capability: no Docker means no
  code execution and a fully working knowledge base, router, policy engine,
  audit chain and document writer. Refusing to start over one unavailable tool
  would be the same mistake as marking the whole gateway unhealthy for it.

  NOTHING IS FIXED SILENTLY. This reports and explains; it never installs,
  downloads, edits a config or changes a setting. An operator who runs a
  start script and finds their configuration rewritten has lost the thing that
  makes an air-gapped box auditable.

Run on its own at any time:  python scripts/preflight.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

OK = "  [ok]"
WARN = "  [!] "
BAD = "  [X] "


def _say(mark: str, line: str, *detail: str) -> None:
    print(f"{mark} {line}")
    for extra in detail:
        print(f"        {extra}")


def _docker_ready(image: str) -> tuple[bool, bool]:
    """(docker responds, image present). Never raises, never blocks for long."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=12, check=True)
    except Exception:                                 # noqa: BLE001
        return False, False
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=12, check=False,
        )
        return True, out.returncode == 0
    except Exception:                                 # noqa: BLE001
        return True, False


def _port_holder(port: int) -> str | None:
    """Who is already listening on the port, or None if it is free.

    A second gateway holding the port is the ONE remaining way start.bat fails
    with a stack trace instead of a sentence: uvicorn cannot bind, dies with
    `WinError 10048`, and the window closes before anyone reads it. Worse, the
    browser then opens onto the OTHER process - a different build, a different
    profile - and everything looks fine while nothing you just started is
    running.

    Loopback-only and read-only: this connects to 127.0.0.1 to see whether
    anything answers, and asks the OS who owns the socket. It never kills
    anything - see the "nothing is fixed silently" rule at the top.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.6)
        if probe.connect_ex(("127.0.0.1", port)) != 0:
            return None                              # nothing there; free

    # Something is listening. Name it if the OS will say who, because "in use"
    # without a name leaves the operator hunting through Task Manager.
    if os.name != "nt":
        return "another process"
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=10, check=False
        )
        for line in out.stdout.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                pid = line.split()[-1]
                name = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=10, check=False,
                ).stdout.strip().split(",")[0].strip('"')
                return f"PID {pid} ({name or 'unknown'})"
    except Exception:                                 # noqa: BLE001, S110
        # Naming the holder is a courtesy; failing to name it must not turn a
        # clear "the port is taken" into a crash in the check itself.
        pass
    return "another process"


def main() -> int:
    try:
        from kavach.config import get_settings
    except Exception as exc:                          # noqa: BLE001
        _say(BAD, f"the application could not be imported: {exc}")
        return 1

    try:
        settings = get_settings()
    except Exception as exc:                          # noqa: BLE001
        # The one genuinely fatal case: without settings there is no gateway to
        # report anything with, so this is the only path that returns 1.
        _say(BAD, f"config could not be read: {exc}",
             "Check config/profiles.yaml and kavach.env, then try again.")
        return 1

    print()
    _say(OK, f"profile {settings.profile_name}, backend {settings.backend}")

    # ---- weights ---------------------------------------------------
    model_dir = settings.model_dir
    staged = sorted(model_dir.glob("*.gguf")) if model_dir.is_dir() else []
    if staged:
        total = sum(p.stat().st_size for p in staged) / (1024 ** 3)
        _say(OK, f"{len(staged)} model file(s) in {model_dir}  ({total:.1f} GB)")
    else:
        _say(WARN, f"no .gguf files in {model_dir}",
             "The offline stand-in will answer and mark everything [SIMULATED].",
             "Set KAVACH_MODEL_DIR in kavach.env to where your .gguf files are.")

    # ---- llama-server ----------------------------------------------
    if settings.backend == "llamacpp":
        binary = shutil.which("llama-server")
        if not binary and settings.llama_bin_dir:
            for name in ("llama-server.exe", "llama-server"):
                candidate = settings.llama_bin_dir / name
                if candidate.is_file():
                    binary = str(candidate)
                    break
        if binary:
            _say(OK, f"llama-server at {binary}")
        else:
            _say(WARN, "llama-server was not found",
                 "The backend is set to llamacpp but the binary is not on PATH",
                 "and KAVACH_LLAMA_BIN_DIR does not point at it. Generation",
                 "will fail until it does.")
    else:
        _say(WARN, f"backend is {settings.backend!r}, not llamacpp",
             "Every generated line will be marked [SIMULATED]. Set",
             "KAVACH_BACKEND=llamacpp in kavach.env to use the real models.")

    # ---- vision model endpoint -------------------------------------
    try:
        from kavach.knowledge.extract import ImageReader
        ir = ImageReader()
        vl_spec = ir._vision_spec()
        if vl_spec and vl_spec.has_projector:
            _say(OK, f"vision model endpoint ready ({vl_spec.id} + mmproj)")
        elif vl_spec:
            _say(WARN, f"vision model {vl_spec.id} staged but projector missing ({vl_spec.mmproj})")
        else:
            _say(WARN, "no vision model available in profile",
                 "Engineering drawings will rely on OCR and geometry engines.")
    except Exception as exc:  # noqa: BLE001
        _say(WARN, f"vision check: {exc}")

    # ---- the sandbox -----------------------------------------------
    image = os.environ.get("KAVACH_SANDBOX_IMAGE", "kavach/sandbox:0.1.0")
    docker, has_image = _docker_ready(image)
    if docker and has_image:
        _say(OK, f"code sandbox ready ({image})")
    elif docker:
        _say(WARN, f"Docker is running but {image} is not built",
             "Code execution will be refused until you run:",
             "  docker compose build sandbox-runner")
    else:
        _say(WARN, "Docker is not running - code execution will be refused",
             "Everything else works. Start Docker Desktop if you need the",
             "assistant to run code.")

    # ---- the port ---------------------------------------------------
    holder = _port_holder(settings.port)
    if holder is None:
        _say(OK, f"port {settings.port} is free")
    else:
        _say(BAD, f"port {settings.port} is already in use by {holder}",
             "Another Kavach gateway is probably still running - open",
             f"  http://127.0.0.1:{settings.port}",
             "to see. If it is one you no longer want, close its window, or:",
             f"  for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{settings.port}') "
             "do taskkill /PID %a /F",
             "Starting anyway would fail to bind and the browser would open on",
             "the OTHER gateway - a different build, and nothing you started.")
        return 1

    # ---- the database ----------------------------------------------
    try:
        settings.ensure_dirs()
        _say(OK, f"workspace at {settings.var_dir}")
    except Exception as exc:                          # noqa: BLE001
        _say(BAD, f"the var directory cannot be created: {exc}")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
