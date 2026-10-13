"""Open the browser when the gateway answers - and not a moment before.

`start.bat` used to run

    start "" "http://127.0.0.1:8420"
    python -m uvicorn ...

in that order, so the browser opened against a port nothing was listening on
yet and Chrome showed:

    This site can't be reached
    127.0.0.1 refused to connect.   ERR_CONNECTION_REFUSED

The gateway was fine. It just had not finished booting: the registry hashes
every staged weight file against the digest in models.yaml, which is ~34 GB of
reading on this box and takes minutes the first time. A person seeing that
screen reasonably concludes the workbench is broken and closes the window -
which kills the boot that was working.

So this waits for `/api/health` to answer, prints what it is waiting for, and
opens the browser only then. On a timeout it opens NOTHING and says what to
look at, because an error page is worse than no page: the first told the user
the product was broken when it was starting.

Stdlib only, loopback only - `urllib` against 127.0.0.1 resolves no hostname
and leaves the box as offline as it was.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import webbrowser

# Generous, because the thing being waited for is honest work: hashing every
# staged file is the provenance check, not a hang. Past this the problem is
# real and a browser window would only hide it.
TIMEOUT_S = 600.0
POLL_S = 1.0
# Silence for minutes reads as a hung script. One line every few seconds says
# the wait is deliberate and how long it has been.
SAY_EVERY_S = 15.0


def wait_for(url: str, timeout_s: float = TIMEOUT_S) -> bool:
    started = time.monotonic()
    said = 0.0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_s:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 400:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass                                   # not up yet; that is the point

        if elapsed - said >= SAY_EVERY_S:
            said = elapsed
            print(
                f"   ... still starting ({elapsed:.0f}s). First start checks every "
                "staged weight file against its recorded digest.",
                flush=True,
            )
        time.sleep(POLL_S)


def main(argv: list[str]) -> int:
    port = argv[1] if len(argv) > 1 else "8420"
    base = f"http://127.0.0.1:{port}"

    if wait_for(f"{base}/api/health"):
        print(f"   [ok] gateway is answering - opening {base}", flush=True)
        webbrowser.open(base)
        return 0

    # Deliberately no browser. See the module docstring.
    print(
        f"\n   [X] the gateway did not answer on {base} within "
        f"{TIMEOUT_S:.0f}s.\n"
        "       The window running it is still open - read the messages there;\n"
        "       that is where the reason will be. Nothing was opened in the\n"
        "       browser, because an error page would say the workbench is\n"
        "       broken when the real problem is printed above it.\n",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
