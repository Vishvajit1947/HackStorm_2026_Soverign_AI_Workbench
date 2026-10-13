"""The canary (§02.2).

A dashboard that always reads zero is indistinguishable from a dashboard that is
broken. So Kavach deliberately attempts egress - a DNS lookup and a TLS connect to
a pinned address - and shows the enforcement catching it.

This is a SELF-TEST OF OUR OWN FIREWALL. It sends nothing but a SYN and a DNS
query, to an address in the RFC 5737 documentation range by default, and it exists
to prove the block works. It is not a network probe of anyone else's infrastructure.

Verdict honesty matters more than a green tick here:

  BLOCKED    the kernel refused us outright - ENETUNREACH, EHOSTUNREACH, EPERM,
             EACCES - AND an egress policy is armed to have caused it. This is
             enforcement. It is the verdict we want, and it is the only one that
             proves anything.
  TIMEOUT    nothing came back before the deadline. This does NOT prove the
             firewall worked - an unplugged cable looks identical. Reported as
             INCONCLUSIVE, and the dashboard says so in those words.
  REACHED    we got a response. The claim is broken. Say so loudly, fail the
             check, and do not let the demo continue pretending otherwise.

WHY `BLOCKED` NOW REQUIRES A POLICY TO BE ARMED. This is the second half of the
same honesty rule, and it was missing for a long time. With nothing armed, a
connect to the RFC 5737 documentation address returns EHOSTUNREACH from the
ROUTING TABLE, and the panel reported "kernel refused the connection outright -
this is the firewall, not a slow network" on a machine that had no firewall rule
and full internet. Measured, on the box this was found on:

    203.0.113.1:443  tcp   timed out after 2000 ms    <- proves nothing
    1.1.1.1:443      tcp   REACHED in 5 ms            <- the app was never contained

An unroutable address is unreachable on every machine on earth. `enforcement.py`
answers the question the canary could not: is anything actually stopping THIS
PROCESS? When the answer is no, a refusal is downgraded to INCONCLUSIVE and the
panel says what to arm.

AND THE TARGET FOLLOWS THE ANSWER. With a policy armed the probe goes to a
ROUTABLE address, because refusing an address that genuinely has a route is the
only result that separates policy from routing. That probe is only ever fired
when something is armed to stop it; if it gets out, one SYN reaches a public
resolver and the canary reports a breach in the loudest terms it has, which is
precisely what should happen to a containment that does not contain.

If a judge asks "how do you know the counter isn't just hardcoded to zero?", this
is the answer, and the honest handling of TIMEOUT is what makes it credible.
"""

from __future__ import annotations

import errno
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from ..audit import get_audit
from ..store import get_db
from .enforcement import Enforcement, detect

# RFC 5737 TEST-NET-3: reserved for documentation, never routed on the public
# internet. Override with a plant-relevant address when demoing on a real LAN,
# where a definitive ENETUNREACH is more convincing.
DEFAULT_TARGET_IP = "203.0.113.1"

# The address used INSTEAD, and only when an egress policy is armed: a public
# resolver that certainly has a route, so that being refused proves the policy
# rather than the routing table. Never probed on an unprotected box - see the
# module docstring.
ENFORCED_TARGET_IP = "1.1.1.1"
DEFAULT_TLS_PORT = 443
DEFAULT_DNS_PORT = 53
DEFAULT_TIMEOUT_S = 2.0

# Errnos that mean "the stack refused to even try" - i.e. enforcement, not silence.
DEFINITIVE_BLOCK = {
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.EPERM,
    errno.EACCES,
    errno.ECONNREFUSED,
}


@dataclass
class ProbeResult:
    name: str
    proto: str
    target: str
    verdict: str                  # BLOCKED | INCONCLUSIVE | REACHED
    detail: str
    errno_name: str | None = None
    elapsed_ms: float = 0.0

    @property
    def proves_enforcement(self) -> bool:
        return self.verdict == "BLOCKED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "proto": self.proto,
            "target": self.target,
            "verdict": self.verdict,
            "detail": self.detail,
            "errno": self.errno_name,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class CanaryResult:
    ok: bool                      # true only if EVERY probe was definitively blocked
    conclusive: bool              # false if anything timed out
    probes: list[ProbeResult] = field(default_factory=list)
    headline: str = ""
    blocked_before: int = 0
    blocked_after: int = 0
    # What was armed when these probes ran. Without it a reader cannot tell a
    # refusal by policy from an address that has no route, which is the whole
    # distinction this class exists to report.
    enforcement: Enforcement | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "conclusive": self.conclusive,
            "headline": self.headline,
            "enforcement": (
                self.enforcement.as_dict() if self.enforcement is not None else None
            ),
            "blocked_before": self.blocked_before,
            "blocked_after": self.blocked_after,
            "counter_moved": self.blocked_after - self.blocked_before,
            "probes": [p.as_dict() for p in self.probes],
        }


def _classify(
    exc: OSError, elapsed_ms: float, armed: bool = True
) -> tuple[str, str, str | None]:
    """(verdict, detail, errno name).

    `armed` is whether an egress policy is in place to have caused a refusal.
    With nothing armed, EHOSTUNREACH is the routing table declining to carry a
    packet to an address that has no route - true on every machine ever built,
    and not a fact about this one. Calling that BLOCKED is how the panel came to
    print "this is the firewall" on a box with no firewall.
    """
    err = exc.errno
    name = errno.errorcode.get(err, str(err)) if err else None

    if err in DEFINITIVE_BLOCK:
        if not armed:
            return (
                "INCONCLUSIVE",
                (
                    f"the stack declined to carry this ({name}) after {elapsed_ms:.0f} ms, "
                    "but no egress policy is armed on this box - that is the routing "
                    "table saying the address has no route, which is true everywhere "
                    "and proves nothing about this application"
                ),
                name,
            )
        return (
            "BLOCKED",
            (
                f"the egress policy refused the connection outright ({name}) after "
                f"{elapsed_ms:.0f} ms - this is enforcement, not a slow network"
            ),
            name,
        )
    if isinstance(exc, socket.timeout) or err in (errno.ETIMEDOUT, errno.EAGAIN):
        return (
            "INCONCLUSIVE",
            (
                "no reply at all. That looks the same as an unplugged cable, so "
                "on its own it does not prove anything was blocked"
            ),
            name,
        )
    # Some other OSError. It failed, which is good, but with nothing armed there
    # is nothing entitled to take credit for it.
    return (
        "BLOCKED" if armed else "INCONCLUSIVE",
        f"connection failed ({name or type(exc).__name__}): {exc}"
        + ("" if armed else " - with no egress policy armed, the cause is unknown"),
        name,
    )


def _tcp_probe(
    name: str, ip: str, port: int, timeout_s: float, armed: bool = True
) -> ProbeResult:
    target = f"{ip}:{port}"
    started = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((ip, port))
    except OSError as exc:
        elapsed = (time.perf_counter() - started) * 1000
        verdict, detail, err_name = _classify(exc, elapsed, armed)
        return ProbeResult(name, "tcp", target, verdict, detail, err_name, elapsed)
    else:
        elapsed = (time.perf_counter() - started) * 1000
        return ProbeResult(
            name,
            "tcp",
            target,
            "REACHED",
            "CONNECTION SUCCEEDED - the sovereignty claim is BROKEN. Stop the demo "
            "and fix the network policy before continuing.",
            None,
            elapsed,
        )
    finally:
        sock.close()


def _dns_probe(
    ip: str, port: int, timeout_s: float, armed: bool = True
) -> ProbeResult:
    """A raw DNS query for a name we never expect to resolve. Sent over UDP to a
    pinned resolver IP so no system resolver config can quietly make it succeed."""
    target = f"{ip}:{port}"
    query = (
        b"\xab\xcd"          # transaction id
        b"\x01\x00"          # standard query, recursion desired
        b"\x00\x01\x00\x00\x00\x00\x00\x00"
        b"\x06kavach\x07canary\x07invalid\x00"
        b"\x00\x01\x00\x01"  # A, IN
    )
    started = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_s)
    try:
        sock.sendto(query, (ip, port))
        data, _ = sock.recvfrom(512)
    except OSError as exc:
        elapsed = (time.perf_counter() - started) * 1000
        verdict, detail, err_name = _classify(exc, elapsed, armed)
        return ProbeResult("dns_lookup", "udp", target, verdict, detail, err_name, elapsed)
    else:
        elapsed = (time.perf_counter() - started) * 1000
        return ProbeResult(
            "dns_lookup",
            "udp",
            target,
            "REACHED",
            f"a resolver answered with {len(data)} bytes - DNS is reachable and the "
            "sovereignty claim is BROKEN",
            None,
            elapsed,
        )
    finally:
        sock.close()


def run_canary(
    target_ip: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    actor: str = "local",
) -> CanaryResult:
    """Deliberately attempt egress and report what the enforcement did about it.

    `target_ip` defaults to whichever address actually tests something on this
    box: the routable one when a policy is armed to stop it, the documentation
    one when nothing is. An explicit argument is always honoured - a plant LAN
    has its own more convincing address, and that was always the intent.
    """
    db = get_db()
    audit = get_audit()
    before = db.egress_counts()["blocked"]

    armed = detect()
    if target_ip is None:
        target_ip = ENFORCED_TARGET_IP if armed.present else DEFAULT_TARGET_IP

    audit.append(
        "canary.start",
        actor=actor,
        target=target_ip,
        enforcement=armed.mechanism,
        note="deliberate egress attempt, self-test of the egress policy",
    )

    probes = [
        _dns_probe(target_ip, DEFAULT_DNS_PORT, timeout_s, armed.present),
        _tcp_probe("tls_connect", target_ip, DEFAULT_TLS_PORT, timeout_s, armed.present),
    ]

    for p in probes:
        db.record_egress(
            iface="canary",
            verdict="BLOCKED" if p.verdict == "BLOCKED" else p.verdict,
            proto=p.proto,
            dest=p.target,
            detail=p.detail,
        )

    reached = [p for p in probes if p.verdict == "REACHED"]
    inconclusive = [p for p in probes if p.verdict == "INCONCLUSIVE"]

    if reached:
        headline = (
            f"SOVEREIGNTY BREACH - {len(reached)} of {len(probes)} probes reached the "
            "outside world. Do not present this build."
        )
        ok, conclusive = False, True
    elif not armed.present:
        # The common case on a workstation, and the one that used to lie. Nothing
        # is stopping this process, so whatever the probes reported is the
        # network's answer rather than a policy's - including a refusal.
        headline = (
            "NOTHING IS ARMED on this computer, so this proves nothing yet. No egress "
            "policy is blocking this application - it can reach the network like any "
            f"other program, and {target_ip} simply did not answer. "
            f"{armed.fix} Then fire the canary again."
        )
        ok, conclusive = False, False
    elif inconclusive:
        headline = (
            f"{len(inconclusive)} of {len(probes)} probes timed out rather than being "
            "refused. Nothing got out, but a timeout is not proof - a policy is armed "
            f"({armed.mechanism}), so a refusal was expected and silence is not one."
        )
        ok, conclusive = False, False
    else:
        headline = (
            f"All {len(probes)} probes refused by the {armed.mechanism} policy, at an "
            f"address that has a route ({target_ip}). This application cannot reach "
            "the network; the computer it runs on still can."
        )
        ok, conclusive = True, True

    after = db.egress_counts()["blocked"]

    audit.append(
        "canary.result",
        actor=actor,
        ok=ok,
        conclusive=conclusive,
        enforcement=armed.mechanism,
        probes=[p.as_dict() for p in probes],
    )

    return CanaryResult(
        ok=ok,
        conclusive=conclusive,
        probes=probes,
        headline=headline,
        blocked_before=before,
        blocked_after=after,
        enforcement=armed,
    )
