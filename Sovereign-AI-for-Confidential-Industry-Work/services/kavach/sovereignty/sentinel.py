"""Egress sentinel (section 02.1).

The dashboard's zero has to be a *measurement*, not an assertion. The sentinel
samples real interface counters and the nftables drop counter, so the number on
screen is derived from the kernel rather than from our own good intentions.

Sources, in descending order of trustworthiness:
  1. `nft list counter inet kavach egress_drop`  - the firewall's own count
  2. per-interface tx counters, from the kernel of whichever box this is:
     /proc/net/dev on Linux, GetIfTable2 on Windows (see `netcounters.py`)

Both platforms can therefore measure. What Windows still cannot supply is (1) -
there is no nftables drop counter, so `blocked_packets` there is Kavach's own
record of what *it* refused, not proof that a kernel dropped anything. The
sentinel says which of the two it is, in `mode` and in `detail`, every sample.

If neither counter source can be read, the sentinel reports DEGRADED rather
than 0. A monitor that cannot see is not the same as a monitor that sees
nothing, and conflating the two is exactly the failure the canary exists to
catch.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..store import Database, get_db
from .netcounters import CounterRead, InterfaceSample, read_counters

__all__ = ["InterfaceSample", "Sentinel", "SentinelState", "get_sentinel"]


@dataclass
class SentinelState:
    healthy: bool
    mode: str                       # nftables | proc | iphlpapi | degraded
    external_packets: int           # THE number on the dashboard
    blocked_packets: int
    interfaces: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    sampled_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "mode": self.mode,
            "external_packets": self.external_packets,
            "blocked_packets": self.blocked_packets,
            "interfaces": self.interfaces,
            "detail": self.detail,
            "sampled_at": self.sampled_at,
        }


class Sentinel:
    """Samples on demand and on a background tick. Cheap enough to poll at 1 Hz."""

    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.Lock()
        self._baseline: dict[str, InterfaceSample] = {}
        self._state = SentinelState(
            healthy=False, mode="degraded", external_packets=0, blocked_packets=0
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.establish_baseline()

    # ---------------------------------------------------------------- sources

    def _read_counters(self) -> CounterRead:
        """Overridable seam: the tests drive the degraded path through this."""
        return read_counters()

    def _read_nft_drop_counter(self) -> int | None:
        """The firewall's own count of what it refused. This is the number the
        canary makes tick, and it is the strongest evidence we can show."""
        if not shutil.which("nft"):
            return None
        try:
            out = subprocess.run(
                ["nft", "-j", "list", "counter", "inet", "kavach", "egress_drop"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if out.returncode != 0:
                return None
            import json

            data = json.loads(out.stdout)
            for item in data.get("nftables", []):
                counter = item.get("counter")
                if counter and counter.get("name") == "egress_drop":
                    return int(counter.get("packets", 0))
        except (subprocess.SubprocessError, ValueError, KeyError, OSError):
            return None
        return None

    # ---------------------------------------------------------------- sampling

    def establish_baseline(self) -> None:
        """Counters are cumulative since boot; the demo cares about deltas since
        Kavach started. Called once at startup and by `POST /api/sovereignty/reset`."""
        with self._lock:
            self._baseline = self._read_counters().samples

    def sample(self) -> SentinelState:
        read = self._read_counters()
        now = read.samples
        nft_drops = self._read_nft_drop_counter()

        if not now:
            state = SentinelState(
                healthy=False,
                mode="degraded",
                external_packets=0,
                blocked_packets=self.db.egress_counts()["blocked"],
                detail=(
                    # Plain words: this line is read by whoever is standing in
                    # front of the panel, not by whoever wrote the counter.
                    "This box cannot be asked how many packets it sent - "
                    f"{read.reason or 'no interface counter could be read'}. "
                    "Showing zero would be a guess, so nothing is shown."
                ),
                sampled_at=time.time(),
            )
            with self._lock:
                self._state = state
            return state

        external_delta = 0
        ifaces: list[dict[str, Any]] = []
        anomalies: list[str] = []
        with self._lock:
            baseline = dict(self._baseline)

        for name, s in sorted(now.items()):
            base = baseline.get(name)
            # Two ways the baseline stops being a valid subtrahend, both far more
            # likely on Windows, where adapters come and go (Wi-Fi Direct, WAN
            # miniports, a NIC disabled and re-enabled) than they do on a Linux
            # appliance. Both are resolved the same way: count from zero. That
            # over-reports traffic that predates the baseline, and over-reporting
            # is the only safe direction for an egress alarm - it can cry wolf,
            # but it can never let a packet through unseen. A silent max(0, ...)
            # on a reset counter would read zero forever after.
            if base is None:
                d_pkts, d_bytes = s.tx_packets, s.tx_bytes
                reason = "new since baseline"
            elif s.tx_packets < base.tx_packets or s.tx_bytes < base.tx_bytes:
                d_pkts, d_bytes = s.tx_packets, s.tx_bytes
                reason = "counter reset since baseline"
            else:
                d_pkts = s.tx_packets - base.tx_packets
                d_bytes = s.tx_bytes - base.tx_bytes
                reason = ""

            if reason and d_pkts:
                anomalies.append(f"{name} ({reason})")
            if not s.internal:
                external_delta += d_pkts
            ifaces.append(
                {
                    "iface": name,
                    "internal": s.internal,
                    "up": s.up,
                    "tx_packets_delta": d_pkts,
                    "tx_bytes_delta": d_bytes,
                    "baseline_anomaly": reason,
                }
            )

        blocked = nft_drops if nft_drops is not None else self.db.egress_counts()["blocked"]
        mode = "nftables" if nft_drops is not None else read.source

        state = SentinelState(
            healthy=True,
            mode=mode,
            external_packets=external_delta,
            blocked_packets=blocked,
            interfaces=ifaces,
            detail=(
                "counting tx frames on every non-internal interface since baseline"
                + (
                    "; drop count read from the nftables egress_drop counter"
                    if nft_drops is not None
                    # Without nftables the blocked figure is our own bookkeeping.
                    # Saying so is the difference between "the kernel refused it"
                    # and "we decided not to try", which are not the same claim.
                    else "; no firewall drop counter on this box, so the blocked "
                    "figure is Kavach policy denials, not kernel drops"
                )
                + (
                    # A jump the operator cannot explain is worse than a big
                    # number they can, so name the interface that caused it.
                    "; counted in full from boot because the baseline no longer "
                    "applies to " + ", ".join(anomalies)
                    if anomalies
                    else ""
                )
            ),
            sampled_at=time.time(),
        )
        with self._lock:
            self._state = state
        return state

    @property
    def state(self) -> SentinelState:
        with self._lock:
            return self._state

    # ---------------------------------------------------------------- lifecycle

    def start(self, interval_s: float = 1.0) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _loop() -> None:
            while not self._stop.wait(interval_s):
                try:
                    self.sample()
                except Exception as exc:  # noqa: BLE001 - must never take the app down
                    # A sentinel that dies quietly is exactly the failure the canary
                    # exists to catch, so a sampling error marks the state UNHEALTHY
                    # rather than leaving the last good reading on screen.
                    with self._lock:
                        self._state.healthy = False
                        self._state.mode = "degraded"
                        self._state.detail = (
                            f"sentinel sampling failed: {type(exc).__name__}: {exc}"
                        )

        self._stop.clear()
        self._thread = threading.Thread(target=_loop, name="kavach-sentinel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


_sentinel: Sentinel | None = None


def get_sentinel() -> Sentinel:
    global _sentinel
    if _sentinel is None:
        _sentinel = Sentinel(get_db())
    return _sentinel
