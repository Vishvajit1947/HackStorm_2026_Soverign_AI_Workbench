"""The sentinel must measure on the box it is actually running on.

The headline claim of the product is a packet count. For a long time Windows
could not produce one and the panel honestly said so - but "honestly cannot
prove it" is still not proving it, and the demo box is a Windows laptop. These
tests pin the measurement itself, and pin the specific ways a Windows read
produces a *plausible* wrong number instead of an error.
"""

from __future__ import annotations

import platform

import pytest

from kavach.sovereignty.netcounters import (
    CounterRead,
    InterfaceSample,
    is_internal,
    read_counters,
)
from kavach.sovereignty.sentinel import Sentinel

WINDOWS = platform.system() == "Windows"


class _StubDB:
    """The sentinel only ever asks the database one question."""

    def __init__(self, blocked: int = 7) -> None:
        self._blocked = blocked

    def egress_counts(self) -> dict[str, int]:
        return {"blocked": self._blocked}


class _ScriptedSentinel(Sentinel):
    """A sentinel whose kernel is a list. Each sample pops the next read, so a
    test can stage a baseline and then a hostile follow-up."""

    def __init__(self, reads: list[CounterRead], db: _StubDB | None = None) -> None:
        self._reads = list(reads)
        super().__init__(db or _StubDB())  # type: ignore[arg-type]

    def _read_counters(self) -> CounterRead:
        if len(self._reads) > 1:
            return self._reads.pop(0)
        return self._reads[0]


def _sample(iface: str, pkts: int, byts: int, internal: bool = False) -> InterfaceSample:
    return InterfaceSample(iface=iface, tx_bytes=byts, tx_packets=pkts, internal=internal)


def _read(*samples: InterfaceSample, source: str = "iphlpapi") -> CounterRead:
    return CounterRead({s.iface: s for s in samples}, source)


# --------------------------------------------------------------- the real box


def test_this_machine_can_count_its_own_egress():
    """The whole point of the fix. If this fails the panel is back to showing a
    number nobody can defend, whatever the banner says."""
    read = read_counters()
    assert read.samples, (
        f"no interface counters on {platform.system()}: {read.reason}"
    )
    assert read.source in ("proc", "iphlpapi")
    assert read.reason == "", "a successful read must not also carry a failure reason"


def test_at_least_one_interface_has_actually_transmitted_something():
    """A reader that returns the right shape with every counter at zero is the
    failure this test exists for: a struct read at the wrong offset gives
    exactly that, and it looks like a perfectly air-gapped machine."""
    read = read_counters()
    assert any(s.tx_packets > 0 for s in read.samples.values()), (
        "every interface reports zero packets since boot, which no running "
        "machine does - the counter offsets are probably wrong"
    )


@pytest.mark.skipif(not WINDOWS, reason="the LWF duplication is Windows-only")
def test_filter_pseudo_interfaces_are_not_counted_twice():
    """Windows lists one Wi-Fi adapter up to six times - once real, the rest as
    LightWeight Filter rows carrying identical cumulative totals. Summing the
    table naively multiplies egress by six and still looks like a measurement."""
    read = read_counters()
    by_bytes: dict[int, list[str]] = {}
    for s in read.samples.values():
        if s.tx_bytes:
            by_bytes.setdefault(s.tx_bytes, []).append(s.iface)
    duplicated = {b: names for b, names in by_bytes.items() if len(names) > 1}
    assert not duplicated, f"the same counter is being counted more than once: {duplicated}"


@pytest.mark.skipif(not WINDOWS, reason="Windows interface naming")
def test_the_hyper_v_switches_do_not_inflate_external_egress():
    """A WSL or Docker Desktop packet bound for the internet transmits on
    vEthernet and then again on the physical NIC. Counting both double-charges
    exactly the traffic the demo is about."""
    assert is_internal("vEthernet (Default Switch)")
    assert is_internal("vEthernet (WSL (Hyper-V firewall))")
    assert is_internal("vSwitch (Default Switch)")
    assert is_internal("Loopback Pseudo-Interface 1")
    # ...and the patterns must not be so broad that they swallow a real NIC.
    assert not is_internal("Wi-Fi")
    assert not is_internal("Ethernet")
    assert not is_internal("Bluetooth Network Connection")


def test_linux_bridges_are_still_internal():
    """The Windows patterns were added to the same list; a regex typo there
    would quietly un-classify the docker bridge on the Linux demo box."""
    assert is_internal("lo")
    assert is_internal("docker0")
    assert is_internal("br-0123456789ab")
    assert is_internal("veth1a2b3c")
    assert not is_internal("eth0")
    assert not is_internal("wlan0")


def test_counters_do_not_run_backwards_between_two_reads():
    a, b = read_counters(), read_counters()
    for name, first in a.samples.items():
        second = b.samples.get(name)
        if second is None:
            continue
        assert second.tx_packets >= first.tx_packets, f"{name} packet counter went backwards"
        assert second.tx_bytes >= first.tx_bytes, f"{name} byte counter went backwards"


def test_reading_the_table_repeatedly_does_not_leak():
    """GetIfTable2 allocates and the sentinel samples at 1 Hz for the length of
    a demo. A missing FreeMibTable is a slow bleed nothing else would catch."""
    for _ in range(200):
        assert read_counters().samples


# ------------------------------------------------------- the sentinel on top


def test_a_reader_that_sees_nothing_is_degraded_and_says_why():
    """The house rule: a monitor that cannot see must not look like a monitor
    that sees nothing."""
    s = _ScriptedSentinel([CounterRead({}, "", "the kernel refused to answer")])
    st = s.sample()
    assert st.healthy is False
    assert st.mode == "degraded"
    assert st.external_packets == 0
    assert "the kernel refused to answer" in st.detail
    assert "guess" in st.detail.lower(), "the zero must be disclaimed, not just uncommented"


def test_the_mode_names_the_source_the_number_came_from():
    s = _ScriptedSentinel([_read(_sample("Wi-Fi", 10, 100), source="iphlpapi")])
    assert s.sample().mode == "iphlpapi"


def test_without_a_firewall_counter_the_blocked_figure_is_labelled_as_ours():
    """Counting is not blocking. Reporting Kavach policy denials under the same
    word the nftables drop counter uses would overclaim enforcement."""
    s = _ScriptedSentinel([_read(_sample("Wi-Fi", 10, 100))], _StubDB(blocked=3))
    st = s.sample()
    assert st.blocked_packets == 3
    assert "not kernel drops" in st.detail


def test_internal_interfaces_never_reach_the_dashboard_number():
    baseline = _read(_sample("Wi-Fi", 100, 1000), _sample("docker0", 100, 1000, internal=True))
    later = _read(_sample("Wi-Fi", 100, 1000), _sample("docker0", 900, 9000, internal=True))
    s = _ScriptedSentinel([baseline, later])
    assert s.sample().external_packets == 0


def test_an_interface_that_appears_after_the_baseline_is_counted_not_ignored():
    """Plug in an Ethernet cable mid-run and the adapter has no baseline row.
    Treating that as zero would let a whole NIC transmit unseen."""
    s = _ScriptedSentinel([
        _read(_sample("Wi-Fi", 100, 1000)),
        _read(_sample("Wi-Fi", 100, 1000), _sample("Ethernet", 500, 5000)),
    ])
    st = s.sample()
    assert st.external_packets == 500
    row = next(i for i in st.interfaces if i["iface"] == "Ethernet")
    assert row["baseline_anomaly"] == "new since baseline"
    assert "Ethernet" in st.detail, "an unexplained jump must name its cause"


def test_a_counter_that_resets_is_recounted_rather_than_silently_zeroed():
    """Disable and re-enable an adapter and its since-boot counter restarts. A
    plain max(0, now - baseline) then reads zero forever after, which is the
    exact shape of a monitor that has stopped working while looking healthy."""
    s = _ScriptedSentinel([
        _read(_sample("Wi-Fi", 100_000, 1_000_000)),
        _read(_sample("Wi-Fi", 42, 4200)),
    ])
    st = s.sample()
    assert st.external_packets == 42
    row = next(i for i in st.interfaces if i["iface"] == "Wi-Fi")
    assert row["baseline_anomaly"] == "counter reset since baseline"


def test_a_normal_delta_is_not_flagged_as_an_anomaly():
    """The anomaly path over-reports on purpose; if it fires on every ordinary
    sample the detail line becomes noise nobody reads."""
    s = _ScriptedSentinel([
        _read(_sample("Wi-Fi", 100, 1000)),
        _read(_sample("Wi-Fi", 130, 1400)),
    ])
    st = s.sample()
    assert st.external_packets == 30
    assert st.interfaces[0]["baseline_anomaly"] == ""
    assert "baseline no longer applies" not in st.detail
