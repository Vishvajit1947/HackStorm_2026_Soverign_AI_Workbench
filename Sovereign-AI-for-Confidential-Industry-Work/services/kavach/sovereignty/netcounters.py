"""Per-interface transmit counters, on Linux *and* on Windows.

The sentinel's job is to show a number the kernel produced. Linux hands that
over in /proc/net/dev. Windows keeps the same counters - it just does not expose
them as a file, which is why the first version of the sentinel reported
DEGRADED on this platform. It reads them here through the IP Helper API
(`GetIfTable2` in iphlpapi.dll), via ctypes, so the demo box measures for real
without staging a wheel offline.

Two Windows-specific traps, both of which produce a *plausible* wrong number
rather than an error - the worst kind:

1. **Filter (LWF) pseudo-interfaces duplicate their adapter's counters.**
   One Wi-Fi card shows up six times - once real, five times as
   "Wi-Fi-QoS Packet Scheduler-0000" and friends - each reporting the *same*
   cumulative totals. Summing the table naively multiplies egress by six and
   still looks like a measurement. `MIB_IF_ROW2.InterfaceAndOperStatusFlags`
   bit 1 is `FilterInterface`; every one of those rows has it set, and every
   real adapter does not. That bit is the whole deduplication.
2. **Hyper-V virtual switch adapters carry traffic that is counted again on the
   physical NIC.** A WSL container reaching the internet transmits on
   `vEthernet (WSL ...)` and then on `Wi-Fi`. These are the exact analogue of
   `docker0`/`br-*` on Linux and are classified internal for the same reason:
   the frame that actually leaves the box is counted once, on the adapter that
   sends it.

Neither platform's counter distinguishes a packet to the LAN router from a
packet to the internet - both are frames handed to a non-internal adapter. That
is what the dashboard number means, on both platforms.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from pathlib import Path

PROC_NET_DEV = Path("/proc/net/dev")

# Interfaces that carry only intra-stack traffic. Frames here are expected and
# are NOT external egress: loopback, the docker bridge for kavach_core, the veth
# pairs docker creates for containers on it, and - on Windows - the Hyper-V
# virtual switch adapters that back WSL and Docker Desktop.
INTERNAL_IFACE_PATTERNS = [
    re.compile(r"^lo$"),
    re.compile(r"^br-[0-9a-f]{12}$"),
    re.compile(r"^veth"),
    re.compile(r"^docker0$"),
    re.compile(r"^vEthernet \(", re.IGNORECASE),
    re.compile(r"^vSwitch \(", re.IGNORECASE),
    re.compile(r"^Loopback Pseudo-Interface", re.IGNORECASE),
    re.compile(r"^VirtualBox Host-Only", re.IGNORECASE),
    re.compile(r"^VMware Network Adapter VMnet", re.IGNORECASE),
]

# IANA ifType, as reported by MIB_IF_ROW2.Type.
IF_TYPE_SOFTWARE_LOOPBACK = 24

# MIB_IF_ROW2.InterfaceAndOperStatusFlags bitfield, in declaration order.
_FLAG_HARDWARE_INTERFACE = 0x01
_FLAG_FILTER_INTERFACE = 0x02


def is_internal(iface: str, if_type: int | None = None) -> bool:
    if if_type == IF_TYPE_SOFTWARE_LOOPBACK:
        return True
    return any(p.match(iface) for p in INTERNAL_IFACE_PATTERNS)


@dataclass
class InterfaceSample:
    iface: str
    tx_bytes: int
    tx_packets: int
    internal: bool
    up: bool = True


@dataclass
class CounterRead:
    """A read attempt. `reason` is non-empty exactly when `samples` is empty, so
    the sentinel can say *why* it cannot count rather than showing a zero."""

    samples: dict[str, InterfaceSample]
    source: str          # proc | iphlpapi | ""
    reason: str = ""


# --------------------------------------------------------------------- linux


def read_proc_net_dev() -> CounterRead:
    if not PROC_NET_DEV.exists():
        return CounterRead({}, "", f"{PROC_NET_DEV} does not exist on this system")
    samples: dict[str, InterfaceSample] = {}
    for line in PROC_NET_DEV.read_text(encoding="utf-8").splitlines()[2:]:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        cols = rest.split()
        if len(cols) < 10:
            continue
        samples[name] = InterfaceSample(
            iface=name,
            tx_bytes=int(cols[8]),
            tx_packets=int(cols[9]),
            internal=is_internal(name),
        )
    if not samples:
        return CounterRead({}, "", f"{PROC_NET_DEV} listed no interfaces")
    return CounterRead(samples, "proc")


# ------------------------------------------------------------------- windows

_iphlpapi = None
_MIB_IF_TABLE2 = None
_MIB_IF_ROW2 = None


def _load_iphlpapi():
    """Bind GetIfTable2 once. Returns None on anything that is not Windows or
    does not have the entry point, which the caller turns into a DEGRADED
    sentinel rather than a zero."""
    global _iphlpapi, _MIB_IF_TABLE2, _MIB_IF_ROW2
    if _iphlpapi is not None:
        return _iphlpapi
    if platform.system() != "Windows":
        return None

    import ctypes
    from ctypes import wintypes

    ulong64 = ctypes.c_ulonglong

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class MIB_IF_ROW2(ctypes.Structure):
        # Field-for-field from netioapi.h. The layout is load-bearing: get one
        # field wrong and every counter after it reads as garbage that still
        # looks like a number. IF_MAX_STRING_SIZE is 256, so the strings are 257
        # wchars including the terminator; IF_MAX_PHYS_ADDRESS_LENGTH is 32.
        _fields_ = [
            ("InterfaceLuid", ulong64),
            ("InterfaceIndex", wintypes.ULONG),
            ("InterfaceGuid", GUID),
            ("Alias", wintypes.WCHAR * 257),
            ("Description", wintypes.WCHAR * 257),
            ("PhysicalAddressLength", wintypes.ULONG),
            ("PhysicalAddress", ctypes.c_ubyte * 32),
            ("PermanentPhysicalAddress", ctypes.c_ubyte * 32),
            ("Mtu", wintypes.ULONG),
            ("Type", wintypes.ULONG),
            ("TunnelType", ctypes.c_int),
            ("MediaType", ctypes.c_int),
            ("PhysicalMediumType", ctypes.c_int),
            ("AccessType", ctypes.c_int),
            ("DirectionType", ctypes.c_int),
            # Eight 1-bit BOOLEAN bitfields packed into one byte; ctypes adds the
            # three padding bytes before the next 4-byte enum for us.
            ("InterfaceAndOperStatusFlags", ctypes.c_ubyte),
            ("OperStatus", ctypes.c_int),
            ("AdminStatus", ctypes.c_int),
            ("MediaConnectState", ctypes.c_int),
            ("NetworkGuid", GUID),
            ("ConnectionType", ctypes.c_int),
            ("TransmitLinkSpeed", ulong64),
            ("ReceiveLinkSpeed", ulong64),
            ("InOctets", ulong64),
            ("InUcastPkts", ulong64),
            ("InNUcastPkts", ulong64),
            ("InDiscards", ulong64),
            ("InErrors", ulong64),
            ("InUnknownProtos", ulong64),
            ("InUcastOctets", ulong64),
            ("InMulticastOctets", ulong64),
            ("InBroadcastOctets", ulong64),
            ("OutOctets", ulong64),
            ("OutUcastPkts", ulong64),
            ("OutNUcastPkts", ulong64),
            ("OutDiscards", ulong64),
            ("OutErrors", ulong64),
            ("OutUcastOctets", ulong64),
            ("OutMulticastOctets", ulong64),
            ("OutBroadcastOctets", ulong64),
            ("OutQLen", ulong64),
        ]

    class MIB_IF_TABLE2(ctypes.Structure):
        _fields_ = [("NumEntries", wintypes.ULONG), ("Table", MIB_IF_ROW2 * 1)]

    try:
        dll = ctypes.WinDLL("iphlpapi.dll")
        dll.GetIfTable2.argtypes = [ctypes.POINTER(ctypes.POINTER(MIB_IF_TABLE2))]
        dll.GetIfTable2.restype = wintypes.ULONG
        dll.FreeMibTable.argtypes = [ctypes.c_void_p]
        dll.FreeMibTable.restype = None
    except (OSError, AttributeError):
        return None

    _MIB_IF_ROW2 = MIB_IF_ROW2
    _MIB_IF_TABLE2 = MIB_IF_TABLE2
    _iphlpapi = dll
    return dll


def read_windows_if_table() -> CounterRead:
    dll = _load_iphlpapi()
    if dll is None:
        return CounterRead(
            {}, "", "iphlpapi.dll GetIfTable2 is not available on this system"
        )

    import ctypes

    table_ptr = ctypes.POINTER(_MIB_IF_TABLE2)()
    rc = dll.GetIfTable2(ctypes.byref(table_ptr))
    if rc != 0:  # anything but NO_ERROR
        return CounterRead({}, "", f"GetIfTable2 failed with status {rc}")

    samples: dict[str, InterfaceSample] = {}
    try:
        count = table_ptr.contents.NumEntries
        rows = ctypes.cast(
            table_ptr.contents.Table, ctypes.POINTER(_MIB_IF_ROW2 * count)
        ).contents
        for row in rows:
            if row.InterfaceAndOperStatusFlags & _FLAG_FILTER_INTERFACE:
                # See module docstring: these mirror another row's counters.
                continue
            name = row.Alias or row.Description or f"if{row.InterfaceIndex}"
            if name in samples:
                # Aliases are meant to be unique; if the box disagrees, keep both
                # rather than silently dropping one adapter's traffic.
                name = f"{name}#{row.InterfaceIndex}"
            samples[name] = InterfaceSample(
                iface=name,
                tx_bytes=int(row.OutOctets),
                # NUcast is broadcast/multicast - mDNS, ARP, SSDP. It leaves the
                # box like anything else, so it counts.
                tx_packets=int(row.OutUcastPkts) + int(row.OutNUcastPkts),
                internal=is_internal(name, row.Type),
                up=row.OperStatus == 1,  # IfOperStatusUp
            )
    finally:
        # GetIfTable2 allocates; leaking it once a second would be a slow bleed.
        dll.FreeMibTable(table_ptr)

    if not samples:
        return CounterRead({}, "", "GetIfTable2 returned no non-filter interfaces")
    return CounterRead(samples, "iphlpapi")


# ------------------------------------------------------------------- dispatch


def read_counters() -> CounterRead:
    """Whatever this box can actually measure, or an empty read with a reason."""
    if platform.system() == "Windows":
        return read_windows_if_table()
    return read_proc_net_dev()
