"""The canary may not credit a firewall that is not there.

Found by asking the obvious question out loud: the claim is that THIS
APPLICATION cannot reach the network, and nobody unplugs the plant to run a
workbench - so on a machine that is online, what exactly was the canary proving?

Measured on the development box before this existed:

    203.0.113.1:443  tcp   timed out after 2000 ms
    1.1.1.1:443      tcp   REACHED in 5 ms

Nothing was blocking the gateway at all. The panel had nonetheless reported
"BLOCKED - kernel refused the connection outright (WSAEHOSTUNREACH) - this is
the firewall, not a slow network", because an RFC 5737 documentation address has
no route on any machine on earth and the routing table says so with the same
errno a firewall would use.
"""

from __future__ import annotations

import errno
import re
from pathlib import Path

import pytest

from kavach.sovereignty import canary as C
from kavach.sovereignty import enforcement as E

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- classifying


def test_a_refusal_with_nothing_armed_is_not_credited_to_a_firewall():
    """The bug itself. EHOSTUNREACH with no policy in place is the routing table
    declining to carry a packet to an address with no route - a fact about the
    address, true everywhere, and not a fact about this machine."""
    verdict, detail, _ = C._classify(OSError(errno.EHOSTUNREACH, "no route"), 11.0, armed=False)
    assert verdict == "INCONCLUSIVE"
    assert "no egress policy is armed" in detail
    assert "firewall" not in detail.lower()


def test_the_same_refusal_with_a_policy_armed_is_enforcement():
    verdict, detail, _ = C._classify(OSError(errno.EHOSTUNREACH, "no route"), 11.0, armed=True)
    assert verdict == "BLOCKED"
    assert "policy refused" in detail


@pytest.mark.parametrize("code", sorted(C.DEFINITIVE_BLOCK))
def test_no_errno_at_all_proves_enforcement_on_an_unarmed_box(code):
    """Every errno in the definitive set, not just the one that was seen. Each
    of them can come from routing as easily as from policy."""
    verdict, _, _ = C._classify(OSError(code, "refused"), 5.0, armed=False)
    assert verdict == "INCONCLUSIVE"


def test_an_unexpected_error_is_not_promoted_to_proof_either():
    verdict, detail, _ = C._classify(OSError(errno.EFAULT, "odd"), 5.0, armed=False)
    assert verdict == "INCONCLUSIVE"
    assert "unknown" in detail


def test_a_timeout_is_inconclusive_however_it_is_armed():
    """A timeout has never proved anything and still does not. UDP produces one
    almost every time: there is no handshake to refuse, so a dropped datagram
    and a datagram nobody answered are the same event seen from here."""
    for armed in (True, False):
        verdict, _, _ = C._classify(TimeoutError("timed out"), 2000.0, armed=armed)
        assert verdict == "INCONCLUSIVE"


# ------------------------------------------------------------------ the run


class _Armed:
    """Stands in for enforcement.detect()."""

    def __init__(self, present, mechanism="windows-firewall"):
        self.present = present
        self.mechanism = mechanism if present else "none"
        self.scope = "application" if present else "none"
        self.detail = "test"
        self.fix = "" if present else "arm it with the deploy script"

    def as_dict(self):
        return {"present": self.present, "mechanism": self.mechanism,
                "scope": self.scope, "detail": self.detail, "fix": self.fix}


def test_an_unarmed_box_can_never_report_a_passing_canary(monkeypatch):
    """Even if every probe fails. `ok` means "egress is enforced here", and with
    nothing armed that sentence has no basis whatever the sockets did."""
    monkeypatch.setattr(C, "detect", lambda: _Armed(False))
    monkeypatch.setattr(C, "_dns_probe", lambda *a, **k: C.ProbeResult(
        "dns_lookup", "udp", "t", "BLOCKED", "refused"))
    monkeypatch.setattr(C, "_tcp_probe", lambda *a, **k: C.ProbeResult(
        "tls_connect", "tcp", "t", "BLOCKED", "refused"))

    result = C.run_canary(actor="test")
    assert result.ok is False
    assert result.conclusive is False
    assert "NOTHING IS ARMED" in result.headline


def test_the_unarmed_headline_says_what_to_do_about_it(monkeypatch):
    monkeypatch.setattr(C, "detect", lambda: _Armed(False))
    monkeypatch.setattr(C, "_dns_probe", lambda *a, **k: C.ProbeResult(
        "dns_lookup", "udp", "t", "INCONCLUSIVE", "silence"))
    monkeypatch.setattr(C, "_tcp_probe", lambda *a, **k: C.ProbeResult(
        "tls_connect", "tcp", "t", "INCONCLUSIVE", "silence"))
    result = C.run_canary(actor="test")
    assert "arm it with the deploy script" in result.headline


def test_a_breach_outranks_an_unarmed_box(monkeypatch):
    """Reaching the network is the loudest thing this system can report, and it
    must not be softened into "nothing is armed" - the two are different
    failures and only one of them means data left."""
    monkeypatch.setattr(C, "detect", lambda: _Armed(False))
    monkeypatch.setattr(C, "_dns_probe", lambda *a, **k: C.ProbeResult(
        "dns_lookup", "udp", "t", "REACHED", "got an answer"))
    monkeypatch.setattr(C, "_tcp_probe", lambda *a, **k: C.ProbeResult(
        "tls_connect", "tcp", "t", "INCONCLUSIVE", "silence"))
    result = C.run_canary(actor="test")
    assert "BREACH" in result.headline


def test_an_armed_box_that_refuses_everything_passes(monkeypatch):
    monkeypatch.setattr(C, "detect", lambda: _Armed(True))
    monkeypatch.setattr(C, "_dns_probe", lambda *a, **k: C.ProbeResult(
        "dns_lookup", "udp", "t", "BLOCKED", "refused"))
    monkeypatch.setattr(C, "_tcp_probe", lambda *a, **k: C.ProbeResult(
        "tls_connect", "tcp", "t", "BLOCKED", "refused"))
    result = C.run_canary(actor="test")
    assert result.ok is True and result.conclusive is True
    assert "still can" in result.headline      # the machine keeps its network


# ------------------------------------------------------------- which target


def _target_of(monkeypatch, armed: bool) -> str:
    seen: list[str] = []

    def spy(ip, *a, **k):
        seen.append(ip)
        return C.ProbeResult("p", "tcp", ip, "INCONCLUSIVE", "x")

    monkeypatch.setattr(C, "detect", lambda: _Armed(armed))
    monkeypatch.setattr(C, "_dns_probe", spy)
    monkeypatch.setattr(C, "_tcp_probe", lambda name, ip, *a, **k: spy(ip))
    C.run_canary(actor="test")
    return seen[0]


def test_an_unarmed_box_probes_the_documentation_address(monkeypatch):
    """It proves nothing, but it also sends nothing to anybody else's machine.
    Probing a real host from a box with no containment would be reaching the
    internet on purpose to discover that we can."""
    assert _target_of(monkeypatch, armed=False) == C.DEFAULT_TARGET_IP


def test_an_armed_box_probes_an_address_that_actually_has_a_route(monkeypatch):
    """The only probe that separates policy from routing. Fired exclusively
    where something is armed to stop it - and if it gets out anyway, that is a
    breach and the canary says so."""
    assert _target_of(monkeypatch, armed=True) == C.ENFORCED_TARGET_IP


def test_an_explicit_target_is_always_honoured(monkeypatch):
    """A plant LAN has a more convincing address than either default."""
    seen = []
    monkeypatch.setattr(C, "detect", lambda: _Armed(True))
    monkeypatch.setattr(C, "_dns_probe", lambda ip, *a, **k: (
        seen.append(ip), C.ProbeResult("p", "udp", ip, "BLOCKED", "x"))[1])
    monkeypatch.setattr(C, "_tcp_probe", lambda name, ip, *a, **k: (
        seen.append(ip), C.ProbeResult("p", "tcp", ip, "BLOCKED", "x"))[1])
    C.run_canary(target_ip="10.20.30.40", actor="test")
    assert seen == ["10.20.30.40", "10.20.30.40"]


# ------------------------------------------------------------- the detector


def test_nothing_is_claimed_when_nothing_is_armed():
    """The live answer on whatever box runs the suite. A detector that reported
    enforcement here would be the original bug wearing a new coat."""
    found = E.detect()
    if not found.present:
        assert found.mechanism == "none"
        assert found.fix, "an unarmed box must say how to arm it"


def test_a_disabled_windows_rule_is_not_enforcement(monkeypatch):
    """A rule someone turned off is not a policy. This is the most likely way
    for the panel to be wrong in the reassuring direction on a venue machine."""
    monkeypatch.setattr(E.os, "name", "nt")
    monkeypatch.setattr(E.shutil, "which", lambda _: "netsh")
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: _Netsh(
        "Rule Name: Kavach egress block\nEnabled: No\nDirection: Out\n"
        "Program: C:\\k\\python.exe\nAction: Block\n"))
    assert E._windows_firewall() is None


def test_an_allow_rule_with_our_name_is_not_enforcement(monkeypatch):
    monkeypatch.setattr(E.os, "name", "nt")
    monkeypatch.setattr(E.shutil, "which", lambda _: "netsh")
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: _Netsh(
        "Rule Name: Kavach egress block\nEnabled: Yes\nDirection: Out\n"
        "Program: C:\\k\\python.exe\nAction: Allow\n"))
    assert E._windows_firewall() is None


def test_a_live_block_rule_is_reported_as_application_scoped(monkeypatch):
    monkeypatch.setattr(E.os, "name", "nt")
    monkeypatch.setattr(E.shutil, "which", lambda _: "netsh")
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: _Netsh(
        "Rule Name: Kavach egress block\nEnabled: Yes\nDirection: Out\n"
        "Program: C:\\k\\python.exe\nAction: Block\n"))
    found = E._windows_firewall()
    assert found is not None
    assert found.present is True
    assert found.scope == "application"
    assert not found.fix, "an armed box has nothing to fix"


def test_a_machine_wide_block_is_labelled_as_machine_wide(monkeypatch):
    """Blocking every program is a different claim from containing one, and the
    product's whole point is the second. Reported, but never as the first."""
    monkeypatch.setattr(E.os, "name", "nt")
    monkeypatch.setattr(E.shutil, "which", lambda _: "netsh")
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: _Netsh(
        "Rule Name: Kavach egress block\nEnabled: Yes\nDirection: Out\n"
        "Program: Any\nAction: Block\n"))
    assert E._windows_firewall().scope == "machine"


def test_no_matching_rule_reads_as_nothing_armed(monkeypatch):
    monkeypatch.setattr(E.os, "name", "nt")
    monkeypatch.setattr(E.shutil, "which", lambda _: "netsh")
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: _Netsh(
        "No rules match the specified criteria.\n"))
    assert E._windows_firewall() is None


class _Netsh:
    def __init__(self, stdout, returncode=0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


# ------------------------------------------------------- the two-file coupling


def test_the_deploy_script_writes_the_rule_the_detector_looks_for():
    """The rule name lives in a PowerShell script and in a Python constant. A
    rename in one is a machine that is genuinely contained and reports that
    nothing is armed - or worse, the reverse."""
    script = (REPO / "deploy" / "windows-firewall.ps1").read_text(encoding="utf-8")
    declared = re.search(r"\$RuleName\s*=\s*'([^']+)'", script)
    assert declared, "windows-firewall.ps1 no longer declares $RuleName"
    assert declared.group(1) == E.WINDOWS_RULE_NAME


def test_the_breach_banner_can_actually_fire():
    """The panel computes verdict as INCONCLUSIVE | BLOCKED | ALLOWED and then
    raised its loudest banner on `verdict === "REACHED"` - a value that string
    can never hold. The one warning that matters most was unreachable code, and
    nothing failed to say so because the banner simply never appeared."""
    js = (REPO / "services" / "kavach" / "ui" / "app.js").read_text(encoding="utf-8")
    block = re.search(r'caveat\("canary",(.*?)"sovereignty"\);', js, re.DOTALL)
    assert block, "the canary caveat has been renamed or removed"
    condition = block.group(1)
    assert '"REACHED"' not in condition, (
        "the banner is gated on a verdict the panel never produces"
    )
    assert '"ALLOWED"' in condition


def test_the_deploy_script_blocks_one_program_and_not_the_machine():
    """The entire point. `action=block` with no `program=` would take the
    machine off the network, which is the thing this product must never need."""
    script = (REPO / "deploy" / "windows-firewall.ps1").read_text(encoding="utf-8")
    assert "action=block" in script
    assert '"program=$exe"' in script
    assert "-Disarm" in script, "a block that cannot be removed is not acceptable"
