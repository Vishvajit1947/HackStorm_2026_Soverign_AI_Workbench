"""The offline guarantee, and the run sheet (§09, §11).

`scripts/offline_check.py` is the thing you run in front of a judge. This is the
thing that stops the property regressing between demos, which is the version that
actually matters: a check nobody runs on a Tuesday is a check that fails on the
Friday.

The one that earns its keep is `test_the_shipped_ui_has_no_absolute_url`. A
single `<script src="https://cdn...">` added in a hurry turns a sovereign
workbench into a blank panel on a plant network - and into a line in someone
else's request log if the network is only half closed. It would not fail any
other test in this suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from offline_check import (
    CSS_URL,
    EXTERNAL,
    JUSTIFIED,
    NAMESPACE_URIS,
    UI_DIR,
    _inside_namespace_literal,
    check_csp,
    check_dependencies,
    check_ui_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- the bundle

def test_the_shipped_ui_has_no_absolute_url():
    """No CDN, no web font, no favicon from a host, no tracking pixel."""
    offenders = []
    for path in sorted(UI_DIR.rglob("*")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for match in EXTERNAL.finditer(line):
                if _inside_namespace_literal(line, match.start(), match.end()):
                    continue
                offenders.append(f"{path.name}: {match.group(0)}")
    assert offenders == []


def test_even_loopback_is_not_allowed_in_the_bundle():
    """The panel is served same-origin so it never needs to name a host. Hard
    coding http://127.0.0.1:8420 breaks the moment it is served on the demo
    box's real address or a different port."""
    for path in UI_DIR.rglob("*.js"):
        text = path.read_text(encoding="utf-8")
        assert "127.0.0.1" not in text
        assert "localhost" not in text


def test_no_css_url_points_at_a_host():
    for path in UI_DIR.rglob("*.css"):
        for match in CSS_URL.finditer(path.read_text(encoding="utf-8")):
            assert not match.group(1).startswith(("http", "//"))


def test_the_bundle_check_passes_on_what_we_ship():
    assert check_ui_bundle() == []


def test_the_bundle_check_actually_catches_a_cdn(tmp_path, monkeypatch):
    """A checker that cannot fail is decoration. Point it at a poisoned bundle."""
    import offline_check

    poisoned = tmp_path / "ui"
    poisoned.mkdir()
    (poisoned / "index.html").write_text(
        '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
        encoding="utf-8",
    )
    monkeypatch.setattr(offline_check, "UI_DIR", poisoned)

    failures = offline_check.check_ui_bundle()
    assert failures and "cdn.jsdelivr.net" in failures[0]


def test_the_bundle_check_catches_a_web_font_in_css(tmp_path, monkeypatch):
    import offline_check

    poisoned = tmp_path / "ui"
    poisoned.mkdir()
    (poisoned / "app.css").write_text(
        "@import url(https://fonts.googleapis.com/css?family=Inter);", encoding="utf-8"
    )
    monkeypatch.setattr(offline_check, "UI_DIR", poisoned)
    assert offline_check.check_ui_bundle()


def test_a_data_uri_is_not_an_external_reference():
    """The favicon is `data:,` precisely so there is no request to make."""
    assert not EXTERNAL.search('<link rel="icon" href="data:,">')
    assert not CSS_URL.search("background: url(data:image/png;base64,AAA)")


def test_an_ordinary_js_comment_is_not_mistaken_for_a_url():
    """A checker that cries wolf is a checker somebody disables."""
    assert not EXTERNAL.search("// helpers below")
    assert not EXTERNAL.search("  // see app.js for the fetch wrapper")
    assert not EXTERNAL.search("a / b // c")


def test_a_protocol_relative_url_is_still_caught():
    assert EXTERNAL.search('<script src="//cdn.example.com/x.js">')


# ---------------------------------------------------------------- the CSP

def test_the_page_carries_a_self_only_csp():
    assert check_csp() == []


def test_the_csp_is_parsed_whole_and_not_truncated_at_the_first_quote():
    """A CSP is full of 'self' inside a double-quoted attribute. Getting this
    wrong reports every directive as missing, which is the false alarm that gets
    a check switched off."""
    text = (UI_DIR / "index.html").read_text(encoding="utf-8")
    assert "connect-src 'self'" in text
    assert "base-uri 'none'" in text


# ---------------------------------------------------------------- deps

def test_every_declared_dependency_is_already_installed():
    """An offline install that needs one wheel from PyPI is not an offline
    install, and the venue's wifi is where you find out."""
    assert check_dependencies() == []


def test_the_justified_hostname_list_is_explained():
    """Each exception is a decision. An entry with no reason is an entry nobody
    can defend when a judge points at it."""
    for host, reason in JUSTIFIED.items():
        assert reason and len(reason) > 8, host


# ---------------------------------------------------------------- end to end

def test_offline_check_exits_zero_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "offline_check.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------- run sheet

@pytest.fixture(scope="module")
def sheet_text():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sheet

    return run_sheet, run_sheet.render(run_sheet.gather(), with_tests=False)


def test_the_run_sheet_leads_with_what_was_not_proved(sheet_text):
    """Every other status document opens with achievements. The list a judge
    builds during Q&A is the unproven one, and handing it over is the difference
    between a caveat and a discovery."""
    _, text = sheet_text
    not_proved = text.index("What this box did NOT prove")
    proved = text.index("What this box did prove")
    assert not_proved < proved


def test_the_run_sheet_states_where_the_quality_numbers_came_from(sheet_text):
    """Either answer is fine; silence is not.

    This used to assert the seeded wording, which made it a test of the day it
    was written rather than of the sheet. Once the eval harness actually ran,
    the sheet correctly stopped calling the numbers placeholders and the test
    failed for being right. What must always hold is that the sheet SAYS which
    it is - a judge asking "where does 0.65 come from?" gets an answer either
    way, and the way to re-derive it stays printed.
    """
    _, text = sheet_text
    seeded = "hand-written placeholders" in text
    measured = "quality_measured" in text or "measured" in text.lower()
    assert seeded or measured, "the sheet does not say where the numbers came from"
    assert "eval_harness.py" in text, "no way to re-derive the numbers is printed"


def test_the_run_sheet_says_the_canary_is_unproven_here(sheet_text):
    _, text = sheet_text
    assert "nftables" in text
    assert "a timeout is not proof" in text


def test_every_claim_carries_a_status_and_a_way_to_re_derive_it(sheet_text):
    run_sheet, _ = sheet_text
    sheet = run_sheet.gather()
    assert sheet.claims
    for claim in sheet.claims:
        assert claim.status in (run_sheet.PROVEN, run_sheet.PARTIAL, run_sheet.NOT_PROVEN)
        assert claim.evidence.strip(), claim.claim
        assert claim.command.strip(), claim.claim


def test_the_run_sheet_covers_every_layer(sheet_text):
    run_sheet, _ = sheet_text
    areas = {c.area for c in run_sheet.gather().claims}
    assert {"L0 sovereignty", "L1 model fabric", "L2 router",
            "L3 knowledge", "L4 tools", "L5 agent", "L6 interface"} <= areas


def test_the_run_sheet_does_not_run_the_canary(sheet_text, monkeypatch):
    """A document generator that changes the system it documents is a document
    you cannot trust - and the canary deliberately attempts egress."""
    from kavach import sovereignty

    def refuse(*args, **kwargs):
        raise AssertionError("run_sheet must not fire the canary")

    monkeypatch.setattr(sovereignty, "run_canary", refuse)
    run_sheet, _ = sheet_text
    run_sheet.gather()


def test_the_run_sheet_reports_a_mock_backend_as_unproven_weights(sheet_text):
    run_sheet, text = sheet_text
    claim = next(c for c in run_sheet.gather().claims
                 if c.claim.startswith("Every weight file"))
    # The suite runs on the mock backend, where there are no weights at all.
    assert claim.status == run_sheet.NOT_PROVEN
    assert "SIMULATED" in claim.evidence
    assert "SIMULATED" in text


# ---------------------------------------------------------------- the bundle wiring

"""The UI has no build step, so nothing catches a typo'd element id or a syntax
error before a judge sees a blank panel. These two tests are the bundler."""


def test_every_element_the_script_reaches_for_exists_in_the_page():
    import re

    ui = ROOT / "services" / "kavach" / "ui"
    html = (ui / "index.html").read_text(encoding="utf-8")

    # Every script the page loads, not just app.js - splitting the bundle must
    # not silently shrink what this check covers.
    used = set()
    for script in sorted(ui.glob("*.js")):
        used |= set(re.findall(r'\$\("([^"]+)"\)',
                               script.read_text(encoding="utf-8")))
    declared = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted(used - declared)

    # A missing id is `null.textContent = ...` at runtime: the panel stops
    # updating from that line on, with nothing in the UI to say why.
    assert missing == [], f"app.js reaches for ids that do not exist: {missing}"


def test_every_asset_the_page_loads_actually_exists():
    """A 404 on a script leaves a page that renders and does nothing."""
    import re

    ui = ROOT / "services" / "kavach" / "ui"
    html = (ui / "index.html").read_text(encoding="utf-8")

    referenced = set(re.findall(r'(?:src|href)="([^":]+)"', html))
    for ref in referenced:
        assert (ui / ref).exists(), f"index.html references missing asset {ref}"

    # And every script that exists is actually loaded - a file nobody includes
    # is a file whose checks nobody runs.
    for script in ui.glob("*.js"):
        assert f'src="{script.name}"' in html, f"{script.name} is never loaded"


# ---------------------------------------------------- the namespace exemption

def test_the_svg_namespace_is_not_treated_as_a_network_reference():
    """createElementNS needs the literal string; the browser never fetches it."""
    line = 'const NS = "http://www.w3.org/2000/svg";'
    match = EXTERNAL.search(line)
    assert match, "the regex should still see it"
    assert _inside_namespace_literal(line, match.start(), match.end())


def test_the_exemption_does_not_whitelist_the_whole_host():
    """`https://www.w3.org/TR/svg` is a real page on a real server. Exempting a
    host rather than a literal is how a checker stops checking."""
    line = 'see https://www.w3.org/TR/SVG2/ for the spec'
    match = EXTERNAL.search(line)
    assert match
    assert not _inside_namespace_literal(line, match.start(), match.end())


def test_every_exempt_namespace_is_a_w3c_namespace_name():
    for uri in NAMESPACE_URIS:
        assert uri.startswith("http://www.w3.org/")


def test_every_class_the_markup_uses_is_defined_in_the_stylesheet():
    """No build step means no bundler to warn about a typo'd class name.

    `el("div", "notice bad")` against a stylesheet that spells it `.notice-bad`
    renders unstyled: readable, wrong, and easy to miss until it is on a
    projector. This is the other half of the id cross-check.
    """
    import re

    ui = ROOT / "services" / "kavach" / "ui"
    defined = set(re.findall(r"\.([a-zA-Z][\w-]*)",
                             (ui / "app.css").read_text(encoding="utf-8")))

    used: set[str] = set()
    for group in re.findall(r'class="([^"]+)"',
                            (ui / "index.html").read_text(encoding="utf-8")):
        used |= set(group.split())
    for script in ui.glob("*.js"):
        text = script.read_text(encoding="utf-8")
        for group in re.findall(r'el\(\s*"[a-z0-9]+"\s*,\s*"([^"]+)"', text):
            used |= set(group.split())
        for group in re.findall(r'className = "([^"$]+)"', text):
            used |= set(group.split())
        for name in re.findall(r'classList\.(?:add|toggle|remove)\("([^"]+)"', text):
            used.add(name)

    missing = sorted(used - defined)
    assert missing == [], f"classes used but never styled: {missing}"
