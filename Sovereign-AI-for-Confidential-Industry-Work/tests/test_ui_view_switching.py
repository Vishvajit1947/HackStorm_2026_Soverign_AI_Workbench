"""Only one panel may be on screen at a time.

theme.js stamps `data-initial-view` on <html> in the document head so the right
panel is painted before app.js loads - without it the Overview flashes on every
reload. The CSS behind it uses `display: block !important` on an id selector,
which outranks `.hide { display: none !important }`.

That is fine for the few milliseconds before JS runs and wrong forever after:
the named view stays pinned open, and the next navigation renders two panels
stacked on one page. Loading #sovereignty and then clicking Fabric & router
showed both. So app.js must remove the attribute as soon as it can switch views
itself, and this test is the thing that notices if it stops.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path("services/kavach/ui")


@pytest.fixture(scope="module")
def app_js() -> str:
    return (UI / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_css() -> str:
    return (UI / "app.css").read_text(encoding="utf-8")


def test_the_anti_flash_attribute_is_removed_once_js_runs(app_js):
    assert 'removeAttribute("data-initial-view")' in app_js, (
        "data-initial-view is never cleared - the view it names will stay "
        "visible on top of every panel the user navigates to next"
    )


def test_it_is_cleared_inside_the_view_switcher(app_js):
    """Clearing it anywhere else can run before the attribute is read, or after
    the user has already seen two panels."""
    body = app_js.split("function show(view)", 1)
    assert len(body) == 2, "show() not found"
    head = body[1][:600]
    assert "removeAttribute" in head, "show() does not clear the attribute"


def test_hide_still_wins_for_every_other_view(app_css):
    """.hide is the mechanism the switcher relies on; if it stops being
    !important the id-selector rules above would outrank it again."""
    assert re.search(r"\.hide\s*\{[^}]*display:\s*none\s*!important", app_css), \
        ".hide must stay !important or view switching silently stops working"


def test_the_initial_view_css_only_targets_known_views(app_css):
    """A typo'd view name here would paint nothing and hide the overview."""
    named = set(re.findall(r'data-initial-view="([^"]+)"', app_css))
    html = (UI / "index.html").read_text(encoding="utf-8")
    real = set(re.findall(r'id="view-([^"]+)"', html))
    assert named <= real, f"CSS names views that do not exist: {named - real}"
