"""The two themes, as a contract (L6, §09).

The UI has no build step and no visual regression harness, so the properties
that make a theme usable rather than merely coloured are pinned here. Every one
of these has a failure mode that is invisible in the browser the author happens
to be using and obvious on the venue projector.

The rules this file enforces:

  BOTH THEMES ARE DEFINED. A token that exists only on `:root` and is never
  restated for `[data-theme="light"]` is a dark-mode value bleeding into the
  light theme - the exact "just inverted" result this was meant to avoid.

  NOTHING IS FETCHED. `font-src 'self'` plus an air gap means no @import, no
  @font-face pointing anywhere, and no absolute URL in the stylesheet. Naming
  Space Grotesk in a font stack is fine; asking a CDN for it is not.

  STATE IS NEVER COLOUR ALONE. The audit console's verdicts carry a glyph and
  the word itself.

  CONTROLS ARE REACHABLE. One control height token, focus-visible rings, and
  no interactive element that can be tabbed to invisibly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "services" / "kavach" / "ui"


@pytest.fixture(scope="module")
def css() -> str:
    """The stylesheet with its comments stripped.

    Tested against the RULES, not the prose. A comment explaining why a font is
    not fetched would otherwise fail the test that checks no font is fetched -
    which teaches the next person to delete the explanation rather than keep
    the property.
    """
    raw = (UI / "app.css").read_text(encoding="utf-8")
    return re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)


@pytest.fixture(scope="module")
def js() -> str:
    return (UI / "app.js").read_text(encoding="utf-8")


def _block(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css, re.DOTALL)
    assert match, f"{selector} block not found in app.css"
    return match.group(1)


def _tokens(block: str) -> set[str]:
    return set(re.findall(r"(--[a-z0-9-]+)\s*:", block))


def _rule(css: str, selector: str) -> str:
    """The body of the rule whose selector IS this, and not one ending in it.

    `_block(css, ".chip")` finds `.topbar > .chip {...}` first and reads the
    layout rule instead of the component - so a test written with it passes or
    fails on a rule it was never about.
    """
    match = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{(.*?)\}", css, re.DOTALL)
    assert match, f"no top-level rule for {selector} in app.css"
    return match.group(1)


# --------------------------------------------------------- two real themes


def test_the_light_theme_redefines_every_colour_the_dark_theme_sets(css):
    """The "just inverted" guard.

    A colour token defined once on :root is the DARK value, and the light theme
    inherits it - so one forgotten line puts #FAFAFA text on white paper. Only
    the non-colour tokens (radii, fonts, control height) are legitimately
    shared, and they are listed explicitly rather than pattern-matched, so a
    new colour cannot quietly join the exempt set.
    """
    dark = _tokens(_block(css, ":root"))
    light = _tokens(_block(css, ':root[data-theme="light"]'))

    shared = {
        "--mono", "--sans", "--display",
        "--r-sm", "--r", "--r-lg", "--r-pill", "--control-h",
    }
    missing = sorted(dark - light - shared)
    assert missing == [], (
        f"these tokens are only defined for dark, so the light theme inherits "
        f"a dark value: {missing}"
    )


def test_the_two_themes_do_not_share_a_single_surface_colour(css):
    """Light is not dark with the numbers flipped: the surfaces, the ink and
    the shadows are all chosen for their own ground."""
    dark = _block(css, ":root")
    light = _block(css, ':root[data-theme="light"]')
    for token in ("--bg-0", "--bg-1", "--ink", "--line", "--shadow"):
        dark_value = re.search(rf"{token}\s*:\s*([^;]+);", dark).group(1).strip()
        light_value = re.search(rf"{token}\s*:\s*([^;]+);", light).group(1).strip()
        assert dark_value != light_value, (
            f"{token} is identical in both themes ({dark_value}) - one of them "
            "was not designed for its own background"
        )


def test_dark_is_layered_slate_and_not_pure_black(css):
    """Depth here comes from stacking surfaces, not from shadows. Three
    distinct tones have to be visible at once, and #000 would flatten the
    glass cards into nothing."""
    dark = _block(css, ":root")
    surfaces = [
        re.search(rf"--bg-{n}\s*:\s*([^;]+);", dark).group(1).strip().lower()
        for n in range(5)
    ]
    assert len(set(surfaces)) == 5, f"surfaces are not distinct: {surfaces}"
    assert "#000" not in surfaces and "#000000" not in surfaces


def test_the_accent_separates_its_fill_from_its_text_colour(css):
    """Amber as a BACKGROUND and amber as TEXT are different colours: amber-500
    on white is about 2:1 and unreadable, which is why --accent-text exists and
    differs from --accent in the light theme."""
    light = _block(css, ':root[data-theme="light"]')
    fill = re.search(r"--accent\s*:\s*([^;]+);", light).group(1).strip()
    text = re.search(r"--accent-text\s*:\s*([^;]+);", light).group(1).strip()
    ink = re.search(r"--accent-ink\s*:\s*([^;]+);", light).group(1).strip()
    assert fill != text, "amber text on paper would reuse the button fill"
    assert ink != fill, "the text ON the accent cannot be the accent"


# ----------------------------------------------------------- still offline


def test_the_stylesheet_fetches_nothing(css):
    """The CSP on the page would block it, the air gap would fail it, and
    offline_check.py fails the build over it. Three layers, and this is the one
    that names the reason."""
    assert "@import" not in css
    assert "@font-face" not in css
    for pattern in ("http://", "https://", "//fonts."):
        assert pattern not in css, f"app.css references {pattern}"


def test_the_named_fonts_are_a_stack_with_a_local_fallback(css):
    """Space Grotesk and Inter may be installed on the machine; they are never
    downloaded. Naming one without a fallback would render the UI in whatever
    the browser defaults to on a box that lacks it."""
    for token in ("--display", "--sans", "--mono"):
        value = re.search(rf"{token}\s*:\s*([^;]+);", css).group(1)
        assert "system-ui" in value or "monospace" in value, (
            f"{token} names fonts with no system fallback: {value}"
        )


# ------------------------------------------------------- accessible states


def test_every_interactive_element_has_a_focus_ring(css):
    """A control that can be tabbed to and shows nothing is unusable by
    keyboard and invisible in a screen recording of the demo."""
    for selector in ("button:focus-visible", ".nav-item:focus-visible",
                     ".segmented .seg:focus-visible", ".console-row:focus-visible",
                     ".suggest-chip:focus-visible"):
        assert selector in css, f"{selector} has no focus treatment"


def test_focus_rings_use_the_accent_so_they_are_one_recognisable_thing(css):
    rings = re.findall(r":focus-visible\s*\{([^}]*)\}", css)
    assert rings, "no focus-visible rules at all"
    accented = [r for r in rings if "--accent" in r or "--red" in r]
    assert len(accented) >= 5, (
        "focus rings are not consistently the accent colour; a focus indicator "
        "that changes colour per component reads as four different states"
    )


def test_controls_share_one_comfortable_target_height(css):
    """44px, set once. A page where each control picked its own height is one
    where half of them are too small to hit on a touchscreen."""
    height = re.search(r"--control-h\s*:\s*(\d+)px", css).group(1)
    assert int(height) >= 44, f"--control-h is {height}px, below a 44px target"
    assert "height: var(--control-h);" in css


# ------------------------------------------- a label does not look pressable


def test_a_static_chip_is_text_and_not_a_box(css):
    """`span.chip` and `button.chip` used to render identically.

    The role note - "reads everything, incl. RESTRICTED" - is read-only, and it
    carried the same border, fill and radius as the theme toggle beside it. A
    reader who presses a label and gets nothing learns that this page's buttons
    sometimes do nothing, and the doubt then applies to the real ones. Hover was
    the only thing separating them, and hover does not exist on a touchscreen.
    """
    block = _rule(css, ".chip")
    assert re.search(r"border:\s*0", block), ".chip draws a border, so a label looks pressable"
    assert re.search(r"background:\s*none", block), ".chip is filled, so a label looks pressable"


def test_a_chip_that_is_a_button_still_looks_like_one(css):
    """The other half of the same rule, and the half that breaks silently: the
    static rule has nothing left to inherit, so the button form has to state
    its own edge, fill and pointer or the control disappears entirely."""
    block = _rule(css, "button.chip")
    assert "border:" in block and "var(--line)" in block
    assert "background:" in block
    assert "cursor: pointer" in block


@pytest.mark.parametrize("selector", [".badge", ".role-tag", ".layer-tag"])
def test_nothing_that_only_reports_a_fact_carries_a_control_edge(css, selector):
    """Verdicts, the role a record was written under, the layer a page belongs
    to. None of them does anything when clicked. A coloured badge may set a
    border-colour, because there the colour IS the meaning - what none of them
    may do is wear the neutral line every control on this page uses."""
    block = _rule(css, selector)
    assert "var(--line)" not in block, (
        f"{selector} is not clickable and wears a control's border"
    )


def test_no_element_styled_as_a_static_chip_is_given_a_click_handler(js):
    """The rule has to hold in the other direction too. A chip that is made
    clickable must become a `button`, which is what carries the pressable
    treatment - wiring a handler onto a `span.chip` gives the page a control
    with no affordance at all, which is worse than the bug this fixed."""
    static = set(re.findall(r'chip\("([\w-]+)"', js))
    static.update({"role-note", "ask-as"})
    for element_id in static:
        assert f'$("{element_id}").addEventListener("click"' not in js, (
            f"{element_id} is a static chip with a click handler; make it a button"
        )


def test_the_light_theme_strengthens_edges_only_on_things_you_can_press(css):
    """The light theme thickens control borders, because a 9%-opacity line is
    invisible on white paper. Listing a label there puts the box back on it in
    one theme only - which is how the two themes drift apart."""
    edged: set[str] = set()
    for selectors, body in re.findall(r"([^{}]*)\{([^{}]*)\}", css):
        if "--line-strong" in body and "border-color" in body:
            edged.update(s.strip() for s in selectors.split(","))
    assert "button.chip" in edged, "the control-edge rule has been renamed or removed"
    for label in (".chip", ".role-tag", ".layer-tag", ".badge"):
        assert label not in edged, f"{label} is a label and the light theme edges it"


def test_the_verdict_badge_carries_a_glyph_and_not_only_a_colour(css, js):
    """The single most important accessibility property in the audit console:
    ALLOWED and DENIED must be distinguishable in greyscale."""
    assert ".verdict-glyph" in css, "the verdict badge has no glyph element"
    marks = re.search(r"const AUDIT_MARK = \{(.*?)\n\};", js, re.DOTALL)
    assert marks, "AUDIT_MARK has been renamed or removed"
    entries = re.findall(r"(\w+):\s*\[\"(\w*)\",\s*\"(.+?)\"\]", marks.group(1))
    assert entries, "AUDIT_MARK is empty"
    for name, tone, glyph in entries:
        assert glyph.strip(), f"{name} has a colour but no glyph"


def test_allowed_and_denied_do_not_share_a_glyph(js):
    marks = re.search(r"const AUDIT_MARK = \{(.*?)\n\};", js, re.DOTALL).group(1)
    glyphs = {
        m[0]: m[2]
        for m in re.findall(r"(\w+):\s*\[\"(\w*)\",\s*\"(.+?)\"\]", marks)
    }
    assert glyphs["ALLOWED"] != glyphs["DENIED"]
    assert glyphs["OK"] != glyphs["FAILED"]
    assert glyphs["BLOCKED"] != glyphs["ALLOWED"]


def test_the_console_rows_are_reachable_by_keyboard(js):
    """The detail opens on click. If that is the only way in, every audit
    record's payload is mouse-only."""
    body = js.split("function renderAuditConsole(", 1)[1]
    assert "tabIndex" in body, "console rows cannot be focused"
    assert "keydown" in body, "console rows respond to click but not to Enter"


# ------------------------------------------------------- one palette, once


def test_no_status_colour_is_written_down_outside_the_token_blocks(css):
    """The second-palette guard.

    Every status tint used to be an `rgba()` literal spelled out at each of the
    twenty-odd places a status is drawn. That is a second palette, and a second
    palette drifts: re-theming the page left emerald-tinted "good" badges with
    MongoDB-green text in them, exactly the way the charts once kept a hardcoded
    blue after the design system had moved on.

    Neutrals are allowed - a black modal backdrop and white overlays are not
    palette decisions. Anything with a hue has to come from a token.
    """
    body = css
    for block in (_rule(css, ":root"), _rule(css, ':root[data-theme="light"]')):
        body = body.replace(block, "")

    coloured = []
    for match in re.finditer(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", body):
        r, g, b = (int(match.group(i)) for i in (1, 2, 3))
        if not r == g == b:
            coloured.append(match.group(0))
    assert coloured == [], (
        f"these colours are written outside the token blocks, so they are a "
        f"second palette that will drift from the first: {coloured}"
    )


@pytest.mark.parametrize(("selector", "family"), [
    (".verdict.good", "green"),
    (".verdict.warn", "amber"),
    (".verdict.bad", "red"),
    (".verdict.blue", "blue"),
])
def test_each_verdict_is_drawn_in_its_own_status_colour(css, family, selector):
    """APPROVAL_REQUIRED was drawn with the ACCENT tokens, which looked correct
    only for as long as the accent happened to be amber. The moment the accent
    became green, every "a human must decide this" verdict rendered in the one
    colour that means the opposite."""
    rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert rule, f"{selector} was renamed or removed"
    body = rule.group(1)
    assert f"--{family}" in body, f"{selector} does not use the {family} status colour"
    assert "--accent" not in body, (
        f"{selector} is drawn with the accent; a status must not change meaning "
        "when the accent colour changes"
    )


# --------------------------------------------------------- restraint checks


def _rgb(hex_colour: str) -> str:
    value = hex_colour.strip().lstrip("#")
    return ", ".join(str(int(value[i:i + 2], 16)) for i in (0, 2, 4))


def test_the_glow_is_reserved_for_the_accent(css):
    """Ambient glow is the signature effect and stops being one the moment it
    is on everything. It belongs to the accent and to nothing else.

    Derived from `--accent` rather than asserted against a literal: this test
    used to name the amber rgb triple itself, which made it a SECOND place the
    accent colour was written down. Re-theming the page then failed a test that
    was about restraint, for a reason that had nothing to do with restraint.
    """
    accent = re.search(r"--accent\s*:\s*#([0-9A-Fa-f]{6})", _rule(css, ":root")).group(1)
    glow = re.search(r"--glow\s*:\s*([^;]+);", css).group(1)
    assert _rgb(accent) in glow, (
        f"the glow is not the accent colour: {glow.strip()} is not rgba({_rgb(accent)}, ...)"
    )


def test_cards_do_not_move_under_the_cursor(css):
    """These are data panels being read. A table that lifts when the pointer
    crosses it is a table whose row you then have to find again."""
    card_hover = re.search(r"\.card:hover\s*\{([^}]*)\}", css)
    assert card_hover, ".card:hover was removed"
    assert "transform" not in card_hover.group(1), (
        "cards transform on hover, which moves the data under the reader"
    )


def test_the_glass_card_degrades_to_a_solid_one(css):
    """color-mix is recent. A browser without it must not render a transparent
    card over the page wash - which is unreadable, not merely unstyled."""
    assert "@supports not (background: color-mix" in css
