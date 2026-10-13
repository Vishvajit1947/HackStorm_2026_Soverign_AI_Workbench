"""Test isolation.

Kavach holds process-global singletons for the database, audit chain and sentinel -
deliberately, because there is exactly one audit chain per box. Tests therefore have
to redirect KAVACH_VAR_DIR *before* anything imports those modules, or they append
to your real chain and the run-sheet record count is wrong on demo day.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="kavach-tests-"))

os.environ["KAVACH_VAR_DIR"] = str(_TMP / "var")
os.environ["KAVACH_MODEL_DIR"] = str(_TMP / "models")
os.environ["KAVACH_BACKEND"] = "mock"
os.environ["KAVACH_ENFORCE_DIGESTS"] = "0"
os.environ.setdefault("KAVACH_PROFILE", "MID")

# Imported here rather than at the top: every line above must run before
# anything imports kavach, and moving this up would tempt a later edit to
# move the env vars with it.
import pytest


@pytest.fixture(autouse=True)
def _no_card_under_test(monkeypatch):
    """No test may depend on the GPU in the developer's machine.

    The warden clamps its budget to the card that is actually present, which is
    right on a real box and wrong here: the MID profile the suite runs against
    declares 12 GB, and on a 6 GB laptop every warden test would fail with
    VramExhausted - a red suite caused by the hardware rather than by the code,
    and a green one on the reviewer's bigger machine.

    `None` is the value the warden already understands as "the card cannot be
    read", and it makes the profile authoritative exactly as it was before the
    clamp existed. Tests that are ABOUT the clamp patch a card in themselves -
    see tests/test_vram_reality.py.
    """
    monkeypatch.setattr("kavach.runtime.gpu.probe", lambda refresh=False: None)
    monkeypatch.setattr("kavach.runtime.warden.probe", lambda refresh=False: None)
    monkeypatch.setattr("kavach.runtime.warden.total_vram_gb", lambda: None)
