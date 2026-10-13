"""The budget may not exceed the card, and a dead gateway may not keep it.

All three properties here come from one afternoon's failure, which presented to
the operator as "why is this not giving me any output":

  * `config/profiles.yaml` said `vram_gb: 12.0  # the real card on this machine`
    and the card was an RTX 3050 6GB Laptop GPU.
  * Seven `llama-server` processes from earlier gateways were still alive,
    holding 5926 MiB of 6144 MiB.
  * The run therefore had 200 MB to load a model into - and it did not fail, it
    STALLED, inside a 600-second request timeout, showing "running" and nothing
    else for ten minutes.

A budget that exceeds the hardware is the reassuring number this codebase exists
to refuse. It does not produce an error; it produces a silence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kavach.config import Profile
from kavach.runtime import gpu
from kavach.runtime.backends import llamacpp
from kavach.runtime.warden import VramWarden

REPO = Path(__file__).resolve().parents[1]


class _Registry:
    """The warden only calls registry.get() on admission; none of these do."""

    def get(self, model_id):                     # pragma: no cover - unused here
        raise KeyError(model_id)


def _profile(**kw) -> Profile:
    base = {
        "name": "TEST", "description": "", "vram_gb": 12.0, "reserve_gb": 2.0,
        "cpu_only": False, "context": 8192, "swap_slots": 2,
        "pinned": [], "available": [],
    }
    base.update(kw)
    return Profile(**base)


def _warden(monkeypatch, card_gb: float | None, **profile_kw) -> VramWarden:
    card = None if card_gb is None else gpu.Gpu("Test Card", card_gb, 0.0)
    monkeypatch.setattr(gpu, "probe", lambda refresh=False: card)
    monkeypatch.setattr("kavach.runtime.warden.probe", lambda refresh=False: card)
    monkeypatch.setattr(
        "kavach.runtime.warden.total_vram_gb", lambda: (card.total_gb if card else None)
    )
    return VramWarden(_profile(**profile_kw), _Registry())


# ------------------------------------------------------------- the budget


def test_the_budget_is_clamped_to_the_card_that_is_actually_here(monkeypatch):
    """The bug. A profile written for a 12 GB card, carried to a 6 GB one,
    handed out 9.8 GB and the loads it admitted could not fit."""
    warden = _warden(monkeypatch, card_gb=6.0, vram_gb=12.0, reserve_gb=2.0)
    assert warden.budget_gb <= 6.0
    assert warden.budget_gb == pytest.approx(6.0 - 2.0 - warden.kv_cache_gb, abs=0.01)


def test_a_profile_that_fits_is_left_exactly_alone(monkeypatch):
    """The clamp is a ceiling, never a target. A profile deliberately budgeting
    less than the card - to leave room for a display, or another process - must
    keep the smaller number."""
    warden = _warden(monkeypatch, card_gb=24.0, vram_gb=12.0, reserve_gb=2.0)
    assert warden.budget_gb == pytest.approx(12.0 - 2.0 - warden.kv_cache_gb, abs=0.01)


def test_an_unreadable_card_does_not_clamp_anything(monkeypatch):
    """None is not zero. An AMD box, a machine with no driver, a CPU-only venue
    laptop - refusing to load anything there because `nvidia-smi` is missing
    would break the contingency profile whose whole job is to keep working."""
    warden = _warden(monkeypatch, card_gb=None, vram_gb=12.0)
    assert warden.budget_gb == pytest.approx(12.0 - 2.0 - warden.kv_cache_gb, abs=0.01)
    assert warden.vram_note is None


def test_a_cpu_only_profile_is_never_clamped_by_a_gpu(monkeypatch):
    """TINY budgets against system RAM by design (deviation 5). A small card in
    the same box must not cap it."""
    warden = _warden(monkeypatch, card_gb=2.0, cpu_only=True, vram_gb=0.0, ram_gb=16.0)
    assert warden.budget_gb > 2.0


def test_the_mismatch_is_said_out_loud_and_names_the_card(monkeypatch):
    """A clamp nobody is told about is a config that stays wrong on every
    machine it is copied to."""
    warden = _warden(monkeypatch, card_gb=6.0, vram_gb=12.0)
    note = warden.vram_note
    assert note and "Test Card" in note
    assert "12.0" in note and "6.0" in note
    assert "profiles.yaml" in note


def test_a_matching_profile_says_nothing(monkeypatch):
    assert _warden(monkeypatch, card_gb=12.0, vram_gb=12.0).vram_note is None


def test_the_panel_is_given_the_card_beside_the_budget(monkeypatch):
    """"6.2 / 9.8 GB" on a 6 GB card is not a rounding error, and a reader
    cannot catch it without the card's own number next to it."""
    state = _warden(monkeypatch, card_gb=6.0, vram_gb=12.0).state()
    assert state["gpu"]["total_gb"] == 6.0
    assert state["vram_note"]


# --------------------------------------------------------- the real config


def test_the_two_profiles_for_this_box_agree_about_its_card():
    """MIDRANGE (then called DEMO) said 12.0 with the comment "the real card on
    this machine", and the card is 6 GB. The comment was the thing that made it
    believable. This test cannot check the hardware of every future box - what
    it CAN do is stop the two profiles written for THIS one from disagreeing."""
    profiles = yaml.safe_load((REPO / "config" / "profiles.yaml").read_text(encoding="utf-8"))
    assert profiles["MIDRANGE"]["vram_gb"] == profiles["FAST"]["vram_gb"], (
        "FAST and MIDRANGE are both written for the card in this machine, so "
        "they cannot disagree about how big it is"
    )


def test_no_profile_is_named_after_a_demonstration():
    """"profile DEMO" in the topbar reads as "this is demo data", which is the
    one claim this product must never make by accident - every number on the
    page is measured on this box. A profile name describes HARDWARE."""
    profiles = yaml.safe_load((REPO / "config" / "profiles.yaml").read_text(encoding="utf-8"))
    for name in profiles:
        assert "DEMO" not in name.upper(), f"profile {name} is named after a demo"


def test_no_profile_lists_a_model_that_cannot_fit_in_its_own_budget():
    """A profile naming weights that do not FIT is worse than one naming weights
    that are absent: absent produces "no viable candidate" at plan time, too big
    produces a stall at load time."""
    profiles = yaml.safe_load((REPO / "config" / "profiles.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((REPO / "config" / "models.yaml").read_text(encoding="utf-8"))
    sizes = {m["id"]: float(m.get("vram_gb", 0.0)) for m in models}

    for name, profile in profiles.items():
        if profile.get("cpu_only"):
            continue
        budget = float(profile["vram_gb"]) - float(profile.get("reserve_gb", 0.0))
        for model_id in profile.get("available", []):
            size = sizes.get(model_id)
            if size is None:
                continue                  # models.yaml is checked elsewhere
            assert size <= budget, (
                f"profile {name} lists {model_id} ({size} GB) but its budget after "
                f"reserve is {budget:.1f} GB - it can never be admitted"
            )


def test_the_planner_and_the_writer_agree_on_which_formats_exist():
    """The planner refuses a .pdf step so a run does not spend minutes reaching
    a deliverable the box cannot write. Two lists of formats is one list that
    will drift."""
    from kavach.agent.planner import DOC_FORMATS
    from kavach.tools.impl.deliverables import SUPPORTED

    assert DOC_FORMATS == SUPPORTED


# ------------------------------------------------------------ orphan reaping


def test_a_server_that_no_longer_answers_is_not_killed_by_pid(monkeypatch, tmp_path):
    """A pid can be recycled. Killing on the strength of a stale record alone
    would eventually kill something that merely inherited the number."""
    record = tmp_path / llamacpp.ORPHAN_FILE
    record.write_text(json.dumps([{"port": 18000, "pid": 424242, "model": "x"}]))
    monkeypatch.setattr(llamacpp, "_orphan_path", lambda: record)
    monkeypatch.setattr(llamacpp, "_is_llama_server", lambda port: False)

    killed = []
    monkeypatch.setattr(llamacpp.subprocess, "run", lambda *a, **k: killed.append(a))
    assert llamacpp.reap_orphans() == []
    assert killed == []


def test_a_live_orphan_is_killed_and_the_record_cleared(monkeypatch, tmp_path):
    record = tmp_path / llamacpp.ORPHAN_FILE
    record.write_text(json.dumps([{"port": 18000, "pid": 4242, "model": "qwen"}]))
    monkeypatch.setattr(llamacpp, "_orphan_path", lambda: record)
    monkeypatch.setattr(llamacpp, "_is_llama_server", lambda port: True)
    monkeypatch.setattr(llamacpp.os, "name", "nt")

    seen = []
    monkeypatch.setattr(
        llamacpp.subprocess, "run",
        lambda cmd, **k: seen.append(cmd) or _Done(),
    )
    killed = llamacpp.reap_orphans()
    assert [r["pid"] for r in killed] == [4242]
    assert any("4242" in str(c) for c in seen)
    assert json.loads(record.read_text()) == [], "the record must not survive the reap"


def test_a_server_is_recorded_before_it_is_waited_for(monkeypatch, tmp_path):
    """The one that leaks is the server still LOADING when the gateway is
    killed: it holds VRAM, it is not in `self._servers` yet, and nothing on the
    next boot knows it exists."""
    record = tmp_path / llamacpp.ORPHAN_FILE
    monkeypatch.setattr(llamacpp, "_orphan_path", lambda: record)
    llamacpp._remember(18007, 999, "qwen3-8b-q4")
    saved = json.loads(record.read_text())
    assert saved[0]["port"] == 18007 and saved[0]["pid"] == 999


def test_a_cleanly_killed_server_leaves_no_record(monkeypatch, tmp_path):
    record = tmp_path / llamacpp.ORPHAN_FILE
    monkeypatch.setattr(llamacpp, "_orphan_path", lambda: record)
    llamacpp._remember(18007, 999, "qwen3-8b-q4")
    llamacpp._forget(18007)
    assert json.loads(record.read_text()) == []


def test_an_unwritable_var_directory_cannot_stop_a_model_loading(monkeypatch):
    """Bookkeeping is never allowed to be the thing that fails a run. The cost
    of failing here is a leak on the next hard kill, not a broken workbench."""
    monkeypatch.setattr(llamacpp, "_orphan_path", lambda: Path("//?/nope/x.json"))
    llamacpp._remember(18000, 1, "m")       # must not raise
    assert llamacpp._load_records() == []


class _Done:
    returncode = 0
    stdout = ""
    stderr = ""
