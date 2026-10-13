"""The shipped hardware profiles must actually hold what they claim.

These test config/profiles.yaml against config/models.yaml, not synthetic fixtures.
A profile whose numbers do not add up fails silently at runtime - it just thrashes
the card - so the arithmetic is asserted here.

The HIGH profile regressed exactly this way once: a 2.2 GB fragmentation reserve
plus a separately-charged KV cache double-counted, the second swap slot was
squeezed out, and the vision model went back to evicting the text model. Nothing
errored; the profile just quietly stopped doing its job.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.config import Profile, get_settings
from kavach.runtime.backends.mock import MockBackend
from kavach.runtime.registry import ModelRegistry
from kavach.runtime.warden import VramWarden


@pytest.fixture(scope="module")
def profiles():
    settings = get_settings()
    with (settings.config_dir / "profiles.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def registry():
    settings = get_settings()
    return ModelRegistry(
        settings.config_dir / "models.yaml",
        settings.model_dir,
        enforce_digests=False,
        simulate=True,
    )


def build(profiles, registry, name: str) -> VramWarden:
    raw = profiles[name]
    profile = Profile(
        name=name,
        description=raw.get("description", ""),
        vram_gb=float(raw["vram_gb"]),
        reserve_gb=float(raw.get("reserve_gb", 0.0)),
        cpu_only=bool(raw.get("cpu_only", False)),
        context=int(raw.get("context", 8192)),
        pinned=list(raw.get("pinned", [])),
        available=list(raw.get("available", [])),
        swap_slots=int(raw.get("swap_slots", 1)),
        ram_gb=float(raw.get("ram_gb", 0.0)),
    )
    warden = VramWarden(profile, registry, backend=MockBackend(time_scale=0.0))
    warden.pin_profile_models()
    return warden


ALL_PROFILES = ["TINY", "LOW", "MID", "HIGH", "ARC"]


# ---------------------------------------------------------------- sanity

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_every_available_model_exists_in_the_registry(profiles, registry, name):
    missing = [m for m in profiles[name]["available"] if registry.get(m) is None]
    assert not missing, f"{name} lists models not in models.yaml: {missing}"


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_pinned_models_are_also_available(profiles, name):
    raw = profiles[name]
    orphans = [m for m in raw.get("pinned", []) if m not in raw["available"]]
    assert not orphans, f"{name} pins models it does not list as available: {orphans}"


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_pinning_succeeds(profiles, registry, name):
    warden = build(profiles, registry, name)
    assert warden.pin_failures == [], f"{name}: {warden.pin_failures}"


@pytest.mark.parametrize("name", ["LOW", "MID", "HIGH"])
def test_every_available_model_can_be_admitted(profiles, registry, name):
    """No profile may list a model that can never fit on its own card."""
    warden = build(profiles, registry, name)
    too_big = [
        m for m in profiles[name]["available"]
        if registry.get(m).role == "generation" and not warden.fits(m)
    ]
    assert not too_big, f"{name} lists models that cannot fit: {too_big}"


# ---------------------------------------------------------------- the claims

def test_mid_has_one_slot_so_vision_evicts_text(profiles, registry):
    """MID is the reference profile and is genuinely swap-bound. The router's
    swap penalty exists because of this."""
    warden = build(profiles, registry, "MID")
    warden.admit("qwen3-8b-q4")
    decision = warden.admit("qwen3-vl-8b-q4")
    assert "qwen3-8b-q4" in decision.evicted


def test_high_holds_text_and_vision_at_once(profiles, registry):
    """The whole point of HIGH: a multimodal step must not evict the text model.

    If this fails, check reserve_gb against the separately-charged KV cache before
    assuming the slot logic is wrong.
    """
    warden = build(profiles, registry, "HIGH")
    warden.admit("qwen3-8b-q4")
    decision = warden.admit("qwen3-vl-8b-q4")

    assert decision.evicted == [], (
        f"HIGH evicted {decision.evicted} - the second swap slot is not usable. "
        f"budget={warden.budget_gb:.2f} GB, kv={warden.kv_cache_gb:.2f} GB"
    )
    assert warden.is_resident("qwen3-8b-q4")
    assert warden.is_resident("qwen3-vl-8b-q4")


def test_high_revisits_are_free(profiles, registry):
    warden = build(profiles, registry, "HIGH")
    warden.admit("qwen3-8b-q4")
    warden.admit("qwen3-vl-8b-q4")
    assert warden.admit("qwen3-8b-q4").was_resident
    assert warden.swap_cost("qwen3-vl-8b-q4") == 0.0


@pytest.mark.parametrize("name", ["LOW", "MID", "HIGH"])
def test_profiles_keep_real_headroom(profiles, registry, name):
    """Fig 2's claim: every profile keeps genuine headroom for fragmentation and
    display output once its heaviest realistic working set is resident."""
    warden = build(profiles, registry, name)
    generation = [
        m for m in profiles[name]["available"]
        if registry.get(m).role == "generation"
    ]
    # Load the largest models that fit, up to the slot count.
    for model_id in sorted(generation, key=lambda m: -registry.get(m).vram_gb):
        if len(warden.state()["resident"]) - len(profiles[name].get("pinned", [])) >= \
                warden.swap_slots:
            break
        if warden.free_gb >= registry.get(model_id).vram_gb:
            warden.admit(model_id)

    total_committed = warden.used_gb + warden.kv_cache_gb
    headroom = profiles[name]["vram_gb"] - total_committed
    assert headroom >= 1.0, (
        f"{name} leaves only {headroom:.2f} GB free with "
        f"{[r['model_id'] for r in warden.state()['resident']]} resident"
    )


def test_tiny_profile_is_cpu_only(profiles):
    """The contingency profile must never claim VRAM it does not have."""
    assert profiles["TINY"]["cpu_only"] is True
    assert profiles["TINY"]["vram_gb"] == 0.0


def test_tiny_profile_charges_no_kv_cache(profiles, registry):
    warden = build(profiles, registry, "TINY")
    assert warden.kv_cache_gb == 0.0


def test_tiny_profile_can_actually_load_models(profiles, registry):
    """The contingency has to work. A CPU-only profile budgets against system RAM;
    if it budgeted against vram_gb (0.0) nothing could ever be pinned or admitted,
    and the fallback would be dead exactly when it is needed."""
    warden = build(profiles, registry, "TINY")
    assert warden.budget_gb > 0
    assert warden.is_resident("minilm-l6-emb")
    decision = warden.admit("qwen3-1_7b-q4")
    assert decision.admitted


# ------------------------------------------------ machine settings from a file


def test_a_real_environment_variable_beats_the_env_file(tmp_path, monkeypatch):
    """`kavach.env` describes the machine; the environment describes this run.

    If the file won, a developer's local `KAVACH_BACKEND=llamacpp` would follow
    every test into CI and every one-off `KAVACH_PROFILE=LOW ...` would be
    silently ignored. conftest.py sets the backend to `mock` before anything
    imports, and that has to keep holding with a file present.
    """
    import os

    from kavach.config import _load_env_file

    # KAVACH_LLAMA_BIN_DIR for the second half rather than KAVACH_PROFILE:
    # conftest.py already sets the profile for the whole test process, so it is
    # not an unset key here and would prove nothing.
    monkeypatch.delenv("KAVACH_LLAMA_BIN_DIR", raising=False)
    (tmp_path / "kavach.env").write_text(
        "KAVACH_BACKEND=llamacpp\nKAVACH_LLAMA_BIN_DIR=D:/from-the-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAVACH_BACKEND", "mock")
    _load_env_file(tmp_path)

    assert os.environ["KAVACH_BACKEND"] == "mock", (
        "the file overrode a real environment variable"
    )
    assert os.environ["KAVACH_LLAMA_BIN_DIR"] == "D:/from-the-file", (
        "an unset key was not filled in from the file"
    )


def test_the_env_file_cannot_set_anything_outside_kavach(tmp_path, monkeypatch):
    """A stray line in a machine settings file must never reach PATH,
    PYTHONPATH or anything else the process depends on."""
    import os

    from kavach.config import _load_env_file

    monkeypatch.delenv("KAVACH_SAFE_MARKER", raising=False)
    before = os.environ.get("PATH")
    (tmp_path / "kavach.env").write_text(
        "PATH=/nowhere\nPYTHONPATH=/evil\nKAVACH_SAFE_MARKER=ok\n", encoding="utf-8")
    _load_env_file(tmp_path)

    assert os.environ.get("PATH") == before
    assert os.environ.get("PYTHONPATH") != "/evil"
    assert os.environ["KAVACH_SAFE_MARKER"] == "ok"


def test_a_missing_env_file_is_not_an_error(tmp_path):
    """Every machine that has not been configured yet is this case."""
    from kavach.config import _load_env_file

    _load_env_file(tmp_path / "nothing-here")
