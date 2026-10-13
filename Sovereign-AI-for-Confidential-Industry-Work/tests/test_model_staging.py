"""Adding a model to the box (SS 26117, L1).

The brief asks that new open-weight models be addable later "without
redesigning the system". Three properties make that true rather than merely
possible, and each one has a failure mode that is silent:

  BOTH GATES OR NEITHER. `config/models.yaml` makes a model exist;
  a profile's `available:` list decides whether the router may consider it.
  Writing only the first produces a model that is registered, verified and
  never chosen, with nothing anywhere to say why.

  THE CONFIG KEEPS ITS COMMENTS. A `yaml.safe_dump` round-trip destroys every
  comment in models.yaml, including the one saying the quality numbers are
  unmeasured placeholders. Staging a model must not be the thing that deletes
  the warning about the numbers it just added.

  THE FILE IS READ BEFORE IT IS BELIEVED. A .gguf that is not a GGUF, or is one
  this build's llama.cpp cannot load, is caught at staging rather than by a
  failed load in front of an audience.

And one that is not about models at all: a filename arriving from a browser is
a path, and a path from a caller is a traversal attempt until proved otherwise.
"""

from __future__ import annotations

import struct

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app
from kavach.runtime.staging import (
    SEED_QUALITY,
    StagingError,
    add_to_profile,
    append_model,
    inspect_gguf,
    propose_entry,
    propose_id,
    unregistered_files,
)

# --------------------------------------------------------------- fixtures

def _gguf(path, *, architecture="qwen2", context=32768, embedding=0,
          name="test-model", version=3):
    """A minimal but REAL GGUF header, written the way the format specifies.

    Built by hand rather than shipped as a binary fixture: a checked-in .gguf
    would be a megabyte of opaque test data that nobody can adjust, and the
    thing under test is precisely whether this parser agrees with the layout.
    """
    def string(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def kv_string(key: str, value: str) -> bytes:
        return string(key) + struct.pack("<I", 8) + string(value)

    def kv_u32(key: str, value: int) -> bytes:
        return string(key) + struct.pack("<I", 4) + struct.pack("<I", value)

    pairs = [kv_string("general.architecture", architecture),
             kv_string("general.name", name)]
    if context:
        pairs.append(kv_u32(f"{architecture}.context_length", context))
    if embedding:
        pairs.append(kv_u32(f"{architecture}.embedding_length", embedding))

    body = b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", 12)
    body += struct.pack("<Q", len(pairs)) + b"".join(pairs)
    # Padding so the file has a plausible size; the parser must stop at the
    # end of the header and not care what follows.
    path.write_bytes(body + b"\0" * 4096)
    return path


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


# ------------------------------------------------------------- the header


def test_a_real_header_is_read_without_touching_the_network(tmp_path):
    info = inspect_gguf(_gguf(tmp_path / "qwen2.5-7b-instruct-q4_k_m.gguf"))
    assert info.architecture == "qwen2"
    assert info.context_length == 32768
    assert info.quantisation == "Q4_K_M"
    assert info.warnings == []


def test_a_file_that_is_not_a_gguf_is_refused_with_the_reason(tmp_path):
    """Caught at staging, not by a failed load in front of an audience."""
    path = tmp_path / "notamodel.gguf"
    path.write_bytes(b"PK\x03\x04" + b"\0" * 200)
    with pytest.raises(StagingError, match="not a GGUF"):
        inspect_gguf(path)


def test_an_unknown_architecture_is_reported_and_not_refused(tmp_path):
    """The bge-m3 lesson, made preventable.

    The embedder staged in this repo is tagged `xlmr`, which mainline
    llama.cpp does not implement - every build refuses it with "unknown model
    architecture". That cost a profile change to discover. It is a WARNING and
    not a refusal, because a newer llama.cpp may implement it and a hardcoded
    list in our source deciding what the box may run is the opposite of what
    the brief asks for.
    """
    info = inspect_gguf(_gguf(tmp_path / "bge-m3-q8_0.gguf", architecture="xlmr"))
    assert info.architecture == "xlmr"
    assert any("xlmr" in w for w in info.warnings)


def test_a_truncated_header_gives_back_what_it_could_read(tmp_path):
    """A file whose first four bytes are right is probably a model. Refusing it
    outright would lose the fields that were readable."""
    path = tmp_path / "truncated.gguf"
    # Declares 99 metadata pairs and supplies one that is cut off mid-key.
    # Padded past the 32-byte floor so this exercises a truncated METADATA
    # section rather than the separate "too small to be a model file" check.
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 4)
        + struct.pack("<Q", 99)
        + struct.pack("<Q", 24)
        + b"general.architect"
        + b"\0" * 64
    )
    info = inspect_gguf(path)
    assert info.warnings, "a half-read header reported nothing"


def test_an_implausible_metadata_count_is_refused(tmp_path):
    """A malformed or hostile file must not walk the reader through gigabytes
    of 'metadata' one field at a time."""
    path = tmp_path / "hostile.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
                     + struct.pack("<Q", 1 << 40) + b"\0" * 64)
    with pytest.raises(StagingError, match="plausible"):
        inspect_gguf(path)


# ------------------------------------------------------------- the entry


def test_the_proposed_entry_marks_its_guesses_as_guesses(tmp_path):
    """An estimate that looks like a measurement is the failure this whole
    build is organised against."""
    info = inspect_gguf(_gguf(tmp_path / "some-model-q4_k_m.gguf"))
    entry = propose_entry(info, "some-model")
    assert entry["quality_measured"] is False
    assert set(entry["quality"].values()) == {SEED_QUALITY}
    assert entry["vram_gb"] > 0


def test_an_embedder_is_detected_and_gets_its_dimension(tmp_path):
    info = inspect_gguf(_gguf(tmp_path / "minilm-l6-emb.gguf",
                              architecture="bert", context=512, embedding=384))
    entry = propose_entry(info, "minilm")
    assert entry["role"] == "embedding"
    assert entry["dim"] == 384
    assert "embedding" in entry["capabilities"]


def test_a_generation_model_is_routable_shaped(tmp_path):
    """No `tools` capability means the planner can never use it, which makes it
    a model the agent loop cannot drive."""
    info = inspect_gguf(_gguf(tmp_path / "qwen3-8b-q4.gguf"))
    entry = propose_entry(info, "qwen3-8b")
    assert "tools" in entry["capabilities"]
    assert "text" in entry["capabilities"]


def test_ids_do_not_collide(tmp_path):
    path = _gguf(tmp_path / "model.gguf")
    assert propose_id(path, set()) == "model"
    assert propose_id(path, {"model"}) == "model-2"


# ------------------------------------------------------ writing the config


MANIFEST = """\
# A comment that must survive. The quality numbers below are UNMEASURED.
- id: existing-model
  file: existing.gguf
  sha256: null
  license: apache-2.0
  tier: 2
  capabilities: [text, tools]
  context: 8192
  vram_gb: 4.0
  load_ms: 4000
"""


def test_appending_a_model_keeps_every_comment(tmp_path):
    """The one comment this project cannot afford to lose is the one saying the
    quality numbers are placeholders - and it sits in the file that staging
    writes to."""
    path = tmp_path / "models.yaml"
    path.write_text(MANIFEST, encoding="utf-8")
    append_model(path, {"id": "new-model", "file": "new.gguf", "tier": 2},
                 ["staged through the workbench"])
    text = path.read_text(encoding="utf-8")
    assert "A comment that must survive" in text
    assert "UNMEASURED" in text
    assert "new-model" in text


def test_the_appended_entry_parses_as_yaml(tmp_path):
    import yaml

    path = tmp_path / "models.yaml"
    path.write_text(MANIFEST, encoding="utf-8")
    entry = {"id": "new-model", "file": "new.gguf", "sha256": None,
             "license": "apache-2.0", "tier": 2,
             "capabilities": ["text", "tools"], "context": 4096,
             "vram_gb": 4.2, "load_ms": 3600, "tok_per_s": 20.0,
             "quality": {"code_generation": 0.5, "kb_qa": 0.5},
             "quality_measured": False}
    append_model(path, entry, ["a note", "another note"])
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    written = next(m for m in loaded if m["id"] == "new-model")
    assert written["capabilities"] == ["text", "tools"]
    assert written["quality"]["kb_qa"] == 0.5
    assert written["quality_measured"] is False


def test_a_duplicate_id_is_refused_rather_than_appended_twice(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MANIFEST, encoding="utf-8")
    with pytest.raises(StagingError, match="already"):
        append_model(path, {"id": "existing-model", "file": "x.gguf"}, [])


PROFILES = """\
MID:
  description: "the reference profile"
  vram_gb: 12.0
  available: [qwen3-8b-q4, minilm-l6-emb]

LOW:
  description: "small"
  available:
    - qwen3-1_7b-q4
"""


def test_the_second_gate_is_written_too(tmp_path):
    """THE bug this exists to prevent: a model in models.yaml but not in the
    profile's available list is registered, verified, and never chosen - with
    no error anywhere."""
    path = tmp_path / "profiles.yaml"
    path.write_text(PROFILES, encoding="utf-8")
    assert add_to_profile(path, "MID", "new-model") is True
    text = path.read_text(encoding="utf-8")
    assert "new-model" in text
    assert "qwen3-8b-q4" in text, "the existing entries were lost"


def test_the_block_list_form_is_handled_as_well(tmp_path):
    """profiles.yaml uses both an inline list and a block list. A writer that
    only understands one silently does nothing on the other."""
    import yaml

    path = tmp_path / "profiles.yaml"
    path.write_text(PROFILES, encoding="utf-8")
    add_to_profile(path, "LOW", "new-model")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "new-model" in loaded["LOW"]["available"]
    assert "qwen3-1_7b-q4" in loaded["LOW"]["available"]


def test_adding_a_model_twice_to_a_profile_is_a_no_op(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text(PROFILES, encoding="utf-8")
    assert add_to_profile(path, "MID", "qwen3-8b-q4") is False


def test_only_the_named_profile_is_touched(tmp_path):
    import yaml

    path = tmp_path / "profiles.yaml"
    path.write_text(PROFILES, encoding="utf-8")
    add_to_profile(path, "MID", "new-model")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "new-model" not in loaded["LOW"]["available"]


# ------------------------------------------------------------- the folder


def test_the_scan_ignores_files_already_registered(tmp_path):
    _gguf(tmp_path / "known.gguf")
    _gguf(tmp_path / "brand-new.gguf")
    found = unregistered_files(tmp_path, {"known.gguf"})
    assert [f["file"] for f in found] == ["brand-new.gguf"]


def test_the_scan_survives_a_folder_that_does_not_exist(tmp_path):
    assert unregistered_files(tmp_path / "nope", set()) == []


# ------------------------------------------------------------- the gateway


def test_only_an_administrator_may_stage_a_model(client):
    """The most consequential thing this gateway can be asked to do: it puts a
    file the router may run in front of confidential data."""
    for role in ("engineer", "operator", "auditor"):
        response = client.post("/api/models/stage",
                               json={"file": "anything.gguf", "role": role})
        assert response.status_code == 403, role
        assert "administrator" in response.json()["detail"]


def test_a_refused_attempt_is_on_the_record(client):
    """An auditor asking "did anyone try to add a model?" must get an answer."""
    client.post("/api/models/stage", json={"file": "x.gguf", "role": "operator"})
    rows = client.get("/api/audit/console?n=100").json()["rows"]
    denials = [r for r in rows if r["event"] == "model.stage_denied"]
    assert denials, "a refused staging attempt left no trace"
    assert denials[0]["decision"] == "DENIED"
    assert denials[0]["role"] == "operator"


def test_the_upload_refuses_a_non_admin_before_reading_the_body(client):
    response = client.post(
        "/api/models/upload",
        files={"file": ("evil.gguf", b"GGUF", "application/octet-stream")},
        data={"role": "engineer"},
    )
    assert response.status_code == 403


def test_a_filename_cannot_escape_the_models_folder(client):
    """A filename arriving from a browser is a path, and a path from a caller
    is a traversal attempt until proved otherwise. `../` is not a way to
    register /etc/passwd as a language model."""
    for attempt in ("../../../etc/passwd", "..\\..\\windows\\win.ini",
                    "/etc/shadow"):
        response = client.post("/api/models/stage",
                               json={"file": attempt, "role": "admin"})
        assert response.status_code in (400, 404), attempt
        assert "models folder" in response.json()["detail"]


def test_a_missing_file_says_where_it_should_have_been(client):
    response = client.post("/api/models/stage",
                           json={"file": "not-here.gguf", "role": "admin"})
    assert response.status_code == 404
    assert "Copy it there" in response.json()["detail"]


def test_the_unregistered_listing_is_readable_by_anyone(client):
    """Knowing a file is sitting there unused is not a privileged fact, and
    hiding it means an operator cannot tell an administrator what needs doing."""
    body = client.get("/api/models/unregistered").json()
    assert "model_dir" in body and isinstance(body["files"], list)


def test_a_non_gguf_upload_is_refused_by_extension(client):
    response = client.post(
        "/api/models/upload",
        files={"file": ("model.bin", b"\0" * 16, "application/octet-stream")},
        data={"role": "admin"},
    )
    assert response.status_code == 422
    assert "GGUF" in response.json()["detail"]


# --------------------------------------------- the whole path, end to end


def test_an_administrator_can_add_a_model_and_the_router_can_then_see_it(
    client, tmp_path, monkeypatch
):
    """The claim the brief actually makes: a new open-weight model becomes a
    routing candidate with no redesign, no restart and no code change.

    Driven through the gateway against a real registry and real config files,
    because every interesting failure here is an integration failure - the
    manifest written but the profile not, the registry not reloaded, the id
    colliding. A unit test of each half would pass with the whole broken.
    """
    import shutil

    from kavach.config import get_settings, reset_settings_cache
    from kavach.runtime import reset_registry

    settings = get_settings()

    # Work on copies. These tests must not edit the repository's own config.
    config_dir = tmp_path / "config"
    shutil.copytree(settings.config_dir, config_dir)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setenv("KAVACH_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KAVACH_MODEL_DIR", str(model_dir))
    reset_settings_cache()
    reset_registry()

    try:
        fresh = get_settings()
        _gguf(model_dir / "brand-new-7b-q4_k_m.gguf")

        # 1. it shows up as something waiting to be registered
        found = client.get("/api/models/unregistered").json()
        assert "brand-new-7b-q4_k_m.gguf" in [f["file"] for f in found["files"]]

        # 2. the dry run reads the real header
        preview = client.post("/api/models/inspect", json={
            "file": "brand-new-7b-q4_k_m.gguf", "role": "admin"}).json()
        assert preview["gguf"]["architecture"] == "qwen2"
        assert preview["proposed"]["quality_measured"] is False

        # 3. staging writes both gates and reloads
        response = client.post("/api/models/stage", json={
            "file": "brand-new-7b-q4_k_m.gguf", "role": "admin",
            "license": "apache-2.0"})
        assert response.status_code == 200, response.text
        staged = response.json()
        model_id = staged["id"]
        assert staged["added_to_profile"] == fresh.profile.name

        manifest = (config_dir / "models.yaml").read_text(encoding="utf-8")
        profiles = (config_dir / "profiles.yaml").read_text(encoding="utf-8")
        assert model_id in manifest
        assert model_id in profiles, (
            "the model is in the manifest but not in the profile's available "
            "list - registered, verified, and never chosen"
        )

        # 4. the running registry knows about it, with a real digest
        listed = client.get("/api/models").json()["models"]
        entry = [m for m in listed if m["id"] == model_id]
        assert entry, "the registry did not reload"
        assert staged["sha256"] and len(staged["sha256"]) == 64

        # 5. and the run is on the record
        rows = client.get("/api/audit/console?n=120").json()["rows"]
        staged_rows = [r for r in rows if r["event"] == "model.staged"]
        assert staged_rows and staged_rows[0]["role"] == "admin"
        assert staged_rows[0]["decision"] == "ALLOWED"
    finally:
        monkeypatch.undo()
        reset_settings_cache()
        reset_registry()


def test_staging_the_same_file_twice_is_refused_not_duplicated(
    client, tmp_path, monkeypatch
):
    """A duplicate id in models.yaml is a config the registry cannot resolve."""
    import shutil

    from kavach.config import get_settings, reset_settings_cache
    from kavach.runtime import reset_registry

    settings = get_settings()
    config_dir = tmp_path / "config"
    shutil.copytree(settings.config_dir, config_dir)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setenv("KAVACH_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KAVACH_MODEL_DIR", str(model_dir))
    reset_settings_cache()
    reset_registry()

    try:
        _gguf(model_dir / "twice.gguf")
        first = client.post("/api/models/stage",
                            json={"file": "twice.gguf", "role": "admin"})
        assert first.status_code == 200, first.text
        second = client.post("/api/models/stage",
                             json={"file": "twice.gguf", "role": "admin"})
        assert second.status_code == 409
    finally:
        monkeypatch.undo()
        reset_settings_cache()
        reset_registry()
