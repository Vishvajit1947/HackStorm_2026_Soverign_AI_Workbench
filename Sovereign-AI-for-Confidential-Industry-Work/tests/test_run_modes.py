"""The Code / Document toggles (SS 26117, L5/L6).

The brief asks for a workbench where a coding request is handled differently
from a document request, and where a person can switch between the two results.
That is two claims, and this file exists because either one can be faked:

  A toggle that only changes a label is a lie about the routing. So the tests
  below check the RouteRequest the planner actually issued and the system
  prompt it actually sent - not that the string "code" survived to the API.

  A split that only groups filenames is a lie about the deliverable. So the
  tests check that the partition is total (nothing vanishes), that a preview
  read back from a file is distinguishable from the text a run merely ASKED to
  write, and that an empty side says why it is empty.

The mode also has to survive the one thing a demo box does between rehearsals:
being restarted, with a database written by the previous build.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from kavach.agent.outputs import split_outputs
from kavach.agent.planner import MODE_DIRECTIVE, Planner
from kavach.store.db import Database
from kavach.types import Run, TaskClass, WorkMode

# --------------------------------------------------------------- the mapping


def test_auto_routes_exactly_as_the_planner_did_before_the_toggle_existed():
    """AUTO must be a no-op, not a fourth behaviour.

    The planner hardcoded AGENTIC_MULTI. If adding the toggle changed what an
    untoggled run does, every routing figure taken before this change would
    silently stop describing the system.
    """
    assert WorkMode.AUTO.forced_class is TaskClass.AGENTIC_MULTI


def test_each_mode_pins_a_different_class():
    """Two modes that resolve to the same class cannot change the model chosen,
    which is the entire demonstrable claim."""
    assert WorkMode.CODE.forced_class is TaskClass.CODE_GENERATION
    assert WorkMode.DOCUMENT.forced_class is TaskClass.OFFICIAL_DRAFTING
    assert WorkMode.CODE.forced_class is not WorkMode.DOCUMENT.forced_class


def test_every_mode_has_a_class_so_none_can_raise_at_plan_time():
    for mode in WorkMode:
        assert isinstance(mode.forced_class, TaskClass)


# --------------------------------------------------------------- the planner


class _RecordingRouter:
    """Captures the request and system prompt, and returns a valid one-step plan."""

    def __init__(self):
        self.requests = []
        self.systems = []

    def route(self, request, **kwargs):
        self.requests.append(request)
        self.systems.append(kwargs.get("system", ""))
        plan = [{
            "intent": "write it down",
            "tool": "fs.write",
            "args": {"path": "note.txt", "content": "x"},
            "acceptance": "note.txt exists in the workspace",
        }]
        return SimpleNamespace(
            text=json.dumps(plan), result=SimpleNamespace(model_id="fake")
        )


@pytest.mark.parametrize(
    "mode,expected",
    [
        (WorkMode.AUTO, TaskClass.AGENTIC_MULTI),
        (WorkMode.CODE, TaskClass.CODE_GENERATION),
        (WorkMode.DOCUMENT, TaskClass.OFFICIAL_DRAFTING),
    ],
)
def test_the_mode_reaches_the_router_as_a_forced_class(mode, expected):
    """The toggle has to land on the RouteRequest. A mode that is recorded on
    the run and never routed with is decoration."""
    router = _RecordingRouter()
    Planner(router=router).plan("do the thing", mode=mode)
    assert router.requests[0].forced_class is expected


def test_the_mode_directive_reaches_the_model_and_only_its_own():
    """A prompt carrying both directives tells the model to deliver a program
    AND a Word file, which is how a toggle stops changing the output."""
    router = _RecordingRouter()
    Planner(router=router).plan("do the thing", mode=WorkMode.CODE)
    system = router.systems[0]
    assert "CODE MODE" in system
    assert "DOCUMENT MODE" not in system

    router = _RecordingRouter()
    Planner(router=router).plan("do the thing", mode=WorkMode.DOCUMENT)
    system = router.systems[0]
    assert "DOCUMENT MODE" in system
    assert "CODE MODE" not in system


def test_auto_adds_nothing_to_the_prompt():
    """Same reasoning as the routing no-op: an untoggled run must be the run
    this build has always planned."""
    router = _RecordingRouter()
    Planner(router=router).plan("do the thing", mode=WorkMode.AUTO)
    assert MODE_DIRECTIVE[WorkMode.AUTO] == ""
    assert "MODE." not in router.systems[0]


def test_each_directive_names_the_tool_that_can_actually_deliver_it():
    """Telling a small model "write code" without naming the tool produces a
    prose description in a .txt file. The directive has to name the tool."""
    assert "fs.write" in MODE_DIRECTIVE[WorkMode.CODE]
    assert "code.exec" in MODE_DIRECTIVE[WorkMode.CODE]
    assert "doc.write" in MODE_DIRECTIVE[WorkMode.DOCUMENT]


def test_the_document_directive_uses_the_reference_syntax_the_resolver_accepts():
    """MODE_DIRECTIVE is appended AFTER str.format, so its braces are literal.
    Writing the prompt's escaped form here would teach the model `{{{{step0}}}}`,
    which resolve_references never matches - and the document would then contain
    the literal token instead of the retrieved passage."""
    from kavach.agent.references import _REFERENCE

    assert _REFERENCE.search(MODE_DIRECTIVE[WorkMode.DOCUMENT]), (
        "the document directive does not contain a reference the resolver "
        "would substitute"
    )


def test_a_replan_keeps_the_mode():
    """A code run that hits a failure and replans is still a code run. Dropping
    the mode here makes the second half of a run a different kind of work from
    the first, with nothing on screen to say so."""
    router = _RecordingRouter()
    Planner(router=router).replan("goal", [], "it failed", mode=WorkMode.CODE)
    assert router.requests[0].forced_class is TaskClass.CODE_GENERATION


# ------------------------------------------------------------- persistence


def test_the_mode_survives_a_round_trip_through_sqlite(tmp_path):
    from kavach.agent.state import RunStore

    db = Database(tmp_path / "runs.db")
    store = RunStore(db)
    store.create(Run(goal="write a script", mode=WorkMode.CODE))
    run_id = store.list_runs()[0]["id"]

    assert store.load(run_id).mode is WorkMode.CODE
    assert store.timeline(run_id)["mode"] == "code"
    assert store.list_runs()[0]["mode"] == "code"


def test_a_database_written_before_the_column_existed_still_opens(tmp_path):
    """The venue box has a database from the last rehearsal.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so
    a new column is invisible on an older file and every read of it raises. The
    migration has to add it WITHOUT touching the rows already there - a run
    recorded before the toggle existed is an AUTO run, not a lost one.
    """
    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        """CREATE TABLE runs (
               id TEXT PRIMARY KEY, user TEXT NOT NULL, goal TEXT NOT NULL,
               label TEXT NOT NULL, status TEXT NOT NULL,
               budget_json TEXT NOT NULL, error TEXT,
               created_at REAL NOT NULL, updated_at REAL NOT NULL)"""
    )
    legacy.execute(
        "INSERT INTO runs VALUES ('run_old','local','older goal','INTERNAL',"
        "'done','{}',NULL,1.0,1.0)"
    )
    legacy.commit()
    legacy.close()

    db = Database(path)
    row = db.one("SELECT * FROM runs WHERE id='run_old'")
    assert row is not None, "the migration lost a run that was already recorded"
    assert row["goal"] == "older goal"
    assert row["mode"] == "auto"


def test_a_mode_this_build_does_not_recognise_does_not_make_a_run_unopenable(tmp_path):
    """Forward compatibility in the one direction that matters: a run recorded
    by a later build must still open here, as AUTO, rather than raising."""
    from kavach.agent.state import RunStore

    db = Database(tmp_path / "runs.db")
    store = RunStore(db)
    store.create(Run(id="run_future", goal="g"))
    db.execute("UPDATE runs SET mode='spreadsheet' WHERE id='run_future'")

    assert store.load("run_future").mode is WorkMode.AUTO


# ------------------------------------------------------------- the split


def _timeline(**over):
    base = {
        "run_id": "run_split",
        "status": "done",
        "mode": "auto",
        "artifacts": [],
        "steps": [],
    }
    base.update(over)
    return base


def _art(path, sha="ab" * 32):
    return {"path": path, "sha256": sha, "approved_by": None}


def test_every_artifact_lands_under_exactly_one_tab():
    """The partition property. An artifact matching neither rule would be
    counted in the run's artifact total and be invisible in both views - a
    deliverable that exists and cannot be found."""
    paths = ["npsh.py", "note.docx", "calc.xlsx", "deck.pptx", "readme.md",
             "trace.log", "scan.png", "weird.qqq", "noext"]
    out = split_outputs(_timeline(artifacts=[_art(p) for p in paths]))

    placed = ([i["path"] for i in out["code"]["items"] if i["kind"] == "file"]
              + [i["path"] for i in out["documents"]["items"]])
    assert sorted(placed) == sorted(paths)
    assert len(placed) == len(set(placed)), "an artifact appears under both tabs"


def test_source_files_are_code_and_office_files_are_documents():
    out = split_outputs(_timeline(artifacts=[_art("npsh.py"), _art("note.docx")]))
    assert [i["path"] for i in out["code"]["items"]] == ["npsh.py"]
    assert [i["path"] for i in out["documents"]["items"]] == ["note.docx"]


def test_an_unknown_extension_is_a_document_rather_than_nothing():
    """The fallback direction is deliberate: an unanticipated extension shows up
    somewhere a human will see it, instead of being dropped for not matching."""
    out = split_outputs(_timeline(artifacts=[_art("survey.dwg")]))
    assert [i["path"] for i in out["documents"]["items"]] == ["survey.dwg"]


def test_an_executed_snippet_is_code_even_though_it_is_not_an_artifact():
    """code.exec runs on a tmpfs that is destroyed with the container, so the
    source and its stdout are the ONLY surviving record that a sandboxed
    execution happened. Losing them loses the sandbox demo."""
    out = split_outputs(_timeline(steps=[{
        "idx": 0, "tool": "code.exec", "intent": "check the margin",
        "status": "passed", "args": {"code": "print(2 + 2)"},
        "observations": [{"stdout": "4", "stderr": "", "exit_code": 0,
                          "data": {"sandboxed": True}}],
    }]))
    item = out["code"]["items"][0]
    assert item["source"] == "print(2 + 2)"
    assert item["stdout"] == "4"
    assert item["sandboxed"] is True


def test_an_unsandboxed_execution_is_reported_as_such():
    """The dev fallback must not be indistinguishable from a container run.
    Defaulting a missing flag to True here would put a green 'sandboxed' badge
    on a snippet that ran against the host filesystem."""
    out = split_outputs(_timeline(steps=[{
        "idx": 0, "tool": "code.exec", "intent": "run it", "status": "passed",
        "args": {"code": "print(1)"},
        "observations": [{"stdout": "1", "exit_code": 0,
                          "data": {"sandboxed": False}}],
    }]))
    assert out["code"]["items"][0]["sandboxed"] is False


def test_a_step_that_has_not_run_is_neither_sandboxed_nor_unsandboxed():
    """Three states, not two. `not run yet` rendered as `NOT SANDBOXED` is a
    false alarm, and rendered as `sandboxed` is a false assurance."""
    out = split_outputs(_timeline(status="running", steps=[{
        "idx": 0, "tool": "code.exec", "intent": "run it", "status": "pending",
        "args": {"code": "print(1)"}, "observations": [],
    }]))
    item = out["code"]["items"][0]
    assert item["sandboxed"] is None
    assert item["ran"] is False


# ------------------------------------------------------------- empty sides


def test_an_empty_code_tab_on_a_code_run_says_the_toggle_was_not_honoured():
    """The most important empty state in the panel. A run asked for code and
    produced none is a failure, and a blank pane reads as 'still loading'."""
    note = split_outputs(_timeline(mode="code"))["code"]["note"]
    assert note and "Code mode" in note and "no code" in note


def test_an_empty_document_tab_on_a_document_run_says_the_same():
    note = split_outputs(_timeline(mode="document"))["documents"]["note"]
    assert note and "Document mode" in note


def test_an_empty_side_of_an_auto_run_is_worded_as_an_absence_not_a_failure():
    """An auto run that produced a document and no code did nothing wrong. The
    note must not accuse it of failing to honour a toggle nobody set."""
    out = split_outputs(_timeline(mode="auto", artifacts=[_art("note.docx")]))
    assert "Code mode" not in (out["code"]["note"] or "")
    assert out["documents"]["note"] is None, (
        "a populated side must not carry an explanation for being empty"
    )


def test_a_running_run_is_not_told_it_failed_to_deliver():
    """Mid-run, nothing is missing yet. Saying otherwise makes every run look
    broken for its first ten seconds."""
    out = split_outputs(_timeline(mode="code", status="running"))
    assert "failed" not in (out["code"]["note"] or "").lower()
    assert "Code mode" not in (out["code"]["note"] or "")


# ------------------------------------------------------------- previews


def _workspace_for(run_id: str) -> Path:
    from kavach.config import get_settings

    root = get_settings().workspace_root / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_a_document_preview_is_read_back_out_of_the_file():
    """The strong claim: these characters came out of the bytes a human will
    open. Anything weaker has to be labelled as weaker."""
    pytest.importorskip("docx")
    from docx import Document

    root = _workspace_for("run_preview")
    doc = Document()
    doc.add_paragraph("The trip setpoint is 2.1 barg.")
    doc.save(root / "note.docx")

    out = split_outputs(_timeline(run_id="run_preview", artifacts=[_art("note.docx")]))
    item = out["documents"]["items"][0]
    assert item["preview_source"] == "file"
    assert "2.1 barg" in item["preview"]
    assert item["preview_note"] is None


def test_a_preview_that_could_not_be_read_back_is_never_labelled_as_the_file():
    """The failure this guards is specific: a doc.write step that did not
    produce a file, showing the text it ASKED to write, under a label claiming
    the file was read. That makes a failed write look like a perfect document.
    """
    _workspace_for("run_missing")
    out = split_outputs(_timeline(
        run_id="run_missing",
        artifacts=[_art("gone.docx")],
        steps=[{
            "idx": 0, "tool": "doc.write", "intent": "draft it", "status": "passed",
            "args": {"path": "gone.docx", "body": "Findings: the pump is fine."},
            "observations": [],
        }],
    ))
    item = out["documents"]["items"][0]
    assert item["preview_source"] == "requested"
    assert item["preview_source"] != "file"
    assert "the pump is fine" in item["preview"]
    assert item["preview_note"], "a weaker preview must say why it is weaker"


def test_a_file_with_no_reader_on_this_box_says_so_rather_than_showing_nothing():
    root = _workspace_for("run_binary")
    (root / "drawing.dwg").write_bytes(b"\x00\x01binary")
    out = split_outputs(_timeline(run_id="run_binary", artifacts=[_art("drawing.dwg")]))
    item = out["documents"]["items"][0]
    assert item["preview"] is None
    assert "no reader" in item["preview_note"]


def test_a_code_artifact_shows_the_source_that_is_on_disk():
    root = _workspace_for("run_code")
    (root / "npsh.py").write_text("def npsh(p):\n    return p\n", encoding="utf-8")
    out = split_outputs(_timeline(run_id="run_code", artifacts=[_art("npsh.py")]))
    item = out["code"]["items"][0]
    assert "def npsh" in item["source"]
    assert item["language"] == "python"


def test_an_artifact_path_that_escapes_the_workspace_is_not_read():
    """A second way to turn a stored path into bytes must not be a second way
    out of the workspace. The item still appears - the deliverable is real and
    hiding it would be worse - but with no content and a reason."""
    out = split_outputs(_timeline(
        run_id="run_escape", artifacts=[_art("../../../etc/passwd")]))
    item = out["documents"]["items"][0]
    assert item["preview"] is None
    assert item["preview_note"]


def test_a_long_source_is_clipped_and_admits_it():
    """A preview that silently stops lies about the end of the document."""
    from kavach.agent.outputs import MAX_SOURCE_CHARS

    out = split_outputs(_timeline(steps=[{
        "idx": 0, "tool": "code.exec", "intent": "big", "status": "passed",
        "args": {"code": "x = 1\n" * MAX_SOURCE_CHARS}, "observations": [],
    }]))
    item = out["code"]["items"][0]
    assert item["truncated"] is True
    assert len(item["source"]) == MAX_SOURCE_CHARS


# ------------------------------------------------------------- the gateway


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from kavach.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_a_run_records_the_mode_it_was_started_with(client):
    body = client.post("/api/runs", json={
        "goal": "Write a script that prints the pressure drop",
        "mode": "code",
    }).json()
    assert body["mode"] == "code"
    assert client.get(f"/api/runs/{body['run_id']}/timeline").json()["mode"] == "code"


def test_a_client_that_never_heard_of_the_toggle_still_works(client):
    """The field is optional and defaults to the pre-toggle behaviour. An older
    page, a script, or a curl from the run sheet must not start failing."""
    body = client.post("/api/runs", json={"goal": "Summarise the SOP"}).json()
    assert body["mode"] == "auto"


def test_a_mode_that_does_not_exist_is_refused_rather_than_coerced(client):
    """422, not a silent fallback to auto. A typo'd mode that quietly routes as
    something else is a demo where the toggle appears not to work."""
    response = client.post("/api/runs", json={"goal": "x", "mode": "spreadsheet"})
    assert response.status_code == 422


def test_the_outputs_endpoint_serves_both_sides_of_the_toggle(client):
    run = client.post("/api/runs", json={
        "goal": "Calculate the pressure drop and write it to a file",
    }).json()
    body = client.get(f"/api/runs/{run['run_id']}/outputs").json()

    for side in ("code", "documents"):
        assert set(body[side]) == {"count", "items", "note"}
        assert body[side]["count"] == len(body[side]["items"])
    assert body["run_id"] == run["run_id"]


def test_the_outputs_endpoint_accounts_for_every_artifact_the_run_reported(client):
    """The two tabs together must equal the artifact list, or the panel is
    hiding a deliverable the same API is reporting two routes away."""
    run = client.post("/api/runs", json={
        "goal": "Calculate the pressure drop and write it to a file",
    }).json()
    outputs = client.get(f"/api/runs/{run['run_id']}/outputs").json()

    shown = {i["path"] for i in outputs["documents"]["items"]}
    shown |= {i["path"] for i in outputs["code"]["items"] if i["kind"] == "file"}
    assert shown == {a["path"] for a in run["artifacts"]}


def test_outputs_for_a_run_that_does_not_exist_is_a_404(client):
    assert client.get("/api/runs/run_nope/outputs").status_code == 404


def test_the_history_list_carries_the_mode_so_an_old_run_can_be_labelled(client):
    client.post("/api/runs", json={"goal": "Draft a note", "mode": "document"})
    runs = client.get("/api/runs?limit=10").json()["runs"]
    assert all("mode" in r for r in runs)


# --------------------------------------------------------------- the panel

# The UI has no build step, so nothing catches a toggle whose value the backend
# has never heard of. These are that check - the same role
# test_every_element_the_script_reaches_for_exists_in_the_page plays for ids.

_UI = Path(__file__).resolve().parents[1] / "services" / "kavach" / "ui"


def test_every_mode_button_on_the_page_is_a_mode_the_api_accepts():
    import re

    html = (_UI / "index.html").read_text(encoding="utf-8")
    offered = set(re.findall(r'data-mode="([^"]+)"', html))
    assert offered, "the page offers no mode buttons at all"
    unknown = offered - {m.value for m in WorkMode}
    assert not unknown, f"the page offers modes the API would reject: {sorted(unknown)}"


def test_every_mode_the_api_accepts_is_reachable_from_the_page():
    """A mode with no button is a mode nobody can use."""
    import re

    html = (_UI / "index.html").read_text(encoding="utf-8")
    offered = set(re.findall(r'data-mode="([^"]+)"', html))
    assert {m.value for m in WorkMode} <= offered


def test_the_forced_classes_the_panel_sends_are_real_task_classes():
    """The router panel posts a class string straight to /api/route/explain. A
    stale name there is a 422 the operator sees as "Explain does nothing"."""
    import re

    js = (_UI / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const FORCED_CLASS = \{(.*?)\};", js, re.DOTALL)
    assert block, "the panel's mode -> class map has been renamed or removed"
    names = re.findall(r'"([a-z_]+)"', block.group(1))
    for name in names:
        assert name in {t.value for t in TaskClass}, (
            f"the panel would post {name!r}, which is not a TaskClass"
        )


def test_the_panel_and_the_backend_agree_on_what_each_mode_routes_as():
    """Two copies of the same mapping, one in Python and one in JavaScript.
    They drift silently: the toggle keeps working, the explainer panel just
    starts describing a different decision from the one the run made."""
    import re

    js = (_UI / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const FORCED_CLASS = \{(.*?)\};", js, re.DOTALL)
    pairs = dict(re.findall(r"(\w+):\s*(?:\"([a-z_]+)\"|null)", block.group(1)))
    for mode in (WorkMode.CODE, WorkMode.DOCUMENT):
        assert pairs.get(mode.value) == mode.forced_class.value, (
            f"the panel routes {mode.value} as {pairs.get(mode.value)!r} but the "
            f"planner routes it as {mode.forced_class.value!r}"
        )


def test_each_output_tab_has_a_pane_to_show():
    """A tab whose pane id does not exist toggles `hide` on null and the click
    does nothing at all - with no error a viewer can see."""
    import re

    html = (_UI / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', html))
    for tab in re.findall(r'data-out="([^"]+)"', html):
        assert f"out-{tab}" in ids, f"the {tab} tab has no pane"


# ------------------------------------------- a step that never reached the tool


def test_a_step_that_failed_before_the_tool_ran_is_not_shown_as_exit_zero():
    """The trap this pins, seen on a box with no Docker.

    Observation defaults exit_code to 0, so a code.exec step refused by
    SandboxUnavailable reports exit 0 with its reason in stderr. Reading the
    badge off exit_code alone puts a green tick directly above the message
    explaining that nothing ran. The tool's own verdict has to travel with it.
    """
    out = split_outputs(_timeline(steps=[{
        "idx": 0, "tool": "code.exec", "intent": "run it", "status": "failed",
        "args": {"code": "print(1)"},
        "observations": [{
            "ok": False, "stdout": "", "exit_code": 0, "data": {},
            "stderr": "SandboxUnavailable: code.exec needs the Docker sandbox",
        }],
    }]))
    item = out["code"]["items"][0]
    assert item["ok"] is False
    assert item["exit_code"] == 0, "the trap itself - this is why ok must exist"
    assert item["sandboxed"] is None, (
        "a step that never reached the sandbox is not an unsandboxed execution"
    )


def test_the_panel_reads_the_verdict_before_the_exit_code():
    """Static check on the renderer, since the UI has no build step: the `ok`
    branch must come before the `exit_code === 0` branch or the green badge
    wins on exactly the failure above."""

    js = (_UI / "app.js").read_text(encoding="utf-8")
    body = js.split("function renderCode(", 1)[1]
    ok_at = body.find("item.ok === false")
    exit_at = body.find("item.exit_code === 0")
    assert ok_at != -1, "renderCode no longer checks the tool's verdict at all"
    assert ok_at < exit_at, (
        "renderCode badges on the exit code before the verdict, so a step that "
        "never ran is drawn as a success"
    )


# ------------------------------------------------------- the mock honours it

# The mock is what every box runs before weights are staged, so a mock that
# ignores the mode makes the toggle look broken on exactly the machine a judge
# sees first.


def _mock_plan(goal: str, mode: WorkMode):
    """The plan the mock backend really produces for this mode, VALIDATED.

    Built from the planner's own system prompt rather than a fixture of it, and
    then put through validate_plan - the same gate a live run uses. That second
    half matters: a mock emitting a plan the validator rejects fails every run
    in that mode before a single step executes, and the error would name the
    planner rather than the backend that wrote it.
    """
    import random

    from kavach.agent.planner import (
        MODE_DIRECTIVE,
        PLANNER_SYSTEM,
        _tool_catalogue,
        validate_plan,
    )
    from kavach.runtime.backends.mock import MockBackend
    from kavach.tools import get_tool_registry
    from kavach.types import GenRequest

    system = PLANNER_SYSTEM.format(
        tools=_tool_catalogue(get_tool_registry()), max_steps=12
    ) + MODE_DIRECTIVE[mode]
    raw = MockBackend()._plan_response(
        GenRequest(prompt=goal, system=system), random.Random(0)
    )
    return validate_plan(json.loads(raw))


def test_the_mock_plans_a_program_and_a_sandboxed_run_in_code_mode():
    steps = _mock_plan("compute the line velocity", WorkMode.CODE)
    tools = [s.tool for s in steps]
    assert "code.exec" in tools, "no sandboxed execution - the Code tab is empty"
    written = [s.args.get("path", "") for s in steps if s.tool == "fs.write"]
    assert any(p.endswith(".py") for p in written), (
        f"code mode wrote {written}, none of which is source"
    )


def test_the_mock_plans_an_office_file_in_document_mode():
    steps = _mock_plan("draft the approval note", WorkMode.DOCUMENT)
    written = [s.args.get("path", "") for s in steps if s.tool == "doc.write"]
    assert any(p.endswith((".docx", ".xlsx", ".pptx")) for p in written), (
        f"document mode wrote {written}, none of which anyone can open and sign"
    )


def test_the_mocks_code_step_is_never_judged_by_a_file_it_cannot_create():
    """The sandbox is a tmpfs destroyed with the container. A code.exec step
    whose acceptance names a file fails every time however good the code is -
    and validate_plan refuses the plan outright, so the mock modelling it wrong
    would fail every code-mode run before anything executed."""
    steps = _mock_plan("compute the line velocity", WorkMode.CODE)
    for step in steps:
        if step.tool == "code.exec":
            assert ".py" not in step.acceptance and "file" not in step.acceptance


def test_everything_the_mock_writes_is_still_stamped_as_simulated():
    """Honouring the mode must not have turned the mock into something whose
    output can be screenshotted as a real deliverable."""
    for mode in (WorkMode.CODE, WorkMode.DOCUMENT):
        steps = _mock_plan("do the work", mode)
        written = " ".join(
            str(v) for s in steps for v in s.args.values() if isinstance(v, str)
        )
        assert "SIMULATED" in written, f"{mode.value} output carries no marker"
