"""Prompts reach the model through its own chat template.

The bug this pins: generate() POSTed the bare prompt to /completion, so an
instruct model CONTINUED it instead of answering it. Asked for three causes of
centrifugal pump vibration, qwen2.5-7b listed three, wrote "Answer:", and then
began an unrelated multiple-choice question about the Internet of Things. The
requested answer is present, so the panel looks fine until you read the tail -
which is what makes it worth a regression test rather than a code comment.
"""

from __future__ import annotations

import pytest

from kavach.runtime.backends.base import BackendError
from kavach.runtime.backends.llamacpp import LlamaCppBackend, _Server
from kavach.runtime.registry import ModelSpec
from kavach.types import GenRequest

TEMPLATED = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _RecordingClient:
    """Stands in for llama-server, recording what each endpoint was sent."""

    def __init__(self, template_prompt=TEMPLATED, template_fails=False):
        self.template_prompt = template_prompt
        self.template_fails = template_fails
        self.seen: dict[str, dict] = {}

    def post(self, url, json=None):
        self.seen[url.rsplit("/", 1)[-1]] = json
        if url.endswith("/apply-template"):
            if self.template_fails:
                raise ValueError("this GGUF carries no chat template")
            return _Response({"prompt": self.template_prompt})
        return _Response({"content": "an answer", "timings": {}})


def _backend(tmp_path, client):
    backend = LlamaCppBackend(model_dir=tmp_path)
    backend._client = client
    spec = ModelSpec(
        id="text", file="text.gguf", license="apache-2.0", tier=2,
        capabilities=["text"], context=2048, vram_gb=1.0, load_ms=100,
    )
    backend._servers["text"] = _Server(spec, 18000, None)
    return backend, spec


def test_the_model_never_receives_the_bare_prompt(tmp_path):
    """The regression itself."""
    client = _RecordingClient()
    backend, spec = _backend(tmp_path, client)

    backend.generate(spec, GenRequest(prompt="hello"))

    assert client.seen["completion"]["prompt"] == TEMPLATED
    assert client.seen["completion"]["prompt"] != "hello"


def test_the_system_prompt_goes_through_the_template(tmp_path):
    """It used to be sent as `system_prompt`, a field this llama-server no
    longer reads - so it was dropped in full, silently."""
    client = _RecordingClient()
    backend, spec = _backend(tmp_path, client)

    backend.generate(spec, GenRequest(prompt="hello", system="You are terse."))

    messages = client.seen["apply-template"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are terse."}
    assert messages[-1] == {"role": "user", "content": "hello"}
    assert "system_prompt" not in client.seen["completion"]


def test_a_model_with_no_chat_template_is_refused_not_downgraded(tmp_path):
    """Falling back to the raw prompt would restore exactly the bug above, and
    it would look like a working answer with a ruined tail."""
    client = _RecordingClient(template_fails=True)
    backend, spec = _backend(tmp_path, client)

    with pytest.raises(BackendError, match="chat template"):
        backend.generate(spec, GenRequest(prompt="hello"))
    assert "completion" not in client.seen


def test_grammar_survives_templating(tmp_path):
    """GBNF-constrained tool calls are the one path that must not change shape
    here - malformed tool JSON has to stay structurally impossible."""
    client = _RecordingClient()
    backend, spec = _backend(tmp_path, client)

    backend.generate(spec, GenRequest(prompt="hello", grammar='root ::= "ok"'))

    assert client.seen["completion"]["grammar"] == 'root ::= "ok"'


def test_images_are_refused_rather_than_silently_dropped(tmp_path):
    """/completion accepts image_data, returns 200 and answers from the text
    alone - verified against a real server with a 1x1 PNG. No --mmproj is
    passed at load either, so there is no projector to encode with. A confident
    answer about a scan nobody looked at must not be reachable."""
    client = _RecordingClient()
    backend, spec = _backend(tmp_path, client)

    with pytest.raises(BackendError, match="Refusing to answer a visual question"):
        backend.generate(spec, GenRequest(prompt="what is this?", images=["base64=="]))
