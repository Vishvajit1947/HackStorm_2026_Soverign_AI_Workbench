"""Port allocation and server identity for spawned llama-server processes.

The bug these pin: two models were handed the SAME port. A 7B on CPU takes tens
of seconds to bind, and every port probe during that window still saw the port
as free, so the embedder and the text model were both assigned 18000. One lost
the bind race; the survivor answered /health for both, so both were marked
"alive" and the residency table looked perfect. The only symptom was an opaque
500 from /completion - because the request had reached the EMBEDDING server,
which has no such endpoint.
"""

from __future__ import annotations

import socket

import pytest

from kavach.runtime.backends.base import BackendError
from kavach.runtime.backends.llamacpp import DEFAULT_BIND, _free_port


def test_ports_already_handed_out_are_skipped():
    """The regression. The probe socket closes before llama-server binds, so the
    OS cannot yet report the port as taken - `exclude` is the only thing that
    can, and without it two loads in the same window collide."""
    first = _free_port()
    assert _free_port(exclude={first}) != first


def test_exclude_accumulates_across_several_allocations():
    """One slow-loading model is enough to keep a port unbound while several
    more are admitted, so skipping only the most recent port is not enough."""
    handed_out: set[int] = set()
    for _ in range(5):
        port = _free_port(exclude=handed_out)
        assert port not in handed_out
        handed_out.add(port)
    assert len(handed_out) == 5


def test_a_returned_port_is_actually_bindable():
    """A port skipped for the wrong reason is as bad as one handed out twice."""
    port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((DEFAULT_BIND, port))


def test_a_port_with_a_live_llama_server_is_never_handed_out():
    """The primary regression, and it only reproduces with the listener set up
    the way llama-server sets ITS socket up: on Windows, SO_REUSEADDR on the
    probe permits binding an address another socket is actively listening on,
    but only when that listener also set the flag. A listener without it (the
    easy thing to write in a test) refuses the probe either way and the test
    passes against the bug."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((DEFAULT_BIND, 0))
    s.listen(1)
    try:
        taken = s.getsockname()[1]
        for _ in range(20):
            assert _free_port(start=taken) != taken
    finally:
        s.close()


# ------------------------------------------------------- identity on readiness

class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _backend_with_props(tmp_path, props):
    """A backend whose HTTP client reports a fixed /props, so readiness can be
    exercised without spawning a real llama-server."""
    from kavach.runtime.backends.llamacpp import LlamaCppBackend

    backend = LlamaCppBackend(model_dir=tmp_path)

    class _Client:
        def get(self, url, timeout=None):
            return _FakeResponse(props)

    backend._client = _Client()
    return backend


def _server(tmp_path, backend, filename="wanted.gguf"):
    from kavach.runtime.backends.llamacpp import _Server
    from kavach.runtime.registry import ModelSpec

    spec = ModelSpec(
        id="wanted", file=filename, license="apache-2.0", tier=2,
        capabilities=["text"], context=2048, vram_gb=1.0, load_ms=100,
    )
    return _Server(spec, 18000, None)


def test_a_server_serving_a_different_model_is_refused(tmp_path):
    """The crossed-wire case. Answering /health is not proof of identity."""
    other = tmp_path / "somethingelse.gguf"
    backend = _backend_with_props(tmp_path, {"model_path": str(other)})
    server = _server(tmp_path, backend)

    with pytest.raises(BackendError) as exc:
        backend._assert_serving_our_weights(server)
    assert "same port" in str(exc.value)


def test_a_server_that_will_not_identify_itself_is_refused(tmp_path):
    """No model_path means we cannot prove which model is behind the port, and
    an unidentifiable server is not one we will route generation to."""
    backend = _backend_with_props(tmp_path, {})
    server = _server(tmp_path, backend)

    with pytest.raises(BackendError, match="unidentifiable"):
        backend._assert_serving_our_weights(server)


def test_the_right_weights_pass(tmp_path):
    wanted = tmp_path / "wanted.gguf"
    backend = _backend_with_props(tmp_path, {"model_path": str(wanted)})
    backend._assert_serving_our_weights(_server(tmp_path, backend))
