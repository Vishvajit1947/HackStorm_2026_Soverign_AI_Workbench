"""cli.py: a short-lived command must tear down whatever it spawned.

`kavach serve` is long-lived and already tears down through the warden in its
FastAPI shutdown hook (api/app.py). Every other command - `route --run`,
`models load`, `run` - is a one-shot process: on the llamacpp backend it can
admit a model, which spawns a real `llama-server` child process, and then the
process exits with nothing left to kill it. That server outlives the CLI
invocation, holding its GPU/RAM allocation until the box runs out of one or
the other - which then looks like an unrelated failure the next time anything
tries to load a model. Found by running `kavach route --run` a few times in a
row against real weights and watching free RAM disappear.
"""

from __future__ import annotations

import argparse

from kavach import cli


class _FakeBackend:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeWarden:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.evict_all_calls: list[bool] = []

    def evict_all(self, include_pinned: bool = False) -> list[str]:
        self.evict_all_calls.append(include_pinned)
        return []


class _StubParser:
    def __init__(self, func) -> None:
        self._func = func

    def parse_args(self, argv):
        return argparse.Namespace(func=self._func)


def test_a_command_that_never_touched_routing_spawns_nothing_to_clean_up(monkeypatch):
    """peek_warden() must never construct a warden. A command like `kavach
    audit tail` has no business pinning profile models - and spawning their
    servers - just because the exit hook asked whether there was anything to
    tear down."""
    peeked: list[bool] = []
    monkeypatch.setattr("kavach.runtime.peek_warden", lambda: peeked.append(True) or None)

    cli._shutdown_any_spawned_backend()

    assert peeked == [True]


def test_an_active_warden_is_torn_down_the_same_way_the_gateway_does_it(monkeypatch):
    """Mirrors api/app.py's lifespan shutdown exactly: evict_all(include_pinned=
    True) then backend.shutdown(). A pinned embedder server must not survive
    the CLI process that admitted it, any more than an unpinned one would."""
    backend = _FakeBackend()
    warden = _FakeWarden(backend)
    monkeypatch.setattr("kavach.runtime.peek_warden", lambda: warden)

    cli._shutdown_any_spawned_backend()

    assert warden.evict_all_calls == [True]
    assert backend.shutdown_calls == 1


def test_main_cleans_up_after_a_command_that_admitted_a_model(monkeypatch):
    """End-to-end through main(): a command function that touches routing must
    not leave its backend process dangling once main() returns control to the
    shell - success or failure, since the process is exiting either way."""
    backend = _FakeBackend()
    warden = _FakeWarden(backend)
    monkeypatch.setattr("kavach.runtime.peek_warden", lambda: warden)
    monkeypatch.setattr(cli, "build_parser", lambda: _StubParser(lambda args: 0))

    rc = cli.main([])

    assert rc == 0
    assert warden.evict_all_calls == [True]
    assert backend.shutdown_calls == 1


def test_main_cleans_up_even_when_the_command_raises(monkeypatch):
    """A crashed command is exactly when a spawned llama-server is most likely
    to be orphaned - the exception must not skip the teardown."""
    backend = _FakeBackend()
    warden = _FakeWarden(backend)
    monkeypatch.setattr("kavach.runtime.peek_warden", lambda: warden)

    def _boom(args):
        raise RuntimeError("simulated command failure")

    monkeypatch.setattr(cli, "build_parser", lambda: _StubParser(_boom))

    rc = cli.main([])

    assert rc == 2
    assert warden.evict_all_calls == [True]
    assert backend.shutdown_calls == 1
