"""llama.cpp backend.

The warden decides *when* a model occupies VRAM; this makes that real by spawning
one `llama-server` process per resident model and killing it on eviction. That is
what gives genuine load/unload on a consumer card - an in-process library would
hold the allocation until the interpreter exits.

Two modes:

  spawn   (default) we start and stop `llama-server` ourselves, one port per model.
          The warden's eviction actually frees VRAM. This is the demo configuration.

  static  we talk to servers someone else started, declared in
          config/endpoints.yaml. load()/unload() become no-ops and the VRAM meter
          reports what the manifests claim rather than what is measured. Useful
          when llama.cpp runs outside compose; say so on the slide if you use it.

Every request goes to 127.0.0.1 or a kavach_core address. There is no code path
here that can reach a public endpoint, and the base URL is asserted to be a
literal IP so a hostname can never be resolved.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ...config import get_settings
from ...types import GenRequest, GenResult
from ..registry import ModelSpec
from .base import Backend, BackendError, ModelNotLoaded

DEFAULT_BIND = "127.0.0.1"
PORT_RANGE_START = 18000
STARTUP_TIMEOUT_S = 180.0
REQUEST_TIMEOUT_S = 600.0

# ------------------------------------------------------------- orphan reaping
#
# `shutdown()` kills the servers this process started, and covers every ORDERLY
# exit. It cannot cover the ones that matter: Ctrl+C at the wrong moment, a
# taskkill, a crash, an IDE stopping the debugger. The children are separate
# processes, so they survive - still holding the card.
#
# Found with seven of them alive on a 6 GB laptop GPU, the oldest three hours
# old, 5926 MiB of 6144 MiB occupied by models nothing was using. The next run
# then had 200 MB to load into, and instead of failing it STALLED: llama.cpp
# thrashed, the request sat inside the 600-second timeout, and the panel showed
# "running" with no output for ten minutes.
#
# A pidfile rather than process enumeration: it is stdlib, it costs nothing at
# boot, and it only ever names processes this product started. Nothing is killed
# on the strength of the pid alone - the port is asked whether a llama.cpp
# server is answering there first, because a pid can be recycled.
ORPHAN_FILE = "llama-servers.json"


def _orphan_path() -> Path:
    return get_settings().var_dir / ORPHAN_FILE


def _load_records() -> list[dict[str, Any]]:
    try:
        raw = _orphan_path().read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _save_records(records: list[dict[str, Any]]) -> None:
    try:
        path = _orphan_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    except OSError:
        # Bookkeeping must never be able to stop a model loading. The cost of
        # failing here is a leak on the next hard kill, not a broken run.
        pass


def _remember(port: int, pid: int, model_id: str) -> None:
    records = [r for r in _load_records() if r.get("port") != port]
    records.append({"port": port, "pid": pid, "model": model_id, "at": time.time()})
    _save_records(records)


def _forget(port: int) -> None:
    _save_records([r for r in _load_records() if r.get("port") != port])


def _is_llama_server(port: int) -> bool:
    """Is a llama.cpp server answering on this port right now?

    The guard on killing by pid. `/props` is llama.cpp's own endpoint and no
    other process on this box serves it, so a positive answer identifies the
    process as one of ours rather than whatever inherited the number.
    """
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"http://{DEFAULT_BIND}:{port}/props")
        return resp.status_code == 200 and "model_path" in resp.text
    except httpx.HTTPError:
        return False


def reap_orphans() -> list[dict[str, Any]]:
    """Kill llama-servers left behind by a previous gateway. Returns what it killed.

    Called at boot, where it is always correct: a process that has just started
    owns no servers, so anything in the file is by definition an orphan.
    """
    killed: list[dict[str, Any]] = []
    for record in _load_records():
        port, pid = record.get("port"), record.get("pid")
        if not isinstance(port, int) or not isinstance(pid, int):
            continue
        if not _is_llama_server(port):
            continue                    # already gone, or the pid was recycled
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True, timeout=15, check=False,
                )
            else:
                os.kill(pid, signal.SIGTERM)
            killed.append(record)
        except (OSError, subprocess.SubprocessError):
            # Reported by omission rather than raised: a boot that refuses to
            # start because it could not tidy up is worse than one that starts
            # with less VRAM than it hoped for.
            continue
    _save_records([])
    return killed


def _assert_local(url: str) -> None:
    """Refuse anything that is not a literal local/private address.

    Belt and braces: the container has no route out anyway, but a config typo
    should fail loudly here rather than look like a working feature.
    """
    host = urlparse(url).hostname or ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise BackendError(
            f"backend endpoint {url!r} is not a literal IP address. Kavach will not "
            "resolve hostnames - put the address in config/endpoints.yaml."
        ) from exc
    if not (ip.is_loopback or ip.is_private):
        raise BackendError(
            f"backend endpoint {url!r} is not loopback or private. Refusing."
        )


def _free_port(start: int = PORT_RANGE_START, exclude: set[int] | None = None) -> int:
    """Find a port to hand to a new llama-server.

    Two independent ways this hands out a port that is already serving another
    model. The symptom of either is a generation request arriving at the
    EMBEDDING server and coming back as an opaque 500, while both entries report
    "alive" - because both really are.

    1. SO_REUSEADDR must NOT be set on the probe. On Windows the flag lets a
       socket bind an address another socket is actively LISTENING on, provided
       that listener also set it - and llama-server does. So the probe below,
       with the flag set, succeeds against a live llama-server and calls its
       port free. On Linux the flag only relaxes TIME_WAIT and the probe is
       honest, which is why this is invisible off Windows. Verified both ways:
       the probe reports "free" against a SO_REUSEADDR listener only when the
       probe sets the flag too.
    2. The probe socket is closed before llama-server is spawned, let alone
       bound, so the result is a hint and NOT a reservation. The window is wide
       - a 7B on CPU takes tens of seconds to bind - and `exclude` carries the
       ports already handed out in this process, which the OS cannot report as
       taken until a child has actually bound them.

    _await_ready() then covers what is left by checking WHICH model answered
    rather than trusting that anything answering means success.
    """
    taken = exclude or set()
    for port in range(start, start + 500):
        if port in taken:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((DEFAULT_BIND, port))
                return port
            except OSError:
                continue
    raise BackendError("no free port for llama-server in the 18000-18500 range")


class _Server:
    def __init__(self, spec: ModelSpec, port: int, proc: subprocess.Popen | None):
        self.spec = spec
        self.port = port
        self.proc = proc
        self.base_url = f"http://{DEFAULT_BIND}:{port}"
        self.started_at = time.monotonic()


def _as_data_url(image: Any) -> str:
    """llama.cpp's chat endpoint takes images as data URLs.

    Accepts raw bytes or an already-encoded string, because callers upstream
    have both and guessing wrong produces a silently image-less request.
    """
    if isinstance(image, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(image)).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    text = str(image)
    return text if text.startswith("data:") else f"data:image/png;base64,{text}"


def _find_binary() -> str:
    """Where llama-server is, in the order an operator would expect.

    PATH first, because that is what the install guide documents and what a
    Linux appliance will always use. Then `KAVACH_LLAMA_BIN_DIR`, for the
    Windows case the guide otherwise handles with a line of PowerShell that has
    to be repeated in every shell - forgetting it is the single most common
    reason a box with correctly staged weights still answers from the mock.

    Falls back to the bare name so the error, when there is one, is
    llama.cpp's own rather than ours.
    """
    found = shutil.which("llama-server")
    if found:
        return found

    from ...config import get_settings

    bin_dir = get_settings().llama_bin_dir
    if bin_dir:
        for name in ("llama-server.exe", "llama-server"):
            candidate = bin_dir / name
            if candidate.is_file():
                return str(candidate)
    return "llama-server"


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(
        self,
        model_dir: Path,
        binary: str | None = None,
        mode: str = "spawn",
        n_gpu_layers: int = 999,
        extra_args: list[str] | None = None,
    ):
        self.model_dir = model_dir
        self.mode = mode
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = extra_args or []
        self.binary = binary or _find_binary()
        self._servers: dict[str, _Server] = {}
        self._lock = threading.RLock()
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT_S)

    # ---------------------------------------------------------------- lifecycle

    def load(self, spec: ModelSpec) -> None:
        with self._lock:
            if spec.id in self._servers:
                return

            if self.mode == "static":
                raise BackendError(
                    "static mode cannot load models on demand - start the server "
                    "yourself and declare it in config/endpoints.yaml"
                )

            weights = spec.path or (self.model_dir / spec.file)
            if not weights.exists():
                raise BackendError(
                    f"weights for {spec.id} are not staged at {weights}. "
                    "Copy the GGUF onto the box (§10: bring them on an SSD)."
                )

            port = _free_port(exclude={s.port for s in self._servers.values()})
            cmd = [
                self.binary,
                "--model", str(weights),
                "--host", DEFAULT_BIND,
                "--port", str(port),
                "--ctx-size", str(spec.context),
                "--n-gpu-layers", "0" if spec.vram_gb == 0 else str(spec.n_gpu_layers if spec.n_gpu_layers is not None else self.n_gpu_layers),
                # No web UI, no metrics endpoint, nothing listening we did not ask for.
                "--no-webui",
                *self.extra_args,
            ]
            if spec.is_embedder:
                cmd.append("--embedding")

            # The projector turns an image into tokens the model can attend to.
            # Without it llama.cpp still loads a vision model, still answers, and
            # never looks at the picture - which is the single most dangerous
            # thing this backend could do with a P&ID. So the file is resolved
            # and checked here, and generate() refuses images unless it loaded.
            projector = self._projector_for(spec)
            if projector is not None:
                # llama.cpp warns on load that Qwen-VL needs at least 1024 image
                # tokens to be reliable on GROUNDING tasks - reading a label off
                # a specific place in the picture. That is exactly what a P&ID
                # is, so the floor is set rather than left to the default.
                cmd += ["--mmproj", str(projector), "--image-min-tokens", "1024"]

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise BackendError(
                    f"llama-server not found at {self.binary!r}. Install llama.cpp or "
                    "set KAVACH_BACKEND=mock for development."
                ) from exc

            server = _Server(spec, port, proc)
            # Recorded BEFORE the readiness wait. A server that is still loading
            # when the gateway is killed is exactly the one that leaks: it holds
            # VRAM, it is not in `self._servers` yet, and nothing on the next
            # boot knows it exists.
            _remember(port, proc.pid, spec.id)
            try:
                self._await_ready(server)
            except BackendError:
                _forget(port)
                raise
            self._servers[spec.id] = server

    def _assert_serving_our_weights(self, server: _Server) -> None:
        """Confirm the process answering on this port is the one we spawned.

        A 200 from /health only proves SOMETHING is listening. When a port is
        handed out twice the loser of the bind race inherits the winner's
        health check and is marked ready, and the mismatch only surfaces later
        as a 500 from an endpoint the other model does not implement - which
        reads like a broken model rather than a crossed wire. Fail loudly here
        instead: a server we cannot identify is not a server we will route to.
        """
        expected = (server.spec.path or (self.model_dir / server.spec.file)).resolve()
        try:
            props = self._client.get(f"{server.base_url}/props", timeout=2.0).json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError(
                f"llama-server for {server.spec.id} on port {server.port} answered "
                f"/health but not /props, so it cannot be identified: {exc}"
            ) from exc

        actual = props.get("model_path")
        if actual is None:
            raise BackendError(
                f"llama-server for {server.spec.id} on port {server.port} reports no "
                "model_path - refusing to route to an unidentifiable server"
            )
        if Path(actual).resolve() != expected:
            raise BackendError(
                f"port {server.port} was assigned to {server.spec.id} but the server "
                f"there is serving {actual!r}, not {str(expected)!r}. Two models were "
                "given the same port; refusing to route generation to the wrong one."
            )

    def _await_ready(self, server: _Server) -> None:
        _assert_local(server.base_url)
        deadline = time.monotonic() + STARTUP_TIMEOUT_S

        while time.monotonic() < deadline:
            if server.proc is not None and server.proc.poll() is not None:
                output = ""
                if server.proc.stdout is not None:
                    output = server.proc.stdout.read()[-2000:]
                raise BackendError(
                    f"llama-server for {server.spec.id} exited with code "
                    f"{server.proc.returncode}:\n{output}"
                )
            try:
                resp = self._client.get(f"{server.base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    self._assert_serving_our_weights(server)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)

        self._kill(server)
        raise BackendError(
            f"llama-server for {server.spec.id} did not become healthy within "
            f"{STARTUP_TIMEOUT_S:.0f}s"
        )

    def unload(self, spec: ModelSpec) -> None:
        with self._lock:
            server = self._servers.pop(spec.id, None)
            if server is not None:
                self._kill(server)

    @staticmethod
    def _kill(server: _Server) -> None:
        if server.proc is None:
            return
        server.proc.terminate()
        try:
            server.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.proc.kill()
            server.proc.wait(timeout=5)
        # Reaped properly, so the next boot has nothing to clean up for it.
        _forget(server.port)

    def _projector_for(self, spec: ModelSpec) -> Path | None:
        """The mmproj beside the weights, or None.

        A missing projector is NOT an error at load time: a vision model with no
        projector is still a perfectly good text model, and refusing to load it
        would take the whole profile down over a capability the run may never
        use. It becomes an error only when an image is actually sent.
        """
        if not spec.mmproj:
            return None
        candidate = get_settings().model_dir / spec.mmproj
        if not candidate.exists():
            return None
        return candidate

    def projector_loaded(self, spec: ModelSpec) -> bool:
        return self._projector_for(spec) is not None

    def shutdown(self) -> None:
        with self._lock:
            for server in list(self._servers.values()):
                self._kill(server)
            self._servers.clear()
        self._client.close()

    def loaded_models(self) -> list[str]:
        with self._lock:
            return sorted(self._servers)

    # ---------------------------------------------------------------- inference

    def _server_for(self, spec: ModelSpec) -> _Server:
        with self._lock:
            server = self._servers.get(spec.id)
        if server is None:
            raise ModelNotLoaded(
                f"{spec.id} is not loaded - the warden's residency table and the "
                "backend have diverged"
            )
        return server

    def _chat_prompt(self, server: _Server, req: GenRequest) -> str:
        """Render the prompt through the MODEL'S OWN chat template.

        An instruct model that is handed a bare prompt does not answer it, it
        continues it. Asked for three causes of pump vibration, qwen2.5-7b
        produced a plausible list, then "Answer:", then an unrelated multiple
        choice question about the Internet of Things - because a raw
        /completion is text continuation and the training data is full of
        quizzes. The answer is in there, so the failure survives a casual read
        of the panel; only the tail gives it away.

        The template is asked of the server rather than rendered here: it is
        baked into each GGUF, differs per model family, and llama-server is the
        only thing that can be right about it by construction. That costs one
        loopback round-trip per generation, against a wrong answer otherwise.

        Note there is no fall back to the raw prompt. A model whose GGUF
        carries no chat template is a base model, and quietly continuing text
        for it is the exact failure this method exists to remove - it would
        look like a working answer with a ruined tail.
        """
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        try:
            resp = self._client.post(
                f"{server.base_url}/apply-template", json={"messages": messages}
            )
            resp.raise_for_status()
            prompt = resp.json()["prompt"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise BackendError(
                f"{server.spec.id} could not apply its chat template ({exc}). "
                "Refusing to send a raw prompt to an instruct model: it would "
                "continue the text instead of answering it."
            ) from exc
        return prompt

    def generate(self, spec: ModelSpec, req: GenRequest) -> GenResult:
        server = self._server_for(spec)
        started = time.perf_counter()

        payload: dict[str, Any] = {
            # Templated, never the bare req.prompt - see _chat_prompt().
            "prompt": self._chat_prompt(server, req),
            "n_predict": req.max_tokens,
            "temperature": req.temperature,
            "stop": req.stop,
            "cache_prompt": True,
        }
        endpoint = "/completion"
        if req.grammar:
            # GBNF-constrained decoding: malformed tool JSON becomes structurally
            # impossible rather than merely unlikely (§05).
            payload["grammar"] = req.grammar
        if req.images:
            # Not "not implemented yet" - actively dangerous. This build takes
            # image_data on /completion, returns 200, and answers from the text
            # alone: verified with a 1x1 PNG, tokens_evaluated counted only the
            # prompt. Images moved to mtmd, and load() passes no --mmproj, so
            # there is no projector resident to encode them with either way.
            # A confident answer about a scan nobody looked at is the worst
            # thing this backend could return, so refuse to be the one to
            # return it.
            if not self.projector_loaded(spec):
                raise BackendError(
                    f"{spec.id} was given {len(req.images)} image(s), but no "
                    "--mmproj projector is staged for it, so the image cannot "
                    "reach the model. Refusing to answer a visual question from "
                    "the text alone: stage the projector and set `mmproj:` on "
                    "this model in config/models.yaml."
                )
            # With a projector resident the server takes images on /chat/completions
            # as data URLs (mtmd). /completion still discards image_data silently,
            # so this path deliberately does NOT reuse it.
            payload["messages"] = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": payload.pop("prompt", "")},
                    *[
                        {"type": "image_url",
                         "image_url": {"url": _as_data_url(image)}}
                        for image in req.images
                    ],
                ],
            }]
            payload.pop("n_predict", None)
            endpoint = "/v1/chat/completions"

        try:
            resp = self._client.post(f"{server.base_url}{endpoint}", json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"generation failed on {spec.id}: {exc}") from exc

        # The two endpoints answer in different shapes. Reading the wrong one
        # yields an empty string rather than an error, which would look exactly
        # like a model that had nothing to say about the image.
        if endpoint != "/completion":
            choices = body.get("choices") or [{}]
            text = (choices[0].get("message") or {}).get("content", "")
            usage = body.get("usage", {})
            return GenResult(
                text=text,
                model_id=spec.id,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                finish_reason=choices[0].get("finish_reason") or "stop",
            )

        timings = body.get("timings", {})
        return GenResult(
            text=body.get("content", ""),
            model_id=spec.id,
            prompt_tokens=int(timings.get("prompt_n", 0)),
            completion_tokens=int(timings.get("predicted_n", 0)),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            finish_reason="length" if body.get("truncated") else "stop",
        )

    def embed(self, spec: ModelSpec, texts: list[str]) -> list[list[float]]:
        server = self._server_for(spec)
        try:
            resp = self._client.post(f"{server.base_url}/v1/embeddings",
                                     json={"input": texts, "model": spec.id})
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"embedding failed on {spec.id}: {exc}") from exc

        return [item["embedding"] for item in sorted(
            body.get("data", []), key=lambda d: d.get("index", 0)
        )]

    # ---------------------------------------------------------------- health

    def health(self) -> dict[str, Any]:
        with self._lock:
            servers = {
                model_id: {
                    "port": s.port,
                    "uptime_s": round(time.monotonic() - s.started_at, 1),
                    "alive": s.proc is None or s.proc.poll() is None,
                }
                for model_id, s in self._servers.items()
            }
        return {
            "backend": self.name,
            "ok": True,
            "simulated": False,
            "mode": self.mode,
            "binary": self.binary,
            "binary_present": shutil.which(self.binary) is not None
            or Path(self.binary).exists(),
            "loaded": sorted(servers),
            "servers": servers,
        }
