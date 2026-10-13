"""Model backends.

    mock      deterministic stub, no weights, no GPU. Dev + CI + the TINY rehearsal.
    llamacpp  real llama.cpp servers on the internal docker network.

The warden owns *when* a model is loaded; a backend owns *how*. Keeping those apart
is what lets the same router run against a mock on a laptop and real GGUFs on the
demo box without a code path changing.
"""

from __future__ import annotations

import threading

from .base import Backend, BackendError, ModelNotLoaded

_backend: Backend | None = None
_lock = threading.Lock()


def get_backend() -> Backend:
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                from ...config import get_settings

                st = get_settings()
                if st.backend == "mock":
                    from .mock import MockBackend

                    _backend = MockBackend()
                elif st.backend == "llamacpp":
                    from .llamacpp import LlamaCppBackend

                    _backend = LlamaCppBackend(model_dir=st.model_dir)
                else:
                    raise ValueError(
                        f"unknown backend {st.backend!r}; expected 'mock' or 'llamacpp'"
                    )
    return _backend


def reset_backend() -> None:
    """Tests only."""
    global _backend
    if _backend is not None:
        _backend.shutdown()
    _backend = None


__all__ = ["Backend", "BackendError", "ModelNotLoaded", "get_backend", "reset_backend"]
