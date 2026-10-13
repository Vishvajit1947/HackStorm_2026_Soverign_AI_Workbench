"""The backend seam.

Deliberately small. Everything the agent loop needs from a model is here, and
nothing about *scheduling* is - that belongs to the warden.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...types import GenRequest, GenResult
from ..registry import ModelSpec


class BackendError(RuntimeError):
    """Generation or loading failed in a way the caller should see."""


class ModelNotLoaded(BackendError):
    """The warden admitted a model the backend does not actually have resident.

    This should be impossible. If it happens, the residency table and reality have
    diverged, which is worth a loud failure rather than a silent reload.
    """


class Backend(ABC):
    name: str = "base"

    # ---------------------------------------------------------------- lifecycle

    @abstractmethod
    def load(self, spec: ModelSpec) -> None:
        """Make the model ready to serve. Called by the warden, holding its lock."""

    @abstractmethod
    def unload(self, spec: ModelSpec) -> None:
        """Release the model's VRAM. Must be safe to call on a model not loaded."""

    def shutdown(self) -> None:
        """Release everything. Called at process exit."""

    # ---------------------------------------------------------------- inference

    @abstractmethod
    def generate(self, spec: ModelSpec, req: GenRequest) -> GenResult:
        """One completion. `req.grammar` is GBNF and MUST be honoured when set -
        it is what stops a 7B model emitting malformed tool JSON (§05)."""

    @abstractmethod
    def embed(self, spec: ModelSpec, texts: list[str]) -> list[list[float]]:
        """Embeddings for the router's classifier and the knowledge plane."""

    # ---------------------------------------------------------------- health

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """What this backend can currently do, for the dashboard."""

    def loaded_models(self) -> list[str]:
        return []
