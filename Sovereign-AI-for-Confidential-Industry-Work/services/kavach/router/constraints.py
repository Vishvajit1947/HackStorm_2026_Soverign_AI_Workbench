"""Stage 1 - the constraint filter (§04).

Removes impossible candidates rather than scoring them, so the learned stage only
ever chooses among models that can actually do the job. A model with no vision
tower is not "a bad choice" for a scanned page - it is not a choice at all, and
scoring it is a category error that lets a high quality number win an argument it
should never have been in.

Every elimination records WHY, and eliminated candidates stay in the list so the
explainer panel can show "6 -> 3" with reasons rather than a silent shortlist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import Candidate, RouteRequest

if TYPE_CHECKING:
    from ..runtime.registry import ModelSpec
    from ..runtime.warden import VramWarden


def apply_constraints(
    specs: list[ModelSpec],
    req: RouteRequest,
    warden: VramWarden | None = None,
    max_tier: int | None = None,
) -> list[Candidate]:
    """Build the candidate list, marking each survivor or its cause of death.

    Order matters only for the reason string: the first failing check is the one
    reported, and they are ordered cheapest-and-most-fundamental first so the
    message is the most informative one.
    """
    candidates: list[Candidate] = []

    for spec in specs:
        resident = warden.is_resident(spec.id) if warden else False
        candidate = Candidate(
            model_id=spec.id,
            tier=spec.tier,
            resident=resident,
            quality=0.0,          # filled in by stage 3 once the class is known
        )

        candidate.eliminated_by = _first_failure(spec, req, warden, max_tier)
        candidates.append(candidate)

    return candidates


def _first_failure(
    spec: ModelSpec,
    req: RouteRequest,
    warden: VramWarden | None,
    max_tier: int | None,
) -> str | None:
    # A model whose weights we do not trust can never serve a request, whatever
    # its quality numbers say.
    if not spec.routable:
        if spec.role != "generation":
            return f"not a generation model (role={spec.role})"
        return f"weights are {spec.provenance.value}, not approved for serving"

    if req.has_images and not spec.has_vision:
        return "images attached, no vision tower"

    if req.needs_tools and not spec.has_tools:
        return "tool calling required, not supported"

    if req.min_context > spec.context:
        return f"needs {req.min_context} ctx, has {spec.context}"

    # RESTRICTED data can be pinned away from experimental tiers.
    if max_tier is not None and spec.tier > max_tier:
        return f"tier {spec.tier} above the {req.label.value} ceiling of {max_tier}"

    # Could this model ever be admitted on this profile, with everything
    # evictable evicted? If not, routing to it would only produce an OOM.
    if warden is not None and not warden.fits(spec.id):
        return f"{spec.vram_gb} GB will not fit on this profile"

    return None


def survivors(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.survived]


def summarise(candidates: list[Candidate]) -> str:
    """The '6 -> 3' line above the arrow in the explainer panel."""
    return f"{len(candidates)} -> {len(survivors(candidates))}"
