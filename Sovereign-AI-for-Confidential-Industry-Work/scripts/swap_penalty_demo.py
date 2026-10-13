"""Show the router changing its mind because a model is already resident.

    python scripts/swap_penalty_demo.py

This is the decision no cloud router ever has to make, and it is the clearest
30 seconds you have for explaining why the router is VRAM-aware rather than a
quality lookup table.

The point is not that the resident model always wins - it is that the swap cost
is *priced in*, so a small quality gap is outweighed by a 4-second reload and a
large one is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

from kavach.router import get_router
from kavach.router.scoring import explain_margin, rank
from kavach.runtime import get_warden
from kavach.types import RouteRequest

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

PROMPT = "Summarise this inspection report and list the findings"


def show(router, warden, heading: str) -> str | None:
    decision, _ = router.decide(RouteRequest(prompt=PROMPT))
    print(f"\n{BOLD}{heading}{RESET}")
    print(DIM + "-" * len(heading) + RESET)

    for cand in rank(decision.candidates):
        state = f"{GREEN}RESIDENT{RESET}" if cand.resident else f"{DIM}cold    {RESET}"
        mark = f"{GREEN} <-{RESET}" if cand.model_id == decision.chosen else ""
        print(
            f"  {cand.model_id:<24} score {cand.score:+.3f}   "
            f"quality {cand.terms['_quality_raw']:.2f}   "
            f"swap {cand.terms['swap']:+.3f}   {state}{mark}"
        )

    margin = explain_margin(decision.candidates)
    if margin is not None:
        colour = YELLOW if margin < 0.05 else DIM
        print(f"  {colour}margin {margin:.4f}{RESET}")
    print(f"  {DIM}{decision.reason}{RESET}")
    return decision.chosen


def main() -> int:
    router, warden = get_router(), get_warden()
    warden.evict_all()

    first = show(router, warden, "1. Nothing resident - pure quality decision")

    print(f"\n{DIM}   ... a multimodal step now loads the vision model ...{RESET}")
    warden.admit("qwen3-vl-8b-q4")

    second = show(
        router, warden,
        "2. Vision model resident - its 4.2 s reload advantage is now priced in",
    )

    print()
    if first == second:
        print(
            f"{BOLD}The better model still won{RESET} - a 0.15 quality gap is more "
            f"than a reload is worth."
        )
        print(
            f"{DIM}But look at the margin: it collapsed to near zero. The router "
            f"reports that as a close call rather than presenting a coin-flip as a "
            f"confident choice.{RESET}"
        )
    else:
        print(
            f"{BOLD}The router kept the resident model{RESET} - the quality gap was "
            f"narrow enough that a 4.2 s reload was not worth paying."
        )

    print(
        f"\n{DIM}Tune this with w_swap in config/router.yaml. Higher keeps resident "
        f"models longer;\nlower prefers quality and reloads more. 0.20 flips the "
        f"decision at roughly a 0.2 quality gap.{RESET}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
