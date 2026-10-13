"""Offline evaluation harness (§04, "Building quality[m][class]").

This is the script that converts "we picked models by vibes" into "here is our
evaluation table" - the difference between a demo and a submission.

Two independent modes, because they have very different requirements:

    --routing   Classifier accuracy over the held-out set in eval_tasks.yaml,
                printed as a confusion matrix. Needs only the embedder, so it
                RUNS TODAY on the mock backend and is worth running constantly
                while you add prototypes.

    --quality   Measures quality[model][class] by generating real answers and
                checking them programmatically. Needs real weights. It REFUSES
                to run on the mock backend, because stamping `quality_measured:
                true` onto numbers produced by a stub would be worse than the
                honest seeds it replaces - it would launder a guess into a
                measurement and put an indefensible number on the judges' screen.

    python scripts/eval_harness.py --routing
    python scripts/eval_harness.py --routing --confusion
    python scripts/eval_harness.py --quality --write     # real backend only
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

import yaml

from kavach.config import get_settings
from kavach.types import ALL_TASK_CLASSES, GenRequest

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


# ---------------------------------------------------------------- checks

def python_parses_and_defines(text: str, expect: Any, **_: Any) -> float:
    code = _strip_fence(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return 1.0 if expect in names else 0.3      # parses but wrong name


def python_runs_assertion(text: str, expect: Any, **_: Any) -> float:
    """Execute the snippet and evaluate an assertion against it.

    Runs in-process with empty globals. That is acceptable HERE because this is
    an offline harness you run yourself on your own eval set - it is not the
    agent's code path. Model-authored code at runtime goes to code.exec, which
    is a container with no network interface at all (Phase 4).
    """
    code = _strip_fence(text)
    scope: dict[str, Any] = {}
    try:
        exec(compile(code, "<eval>", "exec"), scope)  # noqa: S102 - see docstring
        return 1.0 if eval(expect, scope) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def contains_number_near(text: str, expect: Any, tolerance: float = 0.1, **_: Any) -> float:
    numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", text.replace(",", ""))]
    return 1.0 if any(abs(n - float(expect)) <= tolerance for n in numbers) else 0.0


def mentions_all(text: str, expect: Any, **_: Any) -> float:
    lowered = text.lower()
    hits = sum(1 for term in expect if str(term).lower() in lowered)
    return hits / len(expect) if expect else 0.0


def has_citation_and_mentions(text: str, expect: Any, **_: Any) -> float:
    cited = bool(re.search(r"\[[^\]]+\]", text))
    content = mentions_all(text, expect)
    if not cited:
        return content * 0.3        # an uncited answer is heavily penalised
    return content


CHECKS = {
    "python_parses_and_defines": python_parses_and_defines,
    "python_runs_assertion": python_runs_assertion,
    "contains_number_near": contains_number_near,
    "mentions_all": mentions_all,
    "has_citation_and_mentions": has_citation_and_mentions,
}


def _strip_fence(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else text


# ---------------------------------------------------------------- routing

def run_routing(tasks: dict[str, Any], show_confusion: bool) -> int:
    from kavach.router import get_router

    router = get_router()
    classifier = router.classifier

    cases = tasks.get("routing", {})
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    wrong: list[tuple[str, str, str, float]] = []
    total = correct = 0
    confidences: list[float] = []

    for expected, prompts in cases.items():
        for prompt in prompts or []:
            result = classifier.classify(prompt)
            got = result.task_class.value
            matrix[expected][got] += 1
            confidences.append(result.confidence)
            total += 1
            if got == expected:
                correct += 1
            else:
                wrong.append((prompt, expected, got, result.confidence))

    if total == 0:
        print(f"{YELLOW}no routing cases in eval_tasks.yaml{RESET}")
        return 1

    accuracy = correct / total
    mean_conf = sum(confidences) / len(confidences)

    from kavach.runtime import get_backend

    simulated = get_backend().health().get("simulated", False)
    method = classifier.classify("probe").method

    print(f"{BOLD}Routing accuracy on the held-out set{RESET}")
    print(f"{DIM}  method: {method} · {classifier.prototype_count} prototypes{RESET}")
    if simulated:
        print(
            f"{YELLOW}  measured with the MOCK embedder (character n-grams), not "
            f"bge-m3.{RESET}"
        )
        print(
            f"{DIM}  Treat this as a smoke test of the prototype set, not the "
            f"production number.\n"
            f"  Re-run on the real embedder before quoting it anywhere.{RESET}"
        )
    print()

    for expected in ALL_TASK_CLASSES:
        row = matrix.get(expected, {})
        n = sum(row.values())
        if n == 0:
            continue
        hit = row.get(expected, 0)
        colour = GREEN if hit == n else (YELLOW if hit >= n * 0.7 else RED)
        print(f"  {expected:<20} {colour}{hit}/{n}{RESET}")

    print()
    colour = GREEN if accuracy >= 0.9 else (YELLOW if accuracy >= 0.75 else RED)
    print(f"  overall  {colour}{correct}/{total} = {accuracy:.1%}{RESET}")
    print(f"  mean confidence  {mean_conf:.2f}")

    if wrong:
        print(f"\n{BOLD}Misroutes - paste these into config/prototypes.yaml{RESET}")
        for prompt, expected, got, conf in wrong:
            print(f"  {RED}{got:<18}{RESET} should be {GREEN}{expected:<18}{RESET} "
                  f"p={conf:.2f}")
            print(f"    {DIM}{prompt}{RESET}")
        print(f"\n{DIM}  Remember to DELETE them from eval_tasks.yaml when you do, or"
              f" the next run measures memorisation.{RESET}")

    if show_confusion:
        print(f"\n{BOLD}Confusion matrix{RESET} {DIM}(rows = true, cols = predicted){RESET}")
        short = [c[:6] for c in ALL_TASK_CLASSES]
        print("  " + " " * 20 + " ".join(f"{s:>7}" for s in short))
        for expected in ALL_TASK_CLASSES:
            row = matrix.get(expected, {})
            if not sum(row.values()):
                continue
            cells = []
            for predicted in ALL_TASK_CLASSES:
                n = row.get(predicted, 0)
                if n and predicted == expected:
                    cells.append(f"{GREEN}{n:>7}{RESET}")
                elif n:
                    cells.append(f"{RED}{n:>7}{RESET}")
                else:
                    cells.append(f"{DIM}{'.':>7}{RESET}")
            print(f"  {expected:<20}" + " ".join(cells))

    return 0 if accuracy >= 0.75 else 1


# ---------------------------------------------------------------- quality

def run_quality(tasks: dict[str, Any], write: bool) -> int:
    from kavach.runtime import get_backend, get_registry, get_warden

    settings = get_settings()
    backend = get_backend()

    if backend.health().get("simulated"):
        print(f"{RED}REFUSING TO RUN{RESET}")
        print(
            "  The mock backend produces synthetic text. Scoring it would yield\n"
            "  meaningless numbers, and --write would stamp `quality_measured: true`\n"
            "  onto them - laundering a guess into a measurement and putting an\n"
            "  indefensible figure on the router panel in front of judges.\n"
        )
        print(f"{DIM}  Stage real weights, then:{RESET}")
        print(f"{DIM}    python scripts/stage_model.py --write{RESET}")
        print(f"{DIM}    KAVACH_BACKEND=llamacpp python scripts/eval_harness.py "
              f"--quality --write{RESET}")
        return 2

    registry = get_registry()
    warden = get_warden()
    cases = tasks.get("quality", {})
    models = [s for s in registry.routable(warden.profile.available)]

    if not models:
        print(f"{RED}no routable models{RESET}")
        return 1

    results: dict[str, dict[str, float]] = defaultdict(dict)

    for spec in models:
        print(f"\n{BOLD}{spec.id}{RESET}")
        warden.admit(spec.id)
        for task_class, items in cases.items():
            scores = []
            for item in items or []:
                check = CHECKS.get(item["check"])
                if check is None:
                    print(f"  {RED}unknown check {item['check']!r}{RESET}")
                    continue
                out = backend.generate(
                    spec, GenRequest(prompt=item["prompt"], max_tokens=512)
                )
                score = check(
                    out.text,
                    item.get("expect"),
                    tolerance=item.get("tolerance", 0.1),
                )
                scores.append(score)
            if scores:
                mean = sum(scores) / len(scores)
                results[spec.id][task_class] = round(mean, 2)
                colour = GREEN if mean >= 0.7 else (YELLOW if mean >= 0.4 else RED)
                print(f"  {task_class:<20} {colour}{mean:.2f}{RESET} "
                      f"{DIM}({len(scores)} tasks){RESET}")

    if not write:
        print(f"\n{DIM}re-run with --write to record these into models.yaml{RESET}")
        return 0

    _write_quality(settings.config_dir / "models.yaml", results)
    print(f"\n{GREEN}models.yaml updated{RESET} - quality_measured is now true for "
          f"{len(results)} models")
    print(f"{DIM}review the diff, then the router panel can defend its numbers{RESET}")
    return 0


def _write_quality(manifest_path: Path, results: dict[str, dict[str, float]]) -> None:
    """Surgical edit, like stage_model.py - a safe_dump round-trip would delete
    every comment in models.yaml, including the one warning about seeds."""
    NL = "\n"
    lines = manifest_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    current: str | None = None
    in_quality = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("- id:"):
            current = stripped.split("id:", 1)[1].strip()
            in_quality = False

        if current in results:
            if stripped.startswith("quality:"):
                in_quality = True
                out.append(line)
                for cls, value in sorted(results[current].items()):
                    out.append(f"    {cls}: {value}\n")
                continue
            if in_quality:
                if line.startswith("    ") and ":" in stripped:
                    # A class this run did not measure KEEPS its seeded value,
                    # carried through with a marker that says so. Dropping it
                    # would delete visual_doc from a text model - the explicit
                    # 0.00 that is the only reason stage 1 eliminates it for a
                    # scan - and agentic_multi, which the planner routes on. A
                    # measurement run must never leave the manifest saying less
                    # than it did before the run.
                    cls = stripped.split(":", 1)[0].strip()
                    if cls in results[current]:
                        continue          # already re-emitted from results
                    out.append(line.rstrip() + "   # SEEDED - not measured" + NL)
                    continue
                in_quality = False
            if stripped.startswith("quality_measured:"):
                continue

        out.append(line)

    # Add the flag to every model we measured.
    def _flag(model_id: str) -> str:
        # Never a bare `true`. A run that measured 4 of 7 classes and stamped
        # quality_measured: true would be laundering the 3 it never touched -
        # the same sin this harness refuses the mock backend for, arriving by
        # the other door. Naming the classes makes the claim checkable.
        measured = ", ".join(sorted(results[model_id]))
        return f"  quality_measured: [{measured}]\n"

    final: list[str] = []
    current = None
    for line in out:
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current in results:
                final.append(_flag(current))
            current = stripped.split("id:", 1)[1].strip()
        final.append(line)
    if current in results:
        final.append(_flag(current))

    manifest_path.write_text("".join(final), encoding="utf-8")


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="Kavach offline evaluation harness")
    parser.add_argument("--routing", action="store_true",
                        help="classifier accuracy (works on the mock backend)")
    parser.add_argument("--quality", action="store_true",
                        help="measure quality[model][class] (needs real weights)")
    parser.add_argument("--confusion", action="store_true", help="print the matrix")
    parser.add_argument("--write", action="store_true",
                        help="record measured quality into models.yaml")
    args = parser.parse_args()

    if not args.routing and not args.quality:
        args.routing = True

    settings = get_settings()
    path = settings.config_dir / "eval_tasks.yaml"
    tasks = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    code = 0
    if args.routing:
        code |= run_routing(tasks, args.confusion)
    if args.quality:
        code |= run_quality(tasks, args.write)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
