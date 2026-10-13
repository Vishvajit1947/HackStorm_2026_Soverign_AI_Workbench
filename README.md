Written for: anyone opening the GitHub repo, including mentors, judges and developers setting it up.

````markdown
# Kavach Sovereign Workbench

**An offline, multi-model AI workbench for confidential industrial documents. It runs on one mid-range GPU and proves it never talks to anything outside the plant.**

Kavach lets engineers search plant documents (SOPs, inspection reports, scanned P&IDs, handwritten notes), ask questions with verifiable citations, run calculations, and produce real Word, Excel and PowerPoint deliverables. Every model, tool and file stays on the local machine. Every action is permission-checked and written to a tamper-evident audit log.

> Built for the MRPL (Mangalore Refinery and Petrochemicals Ltd.) use case, where documents are confidential and cannot be sent to cloud AI services.

---

## Why Kavach

| Problem | What Kavach does |
|---|---|
| Confidential documents cannot leave the site | All models run locally via llama.cpp. No cloud calls, no CDN, no web fonts |
| "Trust us, it's offline" is not evidence | An egress sentinel, a canary probe, a firewall/nftables check and a code sandbox with no network interface |
| LLMs make things up | Answers must cite a passage that was actually retrieved. Numbers in generated documents are checked against the sources |
| Different users may see different data | Four classification labels and four roles. Restricted text is withheld and the system says so |
| AI actions must be auditable | Every model call, tool call and decision goes into a SHA-256 hash-chained audit log |
| One small GPU, several models | A VRAM-aware router and a warden that swaps models on a 6 GB card |

---

## Features

- **Grounded document Q&A**
  - Hybrid search: BM25 keyword search plus dense vectors, fused with Reciprocal Rank Fusion.
  - Every answer cites `[document p.N s.M]`, and unresolvable citations are rejected.
  - If the documents do not contain the answer, it declines instead of guessing.
- **Reads anything:** native PDF text, Tesseract OCR for scans, P&IDs and handwriting, and an optional vision-language model. Unreadable pages are reported, never silently indexed blank.
- **Agentic tasks**
  - Plan, execute, check, replan.
  - The plan is generated under a grammar constraint, every step has an acceptance test, and budgets are hard limits (12 steps, 20 tool calls, 480 s).
  - Approval gates survive a restart.
- **Real deliverables**
  - Every finished task leaves a downloadable `.docx`, `.xlsx` or `.pptx`.
  - The model writes the document from what was found; invented numbers are dropped or flagged.
- **Multi-model router**
  - Hard constraints, then task classification, then a utility score (quality, confidence, latency, swap cost, VRAM), then verification with at most one escalation.
  - Each decision is explained on screen, including close calls.
- **Sandboxed code execution:** Docker with `--network=none`, a seccomp profile, a read-only root filesystem and a non-root user. Without Docker it refuses rather than running code on the host.
- **Classification policy:** PUBLIC / INTERNAL / CONFIDENTIAL / RESTRICTED labels, and operator / engineer / auditor / admin roles. A run inherits the highest label of anything it has read.
- **Tamper-evident audit log:** the hash chain names the first broken link if any record is edited, deleted or spliced.
- **Model provenance:** every weight file is SHA-256-verified at boot, and a substituted file stops the system from starting.
- **Offline web UI:** plain HTML/CSS/JS, served by the gateway itself, with no build step and a strict Content-Security-Policy.

---

## Architecture

| Layer | Name | Responsibility |
|---|---|---|
| L0 | Sovereignty | Egress sentinel, canary, enforcement detection, policy engine, audit chain |
| L1 | Model fabric | Model registry, digest verification, VRAM warden, llama.cpp backend |
| L2 | Router | 4-stage cascade with an explainable score |
| L3 | Knowledge | Ingest (native / OCR / VLM), page-exact chunking, hybrid retrieval, citation checking |
| L4 | Tools | Tool manifests, workspace jail, `calc`, `fs.*`, `kb.search`, `doc.write`, `code.exec`, `engineering.analyze` |
| L5 | Agent | Planner, critic, loop, budgets, approvals, resumable runs |
| L6 | UI | Web panel with live run streaming (SSE) |

```
Request → Policy check → Planner (grammar-constrained) → Router → VRAM warden → llama.cpp
                                   ↓
              Tool executor: policy → workspace jail → audit → run
                                   ↓
                 Critic (acceptance test) → next step / replan
                                   ↓
                     Downloadable file + audit record
```

---

## Requirements

| Component | Version / notes |
|---|---|
| OS | Windows 10/11 (tested), or Linux |
| GPU | NVIDIA, 6 GB VRAM or more (tested on an RTX 3050 6 GB laptop). A CPU-only profile exists but is slow |
| Python | 3.11 or later |
| llama.cpp | CUDA build, e.g. `llama-b11195-bin-win-cuda-12.4-x64.zip` plus the matching `cudart` zip |
| Tesseract OCR | 5.x, for scanned pages |
| Docker Desktop | Optional. Needed only for code execution (WSL2 on Windows) |
| Disk | About 17 GB for model weights |

---

## Installation

### 1. Python environment

```powershell
git clone https://github.com/Vishvajit1947/Sovereign-AI-for-Confidential-Industry-Work.git
cd Sovereign-AI-for-Confidential-Industry-Work
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,knowledge,deliverables]"
.\.venv\Scripts\python.exe -m pip install opencv-python-headless
```

### 2. Model weights

Download these into one folder (for example `E:\kavach_models`). Use **exactly** these files: their SHA-256 digests are pinned in `config/models.yaml`.

| Registry ID | File to save as | Source (Hugging Face) |
|---|---|---|
| `qwen2_5-1_5b-q4` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| `qwen2_5-7b-q4` / `qwen2_5-7b-instruct` | `qwen2.5-7b-instruct-q4_k_m.gguf` | `QuantFactory/Qwen2.5-7B-Instruct-GGUF` → `Qwen2.5-7B-Instruct.Q4_K_M.gguf` |
| `qwen2_5-coder-7b-q4` | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | `QuantFactory/Qwen2.5-Coder-7B-Instruct-GGUF` → `Qwen2.5-Coder-7B-Instruct.Q4_K_M.gguf` |
| `qwen2_5-vl-7b-q4` | `Qwen2.5-VL-7B-Instruct-q4_k_m.gguf` | `Mungert/Qwen2.5-VL-7B-Instruct-GGUF` |
| (vision projector) | `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf` | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` |
| `minilm-l6-emb` | `all-minilm-l6-v2-q8_0.gguf` | `leliuga/all-MiniLM-L6-v2-GGUF` → `all-MiniLM-L6-v2.Q8_0.gguf` |
| `bge-m3-emb` | `bge-m3-q8_0.gguf` | `cstr/bge-m3-GGUF` |

Then verify them:

```powershell
.\.venv\Scripts\kavach.exe models verify     # every staged model should read VERIFIED
```

### 3. llama.cpp and Tesseract

Unzip llama.cpp (both zips) into one folder, for example `E:\kavach_tools\llamacpp`. Install Tesseract anywhere.

### 4. Machine settings: `kavach.env`

Create `kavach.env` in the project root. It is gitignored, because it describes one machine.

```ini
KAVACH_PROFILE=MIDRANGE
KAVACH_BACKEND=llamacpp
KAVACH_MODEL_DIR=E:/kavach_models
KAVACH_LLAMA_BIN_DIR=E:/kavach_tools/llamacpp
KAVACH_TESSERACT=C:/Program Files/Tesseract-OCR/tesseract.exe
KAVACH_ENFORCE_DIGESTS=1
KAVACH_PORT=8420
```

### 5. Code sandbox (optional)

```powershell
docker compose build sandbox-runner
```

---

## Running

1. Start **Docker Desktop** (optional, for code execution).
2. Double-click **`start.bat`**. It runs a pre-flight check, starts the gateway and opens the browser when it is ready.
3. Open **http://127.0.0.1:8420** (API docs at `/docs`).

Load the demo documents once:

```powershell
.\.venv\Scripts\kavach.exe kb ingest MRPL_DEMO_CORPUS
```

To stop Kavach, close the `start.bat` window.

### Hardware profiles

Set `KAVACH_PROFILE` in `kavach.env`. No code change is needed.

| Profile | Target | Models |
|---|---|---|
| `TINY` | CPU only | 1.5B + embedder |
| `FAST` | 6 GB GPU, fastest | 1.5B + embedder, both resident |
| `MIDRANGE` | 6 GB GPU (recommended) | 1.5B, 7B instruct, 7B coder, 7B vision, embedder (swapped as needed) |
| `MID` | 12 GB GPU | 7B general, coder, vision |
| `HIGH` | 16 GB+ GPU | 14B + reranker |

---

## Verifying the claims yourself

| Claim | Command |
|---|---|
| Audit log is tamper-evident | `python scripts/tamper_demo.py` |
| A substituted model file is refused | `python scripts/provenance_demo.py` |
| Generated code cannot reach the network | `python scripts/sandbox_demo.py` |
| Router prices in model-swap cost | `python scripts/swap_penalty_demo.py` |
| Hybrid retrieval, citations, clearance | `python scripts/knowledge_demo.py` |
| UI makes no external requests | `python scripts/offline_check.py` |
| What this box did and did NOT prove | `python scripts/run_sheet.py --out var/run_sheet.md` |
| Real per-application egress block (Windows, admin) | `.\deploy\windows-firewall.ps1 -Arm`, then **Fire the canary** in the UI |

Run all of the above with `.\.venv\Scripts\python.exe`.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current result: **1256 passed, 3 skipped.** Many tests deliberately try to break things: workspace escapes, tampered audit logs, exhausted budgets, forged citations.

---

## Project structure

```
config/        models.yaml · profiles.yaml · policy.yaml · router.yaml · knowledge.yaml
services/kavach/
  sovereignty/   sentinel, canary, enforcement, policy engine
  audit/         hash-chained audit log + console projection
  runtime/       model registry, VRAM warden, llama.cpp / mock backends
  router/        constraints, classifier, scoring, verifier, cascade
  knowledge/     extract (native/OCR/VLM), chunk, index, retrieval, citations, answer
  tools/         manifests, executor, workspace jail, GBNF grammars, implementations
  agent/         planner, critic, loop, composer, broker, run state
  api/           FastAPI gateway + SSE run stream
  ui/            index.html · app.css · app.js · charts.js
deploy/        Dockerfiles · seccomp profile · nftables.conf · windows-firewall.ps1
scripts/       demos, eval harness, model staging, run sheet, pre-flight
tests/         pytest suite
docs/          DEMO_SCRIPT.md · DESIGN.md · project overview
MRPL_DEMO_CORPUS/  sample refinery documents (native, scanned, P&IDs, handwritten)
```

---

## Known limitations

We list these up front rather than leave them to be discovered.

- **Router quality scores are mostly hand-seeded.** The eval harness saturated on an easy test set; a harder evaluation set is needed (`scripts/eval_harness.py --quality --write`).
- **Egress enforcement on Windows needs the firewall rule armed** (admin). The packet sentinel counts the whole PC, so without the rule the canary honestly reports *inconclusive*. Full enforcement is shown in the Linux compose stack with nftables.
- **The vision model is slow on a 6 GB card,** so OCR is preferred for scans and P&IDs.
- **A 7B model can still misread or misinfer.** Citations, the number check and human approval exist for this reason. Review generated documents before use.
- **Single-site, single-GPU design.** Request queuing for many concurrent users is future work.

---

## Further documentation

- [`INSTALL.md`](INSTALL.md): detailed installation notes
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md): a 12-minute live demonstration script
- [`docs/DESIGN.md`](docs/DESIGN.md): UI design reference
````

Notes before you paste:
- **The current README is 1,369 lines of phase-by-phase test guides.** This version replaces all of that. If you want to keep the old one, save it as `docs/PHASES.md` first and add a link to it under "Further documentation".
- **OpenCV is installed as a separate step.** The code needs it, but `pyproject.toml` doesn't list it yet, so the install section adds a second `pip install` line. Add it to `pyproject.toml` and that line can go.
- **The test count (1256 passed)** is from the current, uncommitted code. It's only accurate once those changes are committed.
