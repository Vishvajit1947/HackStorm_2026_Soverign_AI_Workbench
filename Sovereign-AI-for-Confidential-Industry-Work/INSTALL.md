# Installing Kavach

Everything below was performed and verified on a Windows 11 box with an RTX 3050
(6 GB) on 2026-09-10. Where a step has a caveat, the caveat is in the step.

`README.md` is the reference for what each phase does and how to test it. This
file is only how to get a working box.

---

## 1. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Python 3.11+. `pyproject.toml` is the single source of truth for dependencies -
there is deliberately no `requirements.txt`, because two dependency lists drift
apart and the one you are not looking at is always the stale one.

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `.[dev]` | pytest, ruff, PyMuPDF, rank-bm25 | running the suite |
| `.[deliverables]` | python-docx, openpyxl, python-pptx | `doc.write` (Word/Excel/PowerPoint) |
| `.[knowledge]` | PyMuPDF, rank-bm25 | production without the dev tools |

---

## 2. Model weights

Weights are **not** in this repo. Put the GGUFs anywhere and point
`KAVACH_MODEL_DIR` at them - on this box they live in `D:\kavach_models`.

The registry (`config/models.yaml`) names the file each model id expects. Record
the digests once:

```powershell
$env:KAVACH_MODEL_DIR = "D:\kavach_models"
.\.venv\Scripts\python.exe scripts\stage_model.py            # report only
.\.venv\Scripts\python.exe scripts\stage_model.py --write    # record digests
.\.venv\Scripts\kavach.exe models verify
```

`stage_model.py` refuses to overwrite a digest that already exists and now
disagrees - that is a corrupted download or a substituted file, and it needs a
human rather than a flag. `--force` only when you have decided the new file is
the approved one.

A model is gated **twice**: it must exist in `config/models.yaml`, and its id
must appear in the active profile's `available:` list in `config/profiles.yaml`.
Adding it to one and not the other means it is never a candidate, with no error
anywhere.

---

## 3. llama.cpp

The warden spawns one `llama-server` per resident model and kills it on
eviction, so the binary must be on `PATH`:

```powershell
$env:PATH = "D:\llamacpp;" + $env:PATH
llama-server --version
```

Get a build from the llama.cpp releases (`llama-<build>-bin-win-cuda-12.4-x64.zip`
plus the matching `cudart-*` zip for the runtime DLLs). Do **not** unpack it into
this repo - it is ~45 MB of build output that is not this project's source, and
it was previously committed here by accident.

> Known incompatibility: `bge-m3` GGUFs tagged with architecture `xlmr` are
> refused by every mainline llama.cpp build tested (up to b10809), so the MID
> profile pins `minilm-l6-emb` instead. See the note in `config/profiles.yaml`.

---

## 4. OCR (scanned pages, P&IDs, handwriting)

```powershell
$env:PATH = "C:\Program Files\Tesseract-OCR;" + $env:PATH
tesseract --version
```

Without tesseract on `PATH` the ingest fork falls through to `UNREADABLE` and
reports the pages as gaps - it does not silently index empty pages. With it, all
eleven MRPL demo documents read, including the two P&IDs and the handwritten
annotation.

---

## 5. The sandbox (`code.exec`)

```powershell
docker compose build sandbox-runner
.\.venv\Scripts\kavach.exe tools sandbox
```

Without Docker, `code.exec` **refuses to run** rather than executing
model-authored code on the host. The dev fallback
(`KAVACH_ALLOW_UNSANDBOXED_EXEC=1`) exists, is off by default, and stamps every
result `sandboxed: false`. Never set it on a demo box.

---

## 6. Run it

```powershell
$env:KAVACH_PROFILE   = "FAST"        # or MID
$env:KAVACH_BACKEND   = "llamacpp"
$env:KAVACH_MODEL_DIR = "D:\kavach_models"
$env:PATH = "D:\llamacpp;C:\Program Files\Tesseract-OCR;" + $env:PATH
.\.venv\Scripts\python.exe -m uvicorn kavach.api.app:app --host 127.0.0.1 --port 8420
```

Panel at <http://127.0.0.1:8420/>, API docs at `/docs`.

> These are read from the process environment, not from `.env` - nothing in the
> code loads a dotenv file. `.env` exists for docker-compose.
>
> Port 8080 is reserved by Windows on some machines (`WinError 10013`); 8420 works.

### Which profile

| Profile | Models | Use |
|---|---|---|
| `FAST` | qwen3-1_7b-q4 + minilm | Live demos. Both models stay resident, nothing swaps. A search -> Word-deliverable run measured **34 s** end to end. |
| `MID` | qwen3-8b, coder-7b, vl-8b | Better prose and planning; a multi-step run takes 2.5-4 minutes on a 6 GB card. |
| `HIGH` | 14B + reranker | A real datacentre GPU. |

Switching is one environment variable and no code change - which is the same
property that lets a better model be dropped in later.

---

## 7. Verify the install

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 750 passed
.\.venv\Scripts\python.exe scripts\offline_check.py
.\.venv\Scripts\kavach.exe doctor
.\.venv\Scripts\kavach.exe kb doctor
```

`offline_check.py` is the one that matters for the sovereignty claim: it fails
the build if any UI asset carries an absolute URL.

---

## What this box still cannot do

Honest gaps, not a roadmap:

- **Egress enforcement is unprovable on Windows.** The sentinel reads
  `/proc/net/dev`, which does not exist here, so it reports `DEGRADED` rather
  than a reassuring zero. Inside the Linux compose stack the same code reports
  `mode: proc` and the canary returns `BLOCKED`/`ENETUNREACH`:
  `docker compose up -d gateway`, then `docker exec kavach-gateway-1 kavach canary`.
- **Vision on drawings.** No `mmproj` projector is staged, and the llamacpp
  backend refuses image input rather than answering a visual question from the
  text alone. OCR covers scanned text, P&IDs and handwriting.
- **Router quality scores are hand-seeded.** A measured run saturated at 1.00
  across nearly every class, which made the router rank on speed instead of
  quality - so the seeds were restored. Closing this needs a harder eval set,
  not another measurement run.
