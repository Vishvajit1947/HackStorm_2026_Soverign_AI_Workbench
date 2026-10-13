# Kavach Sovereign Workbench

Air-gapped, multi-model agentic AI workbench that runs on one mid-range GPU — and
proves it never talks to anything outside the plant.

Reference architecture: **KVC-ARCH-001 Rev A** · PS ID 26117 · Client MRPL.

---

## Build phases

Each phase ends in something you can test by hand. Nothing later depends on a
phase you have not signed off.

| # | Ships | Status |
|---|---|---|
| **1** | Foundation + **L0 sovereignty** — config, state store, audit hash chain, sentinel, canary, policy engine, CLI, compose network | ✅ **done** |
| **2** | **L1 model fabric** — registry, digest verification, VRAM warden, mock + llama.cpp backends, hot-reload | ✅ **done** |
| **3** | **L2 router** — 4-stage cascade, classifier, scoring, verifier escalation, explainer panel feed, eval harness | ✅ **done** |
| **4** | **L4 tools + sandbox** — manifest registry, `calc`, `fs.*`, `code.exec` on `--network=none`, workspace jail, GBNF | ✅ **done** |
| **5** | **L5 agent core** — plan/execute/observe/critic, budgets, approval gates, resumable runs | ✅ **done** |
| **6** | **L3 knowledge** — ingest fork, OCR/VLM path, hybrid retrieval, mandatory citations | ✅ **done** |
| **7** | **L6 UI + demo harness** — four live panels, SSE run stream, offline install, run sheet | ✅ **done** |

---

## Phase 1 — what actually exists

```
config/            profiles.yaml (TINY/LOW/MID/HIGH) · models.yaml · policy.yaml
services/kavach/
  config.py        profile + settings loading, all local files
  types.py         Label, SideEffect, TaskClass, Run/Step/Budget vocabulary
  store/           SQLite schema + access (runs, steps, decisions, audit, egress)
  audit/chain.py   the tamper-evident hash chain
  sovereignty/
    sentinel.py    reads the kernel tx counters + the nftables counter
    netcounters.py /proc/net/dev on Linux, GetIfTable2 on Windows
    canary.py      deliberate egress attempt, honest verdicts
    policy.py      classification -> capability engine
  runtime/
    registry.py    model manifests, digest verification, hot reload
    warden.py      VRAM admission control, LRU eviction, swap cost
    backends/      base · mock (deterministic, no weights) · llamacpp (spawns servers)
  router/
    constraints.py stage 1 - hard eliminations, with reasons kept
    classifier.py  stage 2 - embed + kNN, keyword fallback that says it degraded
    scoring.py     stage 3 - the utility score, every term normalised 0..1
    verifier.py    stage 5 - checks + bounded one-tier escalation
    cascade.py     the whole thing; decide() is side-effect free
  tools/
    workspace.py   the path jail - the only place containment is decided
    registry.py    manifest loader + schema validation
    executor.py    the chokepoint: policy -> jail -> audit -> run
    grammar.py     JSON schema -> GBNF, so malformed tool calls are unreachable
    manifests/     calc · code.exec · fs.read · fs.list · fs.write
    impl/          calc.py · fs.py · sandbox.py
  api/app.py       the gateway
  api/routes_models.py · routes_router.py · routes_tools.py
  cli.py           kavach doctor|serve|audit|canary|policy|models|route|tools
deploy/            nftables.conf · Dockerfile.gateway · Dockerfile.sandbox
                   seccomp-strict.json
scripts/           tamper_demo.py · provenance_demo.py · stage_model.py
                   eval_harness.py · swap_penalty_demo.py · sandbox_demo.py
  agent/
    planner.py     goal -> steps, validated against the real tool registry
    critic.py      acceptance tests: mechanical first, model second
    loop.py        plan -> execute -> observe -> critique -> replan
    state.py       run persistence, so a paused run survives a restart
  knowledge/
    extract.py     the ingest fork: native text layer vs a picture of text
    chunk.py       page-exact chunking, so a citation can be followed
    store.py       what was ingested, how each page was read, under which label
    index.py       BM25 + dense vectors, each knowing when it cannot be trusted
    retrieval.py   rank fusion, clearance filtering, coverage that travels along
    citations.py   every citation must resolve to a passage that was retrieved
    answer.py      retrieve -> route -> check, with three refusals
  ui/
    index.html     five views behind a sidebar, no build step
    app.css        design system; system fonts only - a web font is a request
    charts.js      inline SVG: donut, sparkline, stacked + diverging bars
    app.js         views, polling, SSE with a fallback that says it is polling
  agent/broker.py  runs the loop on a thread, fans events out, never blocks it
  api/routes_stream.py  the SSE endpoint: snapshot first, stream_end last
scripts/           ... offline_check.py · run_sheet.py · knowledge_demo.py
tests/             636 tests, most of them trying to break something
```

### The three claims Phase 1 makes

1. **The audit log is tamper-evident.** Append-only triggers stop casual edits; the
   hash chain catches an attacker who drops the triggers. Verification reports the
   **first** broken link, because everything after it is untrustworthy anyway.
2. **The policy engine governs tools it has never seen**, because it reasons over a
   declared side-effect class rather than a tool name.
3. **The monitor is honest about what it cannot prove.** On a box with no firewall
   the canary returns `INCONCLUSIVE`, not a green tick — a timeout and an unplugged
   cable look identical, and pretending otherwise is the one thing that would sink
   this in Q&A.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

> Python 3.11+. Phases 1-6 need no GPU, no weights and no Docker.
>
> `[dev]` pulls in PyMuPDF and rank-bm25 alongside pytest, because the Phase 6
> tests open real PDFs. A suite that skips when a dependency is missing is a
> coverage gap wearing a green tick. In production install `.[knowledge]`.

---

## How to test Phase 1

### 1. Run the Phase 1 tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_audit_chain.py tests\test_policy.py tests\test_api.py -q
```

**Expect:** `35 passed`. Most of these are attempts to break the audit chain —
edited payloads, deleted records, forged predecessor links.

> Running the whole suite (`pytest -q`) now gives **124 passed**, because Phase 2
> added 89 more.

### 2. See what this box can and cannot prove

```powershell
.\.venv\Scripts\kavach.exe doctor
```

**Expect:** profile `MID`, `10.0 GB usable of 12.0 GB`, and `sentinel HEALTHY`
with the counter source in brackets — `(proc)` on Linux, `(iphlpapi)` on
Windows. Both are real kernel counters, so the packet number is a measurement
on either box.

On Windows you will also see **`MEASURED, NOT ENFORCED`**. That is the honest
half that remains: this box can *count* what left it, but it has no firewall
drop counter, so `blocked` is Kavach declining to make a request rather than a
kernel refusing to carry one. Only `sentinel HEALTHY (nftables)` on the Linux
demo machine backs the enforcement claim.

### 3. Break the audit chain on purpose

```powershell
.\.venv\Scripts\python.exe scripts\tamper_demo.py
```

**Expect** five sections:

| # | What it does | Expected |
|---|---|---|
| 1 | Writes 5 records | `chain intact - 5 records` |
| 2 | Tries `UPDATE` | `refused - audit log is append-only` |
| 3 | Drops triggers, rewrites `exit_code 1 -> 0` | `CHAIN BROKEN at seq 4 ... content was altered` |
| 4 | Deletes seq 3 outright | `CHAIN BROKEN at seq 4 ... sequence gap` |
| 5 | Re-points seq 5 at a forged hash | `CHAIN BROKEN at seq 5 ... points at aaaa...` |

This is the script to run when a judge asks *"how do you know the log hasn't been
edited?"*

### 4. Fire the canary

```powershell
.\.venv\Scripts\kavach.exe canary
```

**Expect on Windows/dev:** both probes `INCONCLUSIVE`, exit code 1, and the message
*"a timeout is not proof"*. **This is the correct answer here** — there is no
firewall on your laptop, so nothing has been proven.

**Expect on the demo box** (after `sudo nft -f deploy/nftables.conf`): both probes
`BLOCKED` with `ENETUNREACH`/`EPERM`, exit code 0, and the blocked counter ticking.
That is the version you show judges.

> The canary attempts a DNS query and a TCP connect to `203.0.113.1` — an RFC 5737
> documentation address that is never routed. It is a self-test of our own firewall.

### 5. Ask the policy engine questions

```powershell
# The visible refusal - even an admin cannot write outside a RESTRICTED workspace
.\.venv\Scripts\kavach.exe policy check fs.write --side-effect WRITE --label RESTRICTED --role admin --outside-workspace

# Without --role admin the same call is denied one step earlier, on clearance:
# the default 'engineer' role is only cleared to CONFIDENTIAL
.\.venv\Scripts\kavach.exe policy check fs.write --side-effect WRITE --label RESTRICTED --outside-workspace

# Same call one label down - allowed, but stops for a human
.\.venv\Scripts\kavach.exe policy check fs.write --side-effect WRITE --label CONFIDENTIAL --outside-workspace

# An operator is not cleared for CONFIDENTIAL data
.\.venv\Scripts\kavach.exe policy check kb.search --label CONFIDENTIAL --role operator

# A tool the engine has never heard of is still governed
.\.venv\Scripts\kavach.exe policy check some.brand.new.tool --side-effect DELETE
```

**Expect,** in order:

```
DENIED         a RESTRICTED run may not write outside its workspace
DENIED         role 'engineer' is cleared to CONFIDENTIAL; this run is RESTRICTED
NEEDS APPROVAL writing outside the workspace needs approval at CONFIDENTIAL
DENIED         role 'operator' is cleared to INTERNAL; this run is CONFIDENTIAL
NEEDS APPROVAL DELETE always stops for a human
```

Note the two different refusals for the same call: clearance is checked before the
workspace rule, so the reason string tells you *which* control fired.

### 6. Run the gateway

```powershell
.\.venv\Scripts\python.exe -m uvicorn kavach.api.app:app --host 127.0.0.1 --port 8420
```

> Port 8080 is reserved by Windows on some machines (`WinError 10013`). 8420 works;
> set `KAVACH_PORT` to change the default.

Then open <http://127.0.0.1:8420/docs> for the live API, or:

| Endpoint | Shows |
|---|---|
| `GET /api/health` | profile, backend, sentinel mode |
| `GET /api/sovereignty` | the whole dashboard panel in one poll |
| `POST /api/sovereignty/canary` | the canary beat |
| `GET /api/audit/verify` | chain intact + head hash |
| `GET /api/audit/tail?n=20` | the live log tail |
| `POST /api/policy/check` | one policy question |
| `GET /api/sovereignty/egress` | the egress ledger |

**Expect:** `/api/sovereignty` returns `external_packets: 0`,
`audit.verified: true`, and a `sentinel.mode` of `iphlpapi` on Windows
(`proc`, or `nftables` where the firewall is loaded, on Linux).

### 7. Check the audit chain caught everything you just did

```powershell
.\.venv\Scripts\kavach.exe audit tail -n 20
.\.venv\Scripts\kavach.exe audit verify
```

**Expect:** `system.boot`, `canary.start`, `canary.result`, `policy.check` records,
then `chain intact`. Every record's hash chains to the one before it.

---

## How to test Phase 2

### 1. The test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Expect:** `124 passed` (35 from Phase 1 + 89 new).

`tests/test_profiles.py` runs against the **shipped** `config/profiles.yaml` and
`config/models.yaml` rather than fixtures, so a profile whose numbers stop adding
up fails here instead of quietly thrashing the card at runtime.

### 2. Look at the registry

```powershell
.\.venv\Scripts\kavach.exe models list
```

**Expect:** 14 models, all `SIMULATED`, the pinned `bge-m3-emb` marked resident, and:

```
  VRAM  1.2 / 9.8 GB (8.6 free, 0.2 reserved for KV)
  slots 0 / 1 in use
  14 model(s) are SIMULATED - there are no weights on this box.
```

`SIMULATED` is a *distinct* state from `VERIFIED` — it means the mock backend is
standing in for weights that are not on this machine. It is flagged everywhere and
`models verify` fails on it, so nothing simulated can be mistaken for real.

### 3. Watch two models swap on one card

```powershell
$env:KAVACH_MOCK_TIME_SCALE=1.0
.\.venv\Scripts\kavach.exe models swap qwen3-8b-q4 qwen3-vl-8b-q4 --rounds 2
Remove-Item Env:\KAVACH_MOCK_TIME_SCALE
```

**Expect** — the MID profile has one swap slot, so each model evicts the other:

```
  1.qwen3-8b-q4              LOAD    4203 ms  vram 6.6 GB
  1.qwen3-vl-8b-q4           LOAD    5101 ms  vram 7.3 GB (evicted qwen3-8b-q4)
  2.qwen3-8b-q4              LOAD    4200 ms  vram 6.6 GB (evicted qwen3-vl-8b-q4)
  2.qwen3-vl-8b-q4           LOAD    5101 ms  vram 7.3 GB (evicted qwen3-8b-q4)
  total swap cost 18604 ms over 4 admissions
```

**18.6 seconds of pure swap cost for four admissions.** That number is the entire
argument for the router's swap penalty in Phase 3 — and it is a decision no cloud
router ever has to make.

> `KAVACH_MOCK_TIME_SCALE=1.0` replays the manifest's real `load_ms`. The default
> `0.01` keeps CI fast. Without it the same run reports 42 ms and looks fake.

Compare with the HIGH profile, which has two slots:

```powershell
$env:KAVACH_PROFILE="HIGH"
.\.venv\Scripts\kavach.exe models swap qwen3-8b-q4 qwen3-vl-8b-q4 --rounds 2
Remove-Item Env:\KAVACH_PROFILE
```

**Expect:** the first two are `LOAD`, the rest are `HIT` at 0 ms with no evictions —
the vision model no longer evicts the text model mid-run.

### 3b. Rehearse the contingency profile

If the venue machine has no usable GPU you fall back to `TINY`, so it needs to work:

```powershell
$env:KAVACH_PROFILE="TINY"
.\.venv\Scripts\kavach.exe models list
.\.venv\Scripts\kavach.exe models load qwen3-1_7b-q4
Remove-Item Env:\KAVACH_PROFILE
```

**Expect:** `VRAM 0.1 / 8.0 GB`, `minilm-l6-emb` resident, and the load succeeding.

> On a CPU-only profile there is no VRAM, so the warden budgets against `ram_gb`
> instead. Budgeting against `vram_gb` (`0.0`) meant nothing could ever be pinned —
> the fallback was dead exactly when it would be needed. Pinned by
> `test_tiny_profile_can_actually_load_models`.

### 4. Prove the weights are the ones you approved

```powershell
.\.venv\Scripts\python.exe scripts\provenance_demo.py
```

**Expect** six sections, ending in a refusal to boot:

| # | What it does | Expected |
|---|---|---|
| 1 | Stages weights nobody approved | `UNVERIFIED`, `routable False` |
| 2 | Operator approves that exact file | `VERIFIED`, `routable True` |
| 3 | Boots again, re-hashing from disk | `started cleanly` |
| 4 | Substitutes the file | new hash shown |
| 5 | Next boot | **`HALTED`**, naming the file and both digests |
| 6 | Enforcement off (dev only) | `MISMATCH`, still `routable False` |

This is the answer to *"how do you know the weights on this box are the weights you
approved?"* — and section 6 is the important one: even with enforcement disabled, a
mismatched model can never serve a request.

### 5. Stage a real weight file yourself

```powershell
# Make a stand-in for a GGUF
"pretend weights" | Out-File -Encoding ascii models\qwen3-8b-instruct-q4_k_m.gguf

.\.venv\Scripts\python.exe scripts\stage_model.py              # report only
.\.venv\Scripts\python.exe scripts\stage_model.py --write --only qwen3-8b-q4
.\.venv\Scripts\kavach.exe models verify                       # -> VERIFIED

# Now corrupt it and watch the box refuse to start
"different weights" | Out-File -Encoding ascii models\qwen3-8b-instruct-q4_k_m.gguf
.\.venv\Scripts\kavach.exe models list                         # -> refuses, exit 2
.\.venv\Scripts\python.exe scripts\stage_model.py              # -> CONFLICT
```

**Clean up afterwards:**

```powershell
Remove-Item models\qwen3-8b-instruct-q4_k_m.gguf
# then set that entry's sha256 back to null in config\models.yaml
```

> `stage_model.py` edits only the `sha256:` lines and leaves the rest of
> `models.yaml` byte-identical — a `yaml.safe_dump` round-trip would silently
> delete every comment in the file, including the warning that the `quality`
> numbers are unmeasured placeholders.

### 6. The extensibility beat, over HTTP

```powershell
.\.venv\Scripts\python.exe -m uvicorn kavach.api.app:app --host 127.0.0.1 --port 8420
```

| Endpoint | Shows |
|---|---|
| `GET /api/models` | every model + provenance, `resident`, `swap_cost_ms`, `fits` |
| `GET /api/models/residency` | the live VRAM meter |
| `GET /api/models/verify` | re-hash everything, report what is real |
| `POST /api/models/load` | `{"model_id": "..."}` — admit, showing the eviction |
| `POST /api/models/evict` | `{"include_pinned": false}` |
| `POST /api/models/reload` | re-read `models.yaml` **live** |
| `GET /api/models/backend` | which backend, and whether it is simulated |

Then paste a new entry into `config\models.yaml`, `POST /api/models/reload`, and
`GET /api/models` — it is a candidate with no restart. That is the 10:45 demo beat.

### 7. Confirm it all landed in the audit chain

```powershell
.\.venv\Scripts\kavach.exe audit tail -n 30
.\.venv\Scripts\kavach.exe audit verify
```

**Expect:** `model.admit`, `model.evict`, `model.registry_reload` records, and
`chain intact`. Every model load and eviction is in the tamper-evident log.

---

## How to test Phase 3

### 1. The test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Expect:** `248 passed` (124 from Phases 1–2 + 124 new).

### 2. Two task types, two models — the headline claim

```powershell
.\.venv\Scripts\kavach.exe route "Script the NPSH-available calculation and unit-test it"
.\.venv\Scripts\kavach.exe route "Summarise this 40-page SOP and list the safety interlocks"
```

**Expect** the first to choose `qwen2_5-coder-7b-q4` and the second `qwen3-8b-q4`,
each with the full score breakdown:

```
MODEL                      SCORE   QUAL    FIT    LAT   SWAP   VRAM
qwen2_5-coder-7b-q4        0.575   0.88 +0.002 -0.095 -0.165 -0.047   <-
qwen3-8b-q4                0.356   0.68 +0.002 -0.106 -0.165 -0.055
```

**Read the columns.** `QUAL` swings 0.88→0.55 between the two tasks while every
cost term barely moves — that is what "quality drives the decision" looks like,
and it is checkable rather than asserted. Every term is normalised 0–1, so the
weights in `config/router.yaml` mean what they say.

### 3. Hard constraints eliminate rather than score

```powershell
.\.venv\Scripts\kavach.exe route "Read this scanned inspection report" --images
```

**Expect** 3 of 4 candidates eliminated with reasons, leaving only the VL model:

```
eliminated (3 of 4):
  qwen3-8b-q4              images attached, no vision tower
  qwen2_5-coder-7b-q4      images attached, no vision tower
  bge-m3-emb               not a generation model (role=embedding)
```

`qwen3-8b-q4` scores *higher* than the VL model on summarisation — stage 1 removes
it anyway. A model with no vision tower is not a bad choice for a scan, it is not
a choice, and scoring it would let a high quality number win an argument it should
never have been in.

### 4. The swap penalty, which is the interesting one

```powershell
.\.venv\Scripts\python.exe scripts\swap_penalty_demo.py
```

**Expect** the margin between the top two candidates to collapse once the vision
model is resident:

```
1. Nothing resident              margin 0.2366
2. Vision model resident         margin 0.0013   <- close call
```

A 0.15 quality gap almost exactly cancels a 4.2 s reload. The better model still
wins — but the router reports `close call, margin 0.001` rather than presenting a
coin-flip as a confident choice. **This is the decision no cloud router ever has to
make**, and it is the clearest 30 seconds for explaining why the router is
VRAM-aware rather than a quality lookup table.

Tune with `w_swap` in `config/router.yaml`: higher keeps resident models longer,
lower prefers quality and reloads more often.

### 5. The verifier escalates once, and never loops

```powershell
.\.venv\Scripts\kavach.exe route --run "Write a Python function to compute Reynolds number"
```

**Expect** a `FAIL` (the mock backend never emits valid Python, so the parse check
correctly rejects it) and then:

```
-> already the most capable model available - returning the failure
   rather than pretending it passed
```

On MID every generation model is tier 2, so there is nothing to escalate to. Try
the profile that has a tier-3 model:

```powershell
$env:KAVACH_PROFILE="HIGH"
.\.venv\Scripts\kavach.exe route --run "Write a Python function to compute Reynolds number"
Remove-Item Env:\KAVACH_PROFILE
```

**Expect** attempt 1 on `qwen2_5-coder-7b-q4` → `escalating to tier 3` → attempt 2
on `qwen3-14b-q4` → stop. **Two attempts, never more.**

### 6. Measure the router instead of trusting it

```powershell
.\.venv\Scripts\python.exe scripts\eval_harness.py --routing --confusion
```

**Expect** `40/42 = 95.2%` and a near-clean diagonal. The two residuals are
`code_generation → agentic_multi` on prompts using verbs both classes share; they
degrade to the planner tier, which can still write code.

This is the loop that improves the router: a misroute gets pasted into
`config/prototypes.yaml` under the right label and is fixed in seconds — no
retraining. **The harness found the seed prototype set at 78.6%**, with
`agentic_multi` absorbing 8 of 9 misroutes; that defect is invisible without a
held-out set.

> The number is measured with the **mock n-gram embedder**, not bge-m3, and the
> harness says so on every run. Treat it as a smoke test of the prototype set,
> not the production figure.

### 7. The gate that protects the panel

```powershell
.\.venv\Scripts\python.exe scripts\eval_harness.py --quality --write
```

**Expect** `REFUSING TO RUN`. Quality mode needs real weights; on the mock backend
it will not stamp `quality_measured: true` onto synthetic output. See the warning
below.

### 8. The explainer panel over HTTP

```powershell
.\.venv\Scripts\python.exe -m uvicorn kavach.api.app:app --host 127.0.0.1 --port 8420
```

| Endpoint | Shows |
|---|---|
| `POST /api/route/explain` | stages 1–3 only — **no VRAM, no tokens**; what the panel polls |
| `POST /api/route` | the full cascade with generation + verifier |
| `POST /api/route/classify` | stage 2 alone, with nearest neighbours |
| `GET /api/route/prototypes` | counts per class |
| `POST /api/route/prototypes/reload` | fix a misroute live |
| `GET /api/route/config` | the weights currently in force |

`explain` is deliberately side-effect free so the UI can show the arithmetic for a
prompt the user is still typing.

---

## How to test Phase 4

### 1. The test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Expect:** `382 passed, 5 skipped`. The skips are 3 symlink-escape tests
(unprivileged Windows cannot create symlinks) and 2 sandbox tests (need Docker) —
**both are coverage gaps on this machine, not passes.** They must run in Linux CI.

### 2. The workspace jail

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_workspace.py -q
```

**Expect:** `30 passed, 3 skipped`. These are adversarial — `../` traversal,
absolute paths, drive-relative `C:file`, null bytes, symlink escapes, and the
prefix-matching bug where `/runs/run_a` must not contain `/runs/run_ab`.

A model writes these paths, so they are hostile input by definition — not because
the model is malicious, but because a confused 7B model emitting
`../../../../etc/passwd` produces the same bytes an attacker would.

### 3. Tools are declared, not hard-coded

```powershell
.\.venv\Scripts\kavach.exe tools list
```

```
TOOL           EFFECT    SANDBOX   GATE               ARGS
calc           COMPUTE   -         none               expression
code.exec      EXECUTE   none-net  none               code, language?
fs.list        READ      -         none               -
fs.read        READ      -         none               path
fs.write       WRITE     -         outside -> approve path, content
```

The policy engine keys off `EFFECT`, never the name — so a tool it has never seen
is governed on its first call. Drop a manifest in
`services/kavach/tools/manifests/` and `POST /api/tools/reload`.

### 4. Calculations that show their working

```powershell
.\.venv\Scripts\kavach.exe tools call calc '{\"expression\":\"rho*v*d/mu\",\"substitutions\":\"rho=998,v=1.2,d=0.1524,mu=0.001\"}'
```

**Expect** `182514.24` with the derivation — substitution, evaluation, result.
That is the traced requirement "calculations with steps shown"; a number with no
derivation is not checkable, and in a plant context an uncheckable number is worse
than no answer.

### 5. The policy gate, enforced at the chokepoint

```powershell
# RESTRICTED cannot write outside its workspace - the visible refusal
.\.venv\Scripts\kavach.exe tools call fs.write '{\"path\":\"../escape.txt\",\"content\":\"x\"}' --label RESTRICTED --role admin

# One label down: allowed, but stops for a human
.\.venv\Scripts\kavach.exe tools call fs.write '{\"path\":\"../escape.txt\",\"content\":\"x\"}' --label CONFIDENTIAL
```

**Expect** `DENIED` then `NEEDS APPROVAL`. Note the second one: even *with*
`--approve`, the workspace jail still refuses — approval is not a licence to escape
the sandbox, and there are two independent controls saying so.

### 6. The sandbox — needs Docker

```powershell
.\.venv\Scripts\kavach.exe tools sandbox          # is it available?
docker compose build sandbox-runner               # build it (~2 min)
.\.venv\Scripts\python.exe scripts\sandbox_demo.py
```

**Expect**, with Docker running:

```
1. Ordinary work runs fine            sum: 10   exit 0 in 1085 ms
2. Trying to reach the network        blocked: URLError: [Errno 1] Operation not permitted
                                      blocked: PermissionError: [Errno 1] Operation not permitted
3. Trying to escape the filesystem    blocked: [Errno 30] Read-only file system
                                      /work contains: []
                                      uid=65534 gid=65534
```

**Read the errno.** `Operation not permitted` is **EPERM (1)**, not
`Network is unreachable` (ENETUNREACH, 101). That means the **seccomp profile**
refused `socket()` before the network stack was ever consulted — the syscall
allowlist fired first, and `--network=none` never even had to be the thing that
saved us. Two independent controls, and the outer one is doing the work.

**Without Docker**, `code.exec` **refuses to run**:

```
SandboxUnavailable: code.exec needs the Docker sandbox and it is not available.
To run WITHOUT a sandbox during development, set KAVACH_ALLOW_UNSANDBOXED_EXEC=1.
```

That refusal is deliberate. A sandbox that silently stops sandboxing is worse than
one that stops working. The dev fallback exists but is off by default and stamps
every result `sandboxed: false` with a `HOST - NOT ISOLATED` marker.

**The snippet is run as the language it is.** Python, C, C++, Java, JavaScript
and shell are in the image; the source is written to a file with the matching
extension and compiled or interpreted with its own toolchain, and the result says
which language was chosen and why (`language`, `language_reason`, `source_file`).
Name it with the `language` argument, or let it be read off the mounted file's
extension or the source. A language the image has no toolchain for is **refused
by name**, with the `deploy/Dockerfile.sandbox` line that adds it - running C
through the Python interpreter is how a correct C program came back as
`NameError: name 'printf' is not defined`.

### 7. GBNF grammars

```powershell
.\.venv\Scripts\python.exe -m uvicorn kavach.api.app:app --host 127.0.0.1 --port 8420
# then: GET /api/tools/calc/grammar
```

**Expect** a grammar with the tool name pinned to a literal, required keys
mandatory and optional keys marked `?`. Handed to llama.cpp, output that does not
match becomes *unreachable* rather than merely unlikely.

> **Unverified.** These tests assert the grammar's structure, not that llama.cpp
> accepts it — that needs a real llama.cpp build (checkpoint **R4**). Until then,
> "GBNF prevents malformed tool calls" is a designed property, not a measured one.

### 8. The tool-plane endpoints

| Endpoint | Shows |
|---|---|
| `GET /api/tools` | every tool, schema, side effect, gate |
| `GET /api/tools/sandbox` | can `code.exec` isolate anything right now |
| `POST /api/tools/invoke` | run one tool |
| `POST /api/tools/sandbox/probe` | the egress probe, over HTTP |
| `GET /api/tools/{name}/grammar` | the GBNF for that tool |
| `POST /api/tools/reload` | pick up a new manifest live |

Status codes carry meaning: **403** policy refused, **428** needs human sign-off,
**422** the model produced bad arguments.

---

## How to test Phase 5

### 1. The test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Expect:** `455 passed, 3 skipped` (the skips are the symlink tests; set
`KAVACH_TEST_SYMLINKS=1` in an **elevated** shell to run them).

### 2. An agentic run, end to end

```powershell
.\.venv\Scripts\kavach.exe run "Calculate the pressure drop and write it to a report file"
```

**Expect** a plan, then each step turning from `>>>` to `OK`, then artifacts:

```
plan (3 steps)
  0. compute the quantity needed for: Calculate the pressure drop
     tool calc  |  accept: the step produces output
  1. write the working to a file
     tool fs.write  |  accept: file result.txt exists
  ...
status   done
budget   3/12 steps, 3/20 tool calls, 0.0s
artifacts
  result.txt  6d23a7b7ccc9
```

Read the `accept:` lines. **A plan without acceptance tests is a list of
intentions**, and a loop over intentions cannot tell success from a confident
failure. "Write the calculation" is not checkable; "file result.txt exists" is.

`--plan-only` produces the plan without executing it.

### 3. Budgets are terminal

```powershell
.\.venv\Scripts\kavach.exe run "do a lot of things" --max-steps 1
```

**Expect** `budget_exhausted` with `step budget (1) exhausted`. Exhausted means
stop and report — never "one more try". Re-running `advance` on an exhausted run
changes nothing.

### 4. The approval gate survives a process restart

An approval gate that only holds while the process is up is theatre. This is
pinned by `test_a_paused_run_survives_a_process_restart`, which parks a run, then
reloads it **purely from SQLite** as a fresh process would, and confirms it is
still `awaiting_approval`.

```powershell
.\.venv\Scripts\kavach.exe runs               # list runs
.\.venv\Scripts\kavach.exe runs show <run_id> # the timeline
.\.venv\Scripts\kavach.exe runs approve <run_id> --step 1 --by pverma
```

### 5. Failure replans, boundedly

`test_a_failed_acceptance_test_triggers_a_replan` forces a step's acceptance test
to fail and asserts a `replanned` event. Two further properties:

- **The failed step stays in history.** A plan that edits its own history is not
  a plan, it is a story.
- **Replanning is capped at 2.** `test_replanning_is_bounded` makes *every* step
  fail and asserts the run terminates rather than spinning.

### 6. The run endpoints

| Endpoint | Shows |
|---|---|
| `POST /api/runs` | plan + execute (`execute: false` to plan only) |
| `GET /api/runs/{id}/timeline` | what the centre panel polls |
| `POST /api/runs/{id}/advance` | resume after approval or restart |
| `POST /api/runs/{id}/approve` | sign off one step, by name |
| `GET /api/runs/{id}/artifacts` | deliverables with SHA-256 |

### 7. The whole run is in the audit chain

```powershell
.\.venv\Scripts\kavach.exe audit tail -n 40
.\.venv\Scripts\kavach.exe audit verify
```

**Expect** `run.start`, `plan.accepted`, `tool.invoke`, `step.judged`, `run.done`
— and `chain intact`.

> **On the mock backend the *plan* is simulated too.** The mock emits a
> structurally valid plan so the loop can be exercised offline; the steps are real
> tool calls with real acceptance checks, but the planning is not a real model's
> reasoning. Checkpoint **R4** is where that becomes real.

---

## How to test Phase 6

### 1. The test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Expect:** `573 passed, 3 skipped` (455 from Phases 1–5 + 118 new). The skips are
still the symlink tests.

### 2. Ingest the corpus, and read the number that matters

```powershell
.\.venv\Scripts\kavach.exe kb ingest MRPL_DEMO_CORPUS
```

**Expect** eleven documents, and **six of them reported as gaps**:

```
  OK    SOP_01                         1 page(s), 4 chunk(s), CONFIDENTIAL
        native=1
  OK    inspection_report_01           1 page(s), 0 chunk(s), CONFIDENTIAL
        unreadable=1
        UNREADABLE: p.1
  ...
  ingested 11, skipped 0, failed 0 in 0.8s
  index    16 chunk(s), hybrid: BM25 + bge-m3-emb over 16 chunks

  6 page(s) across 6 document(s) could not be read - they are NOT in the index
```

**That warning is the feature.** There is no OCR engine on this box and the only
vision model is `SIMULATED`, so the four scans, both P&IDs and the handwritten
annotation genuinely cannot be read. The alternative — indexing them as empty
pages — gives you a corpus that reports 11 of 11 documents loaded and then
quietly cannot answer six of the questions it was built for, with nothing on
screen to explain why.

Note the classifications. `SOP_01` is `CONFIDENTIAL` because the marking is on
its own face. `inspection_report_01` is `CONFIDENTIAL` because **nobody read it**
— the marking is usually in the header of the page that failed, so an unread
document is over-classified rather than handed to an operator.

```powershell
.\.venv\Scripts\kavach.exe kb doctor
```

**Expect** `image path UNREADABLE`, `lexical on`, `dense on`, the per-page
breakdown, and the six gaps listed by document and page. Exit code 1 — this box
is honestly degraded, and `doctor` says so rather than printing a tick.

### 3. Both retrieval arms, and why there are two

```powershell
.\.venv\Scripts\kavach.exe kb search "PSLL-101"
.\.venv\Scripts\kavach.exe kb search "what stops the pump if suction pressure falls too low"
```

**Expect** the first to be carried by `bm25` and the second by `dense`, with the
`via` column on every hit naming which arms found it.

`PSLL-101` is a string the embedder has never seen; the paraphrase shares no rare
token with the document that answers it. A refinery corpus is *both* of those at
once, which is why neither arm alone is enough.

> Fusion is **Reciprocal Rank Fusion**, not a weighted sum of the two scores.
> BM25 scores are unbounded and corpus-dependent, cosine is 0–1; adding them makes
> the weight meaningless and lets the lexical arm win every query on a corpus full
> of tag numbers. Same lesson §04 learned about raw seconds.

### 4. Clearance is enforced on retrieval, and counted

```powershell
.\.venv\Scripts\kavach.exe kb search "hot work permit gas testing" --role operator
```

**Expect** the `INTERNAL` passages returned and:

```
  4 passage(s) withheld - role 'operator' is cleared to INTERNAL; this run is CONFIDENTIAL
```

**Read that line, not the hits.** Returning four results instead of eight teaches
the operator their query was weak, which is not what happened. The corpus is the
only place in Kavach where CONFIDENTIAL text enters a model's context, so the
filter is applied by the policy engine at the retrieval boundary — and `kb.search`
receives the caller's role from the invocation, never from a global.

A run that reads one CONFIDENTIAL passage becomes a CONFIDENTIAL run for every
step after it (`run.label_raised` in the audit chain). Policy.yaml has always said
labels are inherited; before Phase 6 nothing could actually raise one.

### 5. The demo script — four beats

```powershell
.\.venv\Scripts\python.exe scripts\knowledge_demo.py
```

**Beat 3 is the one to show a judge.** Four answers, all well-formed, all with
square brackets in the right places:

| | Answer | Verdict |
|---|---|---|
| 1 | cites `[SOP_01 p.1 s.2]`, which was retrieved | **ACCEPTED** |
| 2 | same answer, page nudged to `p.2` | REFUSED — `unknown_page`, SOP_01 has 1 page |
| 3 | cites `[API-610 p.7]` | REFUSED — `unknown_document`, never ingested |
| 4 | fluent, no citation at all | REFUSED — an uncited claim is worse than no answer |

Three of those four survive every check that only looks for the *shape* of a
citation. The check here is stricter: **every citation must resolve to a passage
that was actually in that answer's context.** There is a fourth verdict,
`out_of_context` — the document and page exist, but were not retrieved for this
answer, so the model is citing from its weights rather than from the corpus. That
citation may even be correct, and it is still not evidence, because nothing in
this system checked it.

### 6. A grounded answer, or a refusal with its reason

```powershell
.\.venv\Scripts\kavach.exe kb ask "What is the suction pressure trip setpoint for P-101A?"
```

**Expect** an answer whose every line ends in a citation, the source list, and:

```
  grounded in 3 cited passage(s) - but the prose is SIMULATED OUTPUT from the
  mock backend. The citations resolve; the words between them are not a model's
  reasoning.
```

**Both halves of that sentence are true and they are different facts.** The mock
backend copies citations out of the context it was handed — deliberately, the same
way it emits structurally valid plans in Phase 5 — so the whole citation path is
exercised offline. On a tier-0 or tier-1 model it sometimes shifts a page number
on purpose, because a mock that never misbehaves leaves the resolver untested
against the pipeline that feeds it. Checkpoint **R5** is where the prose becomes
real.

Ask something the corpus cannot answer:

```powershell
.\.venv\Scripts\kavach.exe kb ask "What is the torque spec for the compressor coupling?"
```

**Expect** `NO GROUNDED ANSWER` and `no passage in the corpus matched this
question` — **and no model call at all**. A model asked a plant question with no
context still produces fluent, specific, wrong text about interlock setpoints.

### 7. The image path, on a box that cannot do it

Every scan is an honest gap here. To exercise the plumbing anyway:

```powershell
$env:KAVACH_KB_ALLOW_SIMULATED_OCR="1"
.\.venv\Scripts\kavach.exe kb ingest MRPL_DEMO_CORPUS\scanned_reports --force
Remove-Item Env:\KAVACH_KB_ALLOW_SIMULATED_OCR
```

**Expect** the pages ingested as `SIMULATED`, the documents over-classified to
`CONFIDENTIAL`, **zero chunks indexed**, and:

```
  3 page(s) hold SIMULATED transcriptions (KAVACH_KB_ALLOW_SIMULATED_OCR is set)
  - they are NOT indexed and nothing can be cited to them
```

Off by default, awkward to switch on, stamped everywhere, excluded from retrieval,
and refused as evidence even if something did hand it to the answerer — the same
treatment `code.exec` gives an unsandboxed run.

### 8. The knowledge endpoints

```powershell
.\.venv\Scripts\python.exe -m uvicorn kavach.api.app:app --host 127.0.0.1 --port 8420
```

| Endpoint | Shows |
|---|---|
| `GET /api/kb` | the corpus panel in one poll — counts, extraction breakdown, gaps |
| `GET /api/kb/doctor` | what the knowledge plane can and cannot prove right now |
| `GET /api/kb/documents` | every document with its classification and read pages |
| `GET /api/kb/documents/{id}` | per-page extraction and why a page failed |
| `POST /api/kb/ingest` | `{"path": "MRPL_DEMO_CORPUS"}` |
| `POST /api/kb/search` | hybrid retrieval with citations, coverage and withheld counts |
| `POST /api/kb/answer` | retrieve → route → check every citation |
| `GET /api/kb/chunks?doc_id=…&page=…` | the passages behind a citation |
| `POST /api/kb/reindex` · `POST /api/kb/reload` | rebuild the arms · re-read `knowledge.yaml` |

`POST /api/kb/answer` **always returns 200**, including when it refuses. A 4xx
becomes "something went wrong" in a UI; the refusal *is* the answer and the panel
has to render it with its reason.

### 9. It is all in the audit chain

```powershell
.\.venv\Scripts\kavach.exe audit tail -n 30
.\.venv\Scripts\kavach.exe audit verify
```

**Expect** `kb.ingest_start`, one `kb.document_ingested` per file (with its
sha256, page count and extraction breakdown), `kb.ingest_failed` for anything that
would not open, `kb.answer` with the grounding verdict, and `chain intact`.

Document *text* never enters the chain — only its digest and its shape. The corpus
is CONFIDENTIAL more often than not, and the audit log is not the place for it;
the same rule the tool executor applies to arguments (§06).

---

## How to test Phase 7

### 1. The test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Expect:** `636 passed, 3 skipped` (573 from Phases 1–6 + 63 new). The skips are
still the symlink tests.

### 2. The panel

```powershell
.\.venv\Scripts\kavach.exe serve --port 8420
```

**Expect** the URL printed before uvicorn's log takes the screen, then open
<http://127.0.0.1:8420/>:

```
Kavach gateway
  API      http://127.0.0.1:8420/docs
  panel    http://127.0.0.1:8420/
  every asset is served from this box; nothing is fetched
```

A sidebar and five views: **Overview**, **Agentic run** (L5), **Knowledge**
(L3), **Sovereignty** (L0) and **Fabric & router** (L1/L2). The overview carries
the KPI tiles, the corpus-coverage bar, the VRAM donut and the recent runs.

**Look at the amber band across the top first.** That is the point of the
interface. Every subsystem that cannot currently do its real job hoists its
reason up there, out of the panel it came from:

```
THIS BOX CANNOT CURRENTLY PROVE
  - Egress is MEASURED but not ENFORCED on this box - 0 packets counted...
  - The model backend is the MOCK. Every generation on this page is synthetic...
  - 14 model(s) have no weights on this box (provenance SIMULATED)...
  - Router quality scores are HAND-SEEDED, not measured...
  - 6 of 11 pages in this corpus could not be read on this box...
```

A caveat that lives inside the panel it applies to is a caveat nobody reads while
looking at a different panel — and every one of those lines is something a judge
would otherwise have to extract in Q&A.

> **No framework, no build step, no `node_modules`.** Four files
> (`index.html`, `app.css`, `charts.js`, `app.js`) under `services/kavach/ui/`,
> served same-origin by the gateway. Every chart is inline SVG drawn by hand. That is not minimalism for its own sake: a Vite
> bundle is a dependency tree to stage offline, a lockfile to audit and a build to
> reproduce on a venue machine. Three files a judge can open in Notepad are a
> better answer to *"prove it does not phone home"* — and section 5 turns it from
> an answer into a check. The one thing that *is* spelled like a URL,
> `http://www.w3.org/2000/svg`, is an XML namespace name the browser never
> dereferences; it is exempted as a whole literal, so `https://www.w3.org/TR/…`
> would still fail.

### 3. The live run stream

Type a goal into the run panel and press Run. **Expect** each step to turn from
`...` to `>>>` to `OK` as it happens, not on the next poll.

Watch it from the command line instead:

```powershell
# plan a run without executing it, then attach to the stream, then start it
$id = (curl.exe -s -X POST http://127.0.0.1:8420/api/runs -H "Content-Type: application/json" -d '{\"goal\":\"Calculate the pressure drop and write it to a report file\",\"execute\":false}' | ConvertFrom-Json).run_id
curl.exe -s -N http://127.0.0.1:8420/api/runs/$id/stream
# in another shell:
curl.exe -s -X POST http://127.0.0.1:8420/api/runs/$id/advance -H "Content-Type: application/json" -d '{\"background\":true}'
```

**Expect:**

```
event: snapshot
event: step_started
event: step_passed
event: step_started
event: step_passed
event: step_started
event: step_passed
event: done
event: stream_end
```

**Read the first and last events, not the middle.**

`snapshot` is the complete timeline, sent before anything else. A panel that
connects halfway through a run is **correct immediately**, not eventually — which
is what makes it safe for the stream to lose events.

`stream_end` is terminal. A connection that simply goes quiet is indistinguishable
from a model still thinking, and a panel that cannot tell those apart shows a
spinner forever.

> **The stream is an optimisation, never the source of truth.** Every event it
> carries is already in SQLite, and `GET /api/runs/{id}/timeline` renders the run
> correctly with no stream at all. So the broker is allowed to be lossy — and when
> a subscriber's queue overflows it emits `dropped` with the count, and the panel
> re-fetches the timeline and says *"fell behind (12 events) — re-synced"* in its
> header. Silently discarding events would leave a UI that is subtly, permanently
> wrong with nothing on screen to say so.

Kill the gateway mid-run and reload the page: the run panel falls back to polling
and the header says `polling - the event stream closed`. It never shows a stale
run as a live one.

### 4. Two loops cannot race on one run

```powershell
# advance a run that is already executing
curl.exe -s -X POST http://127.0.0.1:8420/api/runs/$id/advance -d '{\"background\":true}'
```

**Expect** `409` and `run ... is already executing; watch /stream instead`. Two
loops advancing one run interleave their steps and corrupt the budget; a caller
told "fine" would never learn why its second click did nothing.

### 5. Prove nothing here phones home

```powershell
.\.venv\Scripts\python.exe scripts\offline_check.py
```

**Expect** four sections and exit code 0:

| # | Checks | Fails the build |
|---|---|---|
| 1 | every UI asset for an absolute URL — CDN, web font, favicon, `@import url(https://…)` | yes |
| 2 | `index.html` carries a CSP confining the page to `'self'` | yes |
| 3 | every declared dependency imports from this environment **right now** | yes |
| 4 | every hostname in the Python source is on the justified list | no — reports |

Check 1 is the one that matters. A single `<script src="https://cdn…">` added in
a hurry turns a sovereign workbench into a blank panel on a plant network — and
into a line in someone else's request log if the network is only half closed. It
would fail no other test in this repo.

Check 4 **reports rather than fails** on purpose: the canary's RFC 5737 target and
`code.exec`'s egress probe are absolute URLs whose whole job is to be attempted
and fail. A check that fires on its own subject gets suppressed within a week.

Prove the checker can actually fail — `tests/test_offline.py` points it at a
poisoned bundle containing a jsDelivr script and a Google Fonts `@import`, and
asserts it catches both.

> **Staging the wheels.** `pip download -d wheels -e ".[knowledge]"` on a
> connected machine, carry `wheels/` on the SSD, then
> `pip install --no-index --find-links wheels -e ".[knowledge]"` on the box. Check
> 3 is what tells you it worked, before the venue's wifi does.

### 6. The run sheet — the artifact you hand a judge

```powershell
.\.venv\Scripts\python.exe scripts\run_sheet.py --out var\run_sheet.md
```

**Expect** `10 proven, 0 partial, 9 not proven` on a dev box with the corpus
ingested, and a markdown file whose **first section is what could not be shown**:

```markdown
## What this box did NOT prove (9)

### Egress is blocked by enforcement, not by absence of a route
- **why** needs nftables on a Linux box. On this machine the canary returns
  INCONCLUSIVE, and a timeout is not proof - a dropped packet and an unplugged
  cable look identical from here.
- **to close it** `sudo nft -f deploy/nftables.conf && kavach canary`
```

Every other status document opens with achievements and buries the caveats in a
"future work" paragraph. The unproven list is the one a technical judge builds
during Q&A anyway; handing it over up front is the difference between a caveat and
a discovery — and it is what makes the *proven* half worth believing.

**Nothing in it is typed by hand.** Every line is read out of the live system when
the sheet is generated: the audit chain's record count, the registry's provenance
mix, the router's `quality_provenance`, the corpus's unreadable-page count, whether
Docker is actually running. A hand-maintained status document drifts between
rehearsal and demo day; this one cannot.

It also **refuses to fire the canary**. The canary attempts egress, and a document
generator that changes the system it documents is a document you cannot trust —
pinned by `test_the_run_sheet_does_not_run_the_canary`.

### 7. The Phase 7 endpoints

| Endpoint | Shows |
|---|---|
| `GET /` | the panel, served same-origin |
| `GET /api/runs/{id}/stream` | `text/event-stream` — snapshot, then live events, then `stream_end` |
| `GET /api/runs/{id}/live` | is a thread actually running, and is anyone watching |
| `POST /api/runs` with `stream: true` | plan synchronously, execute on a background thread, return at once |
| `POST /api/runs/{id}/advance` with `background: true` | resume without blocking — **409** if already executing |
| `GET /api/ui` | is the panel bundle present on this box at all |

`POST /api/runs` keeps planning **synchronous** even in streaming mode. Planning is
the step that fails outright, and a caller holding a run id should already know the
plan was accepted rather than discovering it from a stream that never emits.

---

## ⚠️ Before the router panel goes in front of judges

The `QUAL` column is **hand-seeded, not measured**. The router ranks on it and the
panel displays it, so the moment a judge asks *"where does 0.84 come from?"* there
has to be an answer on the screen.

Right now the honest answer is "those are seeds" — and the system says so itself,
unprompted, in four places: the CLI prints a warning under every decision,
`/api/route/explain` returns `quality_provenance: "seeded"` plus a
`quality_warning`, every `route.decide` audit record carries the provenance, and
`eval_harness --quality` refuses to launder stub output into a measurement.

**To close it:**

```powershell
# 1. Stage real GGUFs and approve their digests
.\.venv\Scripts\python.exe scripts\stage_model.py --write
# 2. Measure, and write the results into models.yaml
$env:KAVACH_BACKEND="llamacpp"
.\.venv\Scripts\python.exe scripts\eval_harness.py --quality --write
# 3. Confirm the warning is gone
.\.venv\Scripts\kavach.exe route "Script the NPSH calculation"    # no WARNING block
```

Then re-run `--routing` on the real embedder so the accuracy figure is the
production one.

---

## Running against real weights

```powershell
# 1. Stage the GGUFs into .\models (bring them on an SSD - assume no venue internet)
# 2. Record their digests
.\.venv\Scripts\python.exe scripts\stage_model.py --write
# 3. Switch the backend
$env:KAVACH_BACKEND="llamacpp"
.\.venv\Scripts\kavach.exe models verify     # must print "every one of N ... approved digest"
```

The warden spawns one `llama-server` per resident model and kills it on eviction —
that is what makes eviction actually free VRAM. `llama-server` must be on `PATH`.
See the notes at the bottom of `docker-compose.yml` for the GPU-enabled container.

---

## Testing the real sovereignty claim (Linux / demo box only)

Phase 1's Windows behaviour is deliberately honest-but-unproven. The real thing
needs the firewall and the internal network:

```bash
sudo nft -f deploy/nftables.conf          # default DROP + named counters
docker compose up --build gateway
docker network inspect kavach_kavach_core | grep -i internal   # -> "Internal": true
kavach canary                              # -> BLOCKED, ENETUNREACH
sudo nft list counter inet kavach egress_drop
```

> `nftables.conf` **will cut that machine off the network**. Run it on the demo box
> or in a VM. `sudo nft delete table inet kavach` reverses it.

---

## Configuration

Copy `.env.example` to `.env`. The knobs that matter:

| Variable | Default | Notes |
|---|---|---|
| `KAVACH_PROFILE` | `MID` | `TINY` / `LOW` / `MID` / `HIGH` / `ARC` — same code path |
| `KAVACH_BACKEND` | `mock` | `mock` needs no weights; `llamacpp` from Phase 2 |
| `KAVACH_ENFORCE_DIGESTS` | `1` | Halt boot if a weight file is not the one approved |
| `KAVACH_VAR_DIR` | `./var` | SQLite state + audit chain live here |
| `KAVACH_MODEL_DIR` | `./models` | Where staged GGUFs live |
| `KAVACH_MOCK_TIME_SCALE` | `0.01` | `1.0` replays real load latency for rehearsals |
| `KAVACH_PORT` | `8080` | Use `8420` on Windows |
| `KAVACH_KB_ALLOW_SIMULATED_OCR` | `0` | Index synthetic transcriptions for pages this box cannot read. Off by default; everything it produces is stamped `SIMULATED` and excluded from grounded answers |

---

## Known limitations after Phase 7

- **The panel is unstyled for a projector.** It was built for a laptop screen at
  1 metre. Rehearse it on the actual display before the demo; the amber banner in
  particular has to be legible from the back of a room.
- **RBAC is still config-only.** The corpus panel has a role selector, and it is a
  dropdown the user picks - not a login. Every caller is `local`, and the clearance
  filter demonstrates the *mechanism*, not an authenticated identity.
- **The stream has no reconnect-with-cursor.** A browser that reconnects gets a
  fresh snapshot plus whatever the broker still buffers, which is correct but
  re-sends work. `Last-Event-ID` would be the tidy fix and is not implemented.
- **One gateway process, one broker.** Run state is in SQLite and would survive,
  but the in-memory event buffer would not be shared across workers. `--workers 1`
  is assumed, which is right for one box and worth knowing.

## Known limitations after Phase 6

- **The OCR/VLM path is unexercised.** There is no OCR engine on this box and no
  real vision weights, so six of the eleven MRPL documents are honest gaps rather
  than transcriptions. Everything downstream of the fork is built and tested; what
  is unmeasured is whether a real VL model can read an MRPL P&ID well enough to
  cite. That is checkpoint **R5**, and until it runs the corpus panel shows a
  number you can defend rather than one you would rather show.
- **Grounded answering has only been run against the mock backend.** The mock
  copies citations out of its context, which exercises the resolver end to end -
  it does not tell you whether a 7B model under those instructions will. **R4**.
- **Retrieval quality is unmeasured.** There is no labelled question set for the
  MRPL corpus, so "hybrid beats either arm" is demonstrated on two hand-picked
  queries, not measured. The router has `eval_harness.py`; the knowledge plane has
  nothing equivalent yet.
- **No reranker stage.** `bge-reranker-v2-m3` is in the registry and in the HIGH
  profile's swap pool, and nothing calls it. RRF over two arms is the whole
  ranking. Worth adding once there is a question set to show it helps.

## Known limitations after Phase 2

- **The sentinel measures on Windows, but nothing enforces there.** It reads the
  same kernel tx counters through `GetIfTable2`, so the packet count is real on
  either platform — but without nftables there is no drop counter, and the panel
  says `MEASURED, NOT ENFORCED` instead of implying a firewall exists. If neither
  counter source can be read at all it still reports `DEGRADED` rather than a
  zero it cannot stand behind.
- **The canary is inconclusive without nftables** — by design (see above).
- **Every model is `SIMULATED`** until you stage GGUFs and record digests. The mock
  backend is deterministic and honest about timing, but its output is synthetic and
  labelled `[SIMULATED OUTPUT]` so it can never be screenshotted as a real answer.
- **The `quality` numbers in `config/models.yaml` are seeds, not measurements.**
  The router will happily rank models on them in Phase 3 and the explainer panel
  will show the arithmetic — but you cannot defend those numbers in Q&A until the
  eval harness has replaced them. This is the single biggest credibility gap in
  the build right now.
- **VRAM figures are the manifest's claims, not a measured allocation.** The warden
  does the accounting from `vram_gb` in the YAML; it does not read `nvidia-smi`.
  Good enough to prevent OOM if the manifest is honest, worth verifying once on
  real hardware.
- **No router or agent yet.** Nothing selects a model automatically — Phase 3.
- **RBAC is config-only.** Roles are defined and enforced in the policy engine, but
  there is no login yet; every caller is `local`/`engineer`.
