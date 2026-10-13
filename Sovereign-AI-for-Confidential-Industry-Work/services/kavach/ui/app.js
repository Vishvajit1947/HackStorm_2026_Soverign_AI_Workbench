/* Kavach workbench (L6, §09).

   Vanilla ES2020, no dependencies, no build. Every URL below is relative -
   `scripts/offline_check.py` fails the build if that ever stops being true.

   Two rules this file is written around.

   THE STREAM IS AN OPTIMISATION, NEVER THE SOURCE OF TRUTH. The run view opens
   an EventSource, but every event it receives is a hint to re-read state the
   server already holds. If SSE never connects, if the browser drops it, or if
   the broker reports `dropped`, the view falls back to polling /timeline and
   SAYS SO in its header. It never shows a stale run as a live one.

   A DEGRADED SUBSYSTEM IS RENDERED, NOT HIDDEN. Anything this box cannot
   currently prove is hoisted into the banner at the top of the page, whichever
   view it came from, and counted on the nav item that owns it. A caveat inside
   the panel it applies to is a caveat nobody reads while looking at a different
   panel.
*/

"use strict";

const POLL_MS = 2000;
const SPARK_POINTS = 40;

// ------------------------------------------------------------------ helpers

const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

/* Every panel below is rebuilt from an API payload on a 2s tick, and tearing a
   list down only to append an identical one is a visible flicker - on a box
   where nothing is happening, which is most of the time, it is the only thing
   moving on screen. So skip the rebuild when the payload is identical to the
   one already rendered. Keyed per panel so one changing does not redraw the
   rest. Panels whose data genuinely ticks (uptime, idle_s) are deliberately
   NOT guarded: those are live readings and freezing them would be a lie. */
const lastPayload = new Map();
function payloadChanged(key, value) {
  const next = JSON.stringify(value);
  if (lastPayload.get(key) === next) return false;
  lastPayload.set(key, next);
  return true;
}

function set(id, text, tone) {
  const node = $(id);
  if (!node) return;
  node.textContent = text === undefined || text === null ? "-" : String(text);
  if (tone !== undefined) {
    node.className = node.className.replace(/\b(good|warn|bad)\b/g, "").trim() +
      (tone ? " " + tone : "");
  }
}

function chip(id, label, value) {
  const node = $(id);
  clear(node);
  node.appendChild(document.createTextNode(label + " "));
  node.appendChild(el("b", null, value));
}

function kv(host, pairs) {
  // Shared by several panels on the 2s tick. The rows are usually identical
  // from one poll to the next, and rebuilding them is a flicker for no gain.
  if (host.id && !payloadChanged("kv:" + host.id, pairs)) return;
  clear(host);
  for (const [key, value, tone] of pairs) {
    host.appendChild(el("dt", null, key));
    host.appendChild(el("dd", tone || null, value));
  }
}

async function api(path, options) {
  const res = await fetch(path, Object.assign({
    headers: { "Content-Type": "application/json" },
  }, options || {}));
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : res.statusText;
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return body;
}

const post = (path, payload) =>
  api(path, { method: "POST", body: JSON.stringify(payload || {}) });

function fmtAge(seconds) {
  if (seconds < 60) return Math.round(seconds) + "s";
  if (seconds < 3600) return Math.round(seconds / 60) + "m";
  return (seconds / 3600).toFixed(1) + "h";
}

/* A segmented control: exactly one button active, the choice readable back out
   of the DOM. Used for the work-mode toggles and the output tabs.

   The active button is marked with a class rather than held in a variable the
   renderer also reads, so there is ONE place the current choice lives. Two
   copies of that state is how a panel ends up rendering the code tab with the
   document tab highlighted. */
function segmented(host, attr, onPick) {
  host.addEventListener("click", (event) => {
    const button = event.target.closest("." + "seg");
    if (!button || !host.contains(button)) return;
    for (const other of host.querySelectorAll(".seg")) {
      other.classList.toggle("active", other === button);
    }
    onPick(button.dataset[attr]);
  });
}

function segmentedValue(host, attr) {
  const active = host.querySelector(".seg.active");
  return active ? active.dataset[attr] : null;
}

function suggestions(host, input, items) {
  clear(host);
  host.appendChild(el("span", "suggest-label", "try:"));
  items.forEach((text) => {
    const button = el("button", "suggest-chip", text.short);
    button.type = "button";
    button.addEventListener("click", () => { input.value = text.value; input.focus(); });
    host.appendChild(button);
  });
}

/* ------------------------------------------------------------------- theme */

/* Dark by default for control room legibility, light for conference room projectors.
   Uses a temporary class to prevent transition flashes when toggling. */
function applyTheme(name) {
  const light = name === "light";
  document.documentElement.classList.add("theme-transitioning");
  document.documentElement.setAttribute("data-theme", light ? "light" : "dark");
  const icon = $("theme-icon");
  if (icon) icon.textContent = light ? "dark mode" : "light mode";
  try { localStorage.setItem("kavach-theme", light ? "light" : "dark"); } catch (e) {}

  // Re-render open view charts so they reflect the new theme immediately
  if (typeof refreshSovereignty === "function" && activeView === "sovereignty") refreshSovereignty();
  if (typeof refreshCorpus === "function" && activeView === "corpus") refreshCorpus();
  if (typeof refreshRegistry === "function" && activeView === "router") refreshRegistry();

  setTimeout(() => {
    document.documentElement.classList.remove("theme-transitioning");
  }, 40);
}

(function initTheme() {
  let stored = null;
  try { stored = localStorage.getItem("kavach-theme"); } catch (e) {}
  const active = stored || "dark";
  const icon = $("theme-icon");
  if (icon) icon.textContent = active === "light" ? "dark mode" : "light mode";
})();

$("theme-toggle").addEventListener("click", () => {
  const now = document.documentElement.getAttribute("data-theme");
  applyTheme(now === "light" ? "dark" : "light");
});

/* ------------------------------------------------------------------ toasts */

/* A transient failure is news; a standing limitation is a fact. They must not
   share a surface, or the fact scrolls away with the news. Hand-written because
   a toast library is a CDN request, and offline_check fails the build on one. */
function toast(message, tone = "bad", ms = 6000) {
  const host = $("toasts");
  if (!host) return;

  /* THE SAME MESSAGE TWICE IS ONE MESSAGE WITH A COUNT.

     Anything that fails per-request fails several times in a row - six
     in-flight polls rejecting when the gateway went away put six identical red
     toasts up the screen, which buries whatever else was being said. Repeats
     bump a counter on the notice already showing and restart its timer, so the
     newest occurrence is still what decides when it disappears. */
  for (const existing of host.children) {
    if (existing.dataset.message === message) {
      const count = Number(existing.dataset.count || "1") + 1;
      existing.dataset.count = String(count);
      const badge = existing.querySelector(".toast-count")
        || existing.appendChild(el("span", "toast-count"));
      badge.textContent = "x" + count;
      clearTimeout(Number(existing.dataset.timer));
      existing.dataset.timer = String(setTimeout(() => existing.remove(), ms));
      return existing;
    }
  }

  const node = el("div", "toast " + tone);
  node.appendChild(el("span", null, message));
  node.dataset.message = message;
  node.dataset.count = "1";
  host.appendChild(node);
  node.dataset.timer = String(setTimeout(() => node.remove(), ms));
  return node;
}

$("honesty-chip").addEventListener("click", () => {
  const modal = $("honesty-modal");
  if (typeof modal.showModal === "function") modal.showModal();
  else modal.setAttribute("open", "");     // very old browsers: still readable
});
$("honesty-close").addEventListener("click", () => {
  const modal = $("honesty-modal");
  if (typeof modal.close === "function") modal.close();
  else modal.removeAttribute("open");
});

// -------------------------------------------------------------- navigation

let activeView = "overview";

/* The open view lives in the URL fragment, so a refresh comes back to the panel
   you were reading rather than to the overview. That matters here because the
   panel is refreshed constantly while rehearsing, and losing your place every
   time is the difference between watching a run and hunting for it. It also
   makes a view linkable and back/forward work, for free.

   A fragment naming a view this page does not have falls back to the overview
   rather than hiding every section - an unknown #hash from a stale bookmark
   must not leave a blank page with nothing to explain it. */
function viewFromHash() {
  const want = location.hash.replace(/^#/, "");
  return document.getElementById("view-" + want) ? want : "overview";
}

function show(view) {
  /* theme.js stamps data-initial-view in <head> so the right panel is painted
     before this script loads - that stops the Overview flashing on reload. But
     the CSS behind it is `display: block !important` on an id selector, which
     outranks .hide, so the attribute has to come OFF the moment JS can do the
     switching properly. Left on, the view it names is pinned open forever and
     every later navigation renders TWO panels stacked on one page. */
  document.documentElement.removeAttribute("data-initial-view");

  activeView = view;
  for (const section of document.querySelectorAll(".view")) {
    section.classList.toggle("hide", section.id !== "view-" + view);
  }
  for (const item of document.querySelectorAll(".nav-item")) {
    item.classList.toggle("active", item.dataset.view === view);
  }
  // Only when it actually differs: assigning an identical hash still fires
  // hashchange in some browsers, which would re-enter show() every click.
  if (viewFromHash() !== view) location.hash = view;
  $("main").scrollTop = 0;
}

for (const item of document.querySelectorAll(".nav-item")) {
  item.addEventListener("click", () => show(item.dataset.view));
}
$("ov-go-runs").addEventListener("click", () => show("runs"));

window.addEventListener("hashchange", () => show(viewFromHash()));
show(viewFromHash());

// ------------------------------------------------- the honesty banner

/* Caveats are keyed so a subsystem can add and clear its own without stepping on
   another's, and each carries the view that owns it so the nav can flag it.
   Rendering only the ones currently true keeps the banner meaningful: a
   permanent block of warnings is wallpaper within about ninety seconds. */
const caveats = new Map();

function caveat(key, text, view) {
  if (text) caveats.set(key, { text, view }); else caveats.delete(key);

  /* Called once per subsystem on every 2s poll, and each call rebuilt the whole
     banner and re-stamped every nav flag. This sits at the top of the page, so
     redrawing it is the most visible churn there is - and between polls it is
     almost always identical, because a subsystem's honesty state changes when
     something happens, not on a timer. */
  if (!payloadChanged("caveats",
      [...caveats.entries()].map(([k, v]) => [k, v.text, v.view]))) return;

  const list = $("honesty-list");
  clear(list);
  for (const entry of caveats.values()) list.appendChild(el("li", null, entry.text));

  /* The chip stays visible for as long as anything is unproven, and carries the
     count. It is deliberately not dismissible - a limitation the operator can
     close is one a judge never sees, and the whole argument of this panel is
     that it does not hide what it cannot back up. The detail moved into a modal
     only so it stops being a band of amber nobody reads. */
  $("honesty-chip").classList.toggle("hide", caveats.size === 0);
  $("honesty-count").textContent = caveats.size;

  const perView = {};
  for (const entry of caveats.values()) {
    if (entry.view) perView[entry.view] = (perView[entry.view] || 0) + 1;
  }
  for (const item of document.querySelectorAll(".nav-item")) {
    const existing = item.querySelector(".nav-flag");
    if (existing) existing.remove();
    const count = perView[item.dataset.view];
    if (count) {
      const layer = item.querySelector(".nav-layer");
      if (layer) layer.remove();
      item.appendChild(el("span", "nav-flag", count));
    }
  }
}

function connection(state, label) {
  $("conn-dot").className = "dot " + state;
  $("conn-label").textContent = label;
}

// ------------------------------------------------------------- shared state

const egressHistory = [];
// Edge-triggered: one toast when the gateway goes away, one when it returns.
let gatewayWasDown = false;
let lastVram = null;

function renderVram(vram, donutId, subId, kvId, whyId) {
  if (!vram) return;
  const used = vram.used_gb || 0;
  const budget = vram.budget_gb || 0;
  const pct = budget > 0 ? Math.round((used / budget) * 100) : 0;
  const tone = pct > 90 ? "red" : pct > 70 ? "amber" : "blue";

  Charts.donut($(donutId), {
    value: used, max: budget,
    label: used.toFixed(1) + " GB",
    sublabel: "of " + budget.toFixed(1) + " usable",
    tone,
  });
  if (subId) {
    set(subId, `${pct}% · slots ${vram.slots_in_use}/${vram.swap_slots}`);
  }
  /* THE ARITHMETIC, on the card that shows the result. "of 4.8 usable" on a
     machine whose owner believes it has 9 GB reads as the workbench losing
     memory - and the only way to find out was to ask. Naming the card is half
     the answer: Windows shows "GPU memory" as dedicated PLUS shared system
     RAM, so a 6 GB card is routinely remembered as 14. */
  if (whyId) {
    const card = (vram.gpu && vram.gpu.name) || "this card";
    set(whyId,
      `${card}: ${vram.vram_gb.toFixed(1)} GB card `
      + `- ${vram.reserve_gb.toFixed(1)} reserved for the display `
      + `- ${vram.kv_cache_gb.toFixed(1)} KV cache = ${budget.toFixed(1)} GB usable`);
  }
  if (kvId) {
    kv($(kvId), [
      ["card", vram.vram_gb.toFixed(1) + " GB"],
      ["fragmentation reserve", vram.reserve_gb.toFixed(1) + " GB"],
      ["KV cache charge", vram.kv_cache_gb.toFixed(2) + " GB"],
      ["free", vram.free_gb.toFixed(2) + " GB"],
      ["swap slots", `${vram.slots_in_use} / ${vram.swap_slots}`],
    ]);
  }
}

function residentList(host, vram) {
  const resident = (vram && vram.resident) || [];
  /* Only the id and the pinned flag are drawn here. The payload also carries
     idle_s, which ticks every poll - keying off the whole object rebuilt a
     byte-identical list every 2s, which is the flicker. */
  if (!payloadChanged("resident:" + host.id, resident.map((e) => (
    typeof e === "string" ? [e, false] : [e.model_id || e.id, !!e.pinned])))) return;
  clear(host);
  if (!resident.length) {
    host.appendChild(el("li", "muted", "nothing resident"));
    return;
  }
  for (const entry of resident) {
    const id = typeof entry === "string" ? entry : entry.model_id || entry.id;
    const li = el("li");
    li.appendChild(el("span", "ellipsis mono", id));
    if (typeof entry === "object" && entry.pinned) {
      li.appendChild(el("span", "badge blue", "pinned"));
    }
    host.appendChild(li);
  }
}

// ------------------------------------------------------ sovereignty (L0)

async function refreshSovereignty() {
  let body;
  try {
    body = await api("/api/sovereignty");
  } catch (err) {
    connection("bad", "gateway unreachable");
    /* A gateway that is down is an EVENT, not a standing limitation, so it goes
       to a toast rather than into the cannot-prove list - and only once per
       outage, because a 2s poller would otherwise stack sixty of them a minute.
       The connection chip stays red for as long as it is actually down, which
       is the part that must not disappear. */
    if (!gatewayWasDown) {
      gatewayWasDown = true;
      toast("The gateway is not answering: " + err.message, "bad", 8000);
    }
    return;
  }
  if (gatewayWasDown) {
    gatewayWasDown = false;
    toast("Gateway is answering again.", "good", 3500);
  }
  connection("ok", "connected");

  // ---- top bar
  chip("chip-profile", "profile", body.profile.name);
  const backend = body.models.backend.backend || "?";
  chip("chip-backend", "backend", backend);

  /* The caveat itself is raised further down, from `body.models.backend`,
     which is the authority on which backend is bound. Raising a second one
     under the same key from here only meant whichever ran last won - and the
     two said slightly different things. */
  chip("chip-uptime", "uptime", fmtAge(body.uptime_s));
  chip("chip-vram", "vram",
    `${(body.vram.used_gb || 0).toFixed(1)}/${(body.vram.budget_gb || 0).toFixed(1)}G`);

  // ---- KPIs
  set("kpi-egress", body.external_packets, body.external_packets ? "warn" : "good");
  set("kpi-audit", body.audit.records);
  set("kpi-audit-note", body.audit.verified ? "chain intact" : "CHAIN BROKEN");
  $("kpi-audit-note").className = "stat-note " + (body.audit.verified ? "ok" : "bad");
  set("kpi-models", body.models.registered);
  const provenance = body.models.provenance || {};
  set("kpi-models-note", Object.entries(provenance)
    .map(([k, v]) => `${v} ${k.toLowerCase()}`).join(", ") || "-");

  // ---- sovereignty view
  set("sv-external", body.external_packets, body.external_packets ? "warn" : "good");
  set("sv-blocked", body.blocked_packets);

  const sentinel = body.sentinel;
  const healthy = sentinel.healthy;
  // The mode is the provenance of the number. A judge should be able to read
  // where it came from without knowing what "iphlpapi" is, and the three
  // sources are NOT equally strong: only nftables is a kernel that refused.
  const SOURCES = {
    nftables: "the Linux firewall counter",
    proc: "the Linux kernel counter (/proc/net/dev)",
    iphlpapi: "the Windows kernel counter (GetIfTable2)",
  };
  const source = SOURCES[sentinel.mode] || sentinel.mode || "this box";
  /* The note carries the distinction the headline number cannot. A reader
     who takes 4,524 to mean "Kavach sent 4,524 packets out" has been misled by
     a panel that is otherwise about not misleading them. */
  set("kpi-egress-note", !healthy
    ? "NOT MEASURED - this computer cannot count them"
    : body.external_packets
      ? "every app on this PC, not just Kavach"
      : "measured - nothing has left");
  $("kpi-egress-note").className = "stat-note " + (healthy ? "muted" : "warn");
  set("sv-external-note", healthy
    ? "measured from " + source
    : "the sentinel cannot measure here - this is not a measurement");

  const badge = $("sv-sentinel-badge");
  badge.className = "badge " + (healthy ? "good" : "warn");
  badge.textContent = healthy ? "HEALTHY" : "DEGRADED";
  kv($("sv-sentinel"), [
    ["counted from", healthy ? source : "-"],
    ["external packets", sentinel.external_packets],
    ["blocked", sentinel.blocked_packets],
  ]);
  $("sv-sentinel-detail").textContent = sentinel.detail || "";

  // A box that cannot count its own egress cannot support the headline claim,
  // so that gap belongs in the banner, not only in this panel.
  caveat("sentinel", healthy ? null :
    "This computer cannot count network traffic - " + (sentinel.detail || "it cannot measure here") +
    " The packet counters are not a measurement.", "sovereignty");

  // Measuring is not the same as enforcing. Every mode except nftables counts
  // what left the box without any kernel having been asked to stop it, and the
  // blocked figure is then our own bookkeeping. Say so where the number is.
  caveat("egress-enforcement", (healthy && sentinel.mode !== "nftables")
    ? "Egress is being MEASURED but not ENFORCED on this box: " + sentinel.external_packets +
      " packets were counted leaving it, and the blocked count is Kavach refusing " +
      "to make requests, not a firewall dropping them. The nftables demo machine enforces."
    : null, "sovereignty");

  const audit = body.audit;
  const chainBadge = $("sv-chain-badge");
  chainBadge.className = "badge " + (audit.verified ? "good" : "bad");
  chainBadge.textContent = audit.verified
    ? "INTACT · " + audit.records + " records" : "BROKEN";
  $("sv-chain-detail").textContent = audit.detail || "";
  caveat("audit", audit.verified ? null :
    "THE AUDIT CHAIN DOES NOT VERIFY: " + audit.detail, "sovereignty");

  if (body.models.backend.simulated) {
    caveat("backend",
      "NO AI MODEL IS INSTALLED. An offline stand-in is answering instead, and " +
      "everything it writes is a placeholder marked [SIMULATED] - not a real " +
      "answer. To fix: put a .gguf model file in the models folder and restart.",
      "router");
  } else {
    caveat("backend", null);
  }

  const simulated = provenance.SIMULATED || 0;
  caveat("weights", simulated ? simulated +
    " AI model(s) are configured but their files are not on this computer. " +
    "They are listed so you can see what this box is set up for; none of them " +
    "can run until the .gguf file is in the models folder. To fix: download " +
    "the weights on a connected machine, copy them across, and restart."
    : null, "router");

  lastVram = body.vram;
  renderVram(body.vram, "ov-vram-donut", "ov-vram-sub", null, "ov-vram-why");
  renderVram(body.vram, "rt-vram-donut", "rt-vram-sub", "rt-vram-kv", "rt-vram-why");
  residentList($("ov-resident"), body.vram);

  egressHistory.push(body.blocked_packets || 0);
  if (egressHistory.length > SPARK_POINTS) egressHistory.shift();
  Charts.sparkline($("ov-spark"), egressHistory, { tone: "amber" });
}

/* ------------------------------------------------ audit console (L0 / L6) */

/* An operations console, not an activity feed.

   Every column is a field the chain actually stores, and every value is a word
   this system already uses - the projection lives in `audit/console.py` so the
   API and this panel cannot disagree about what an event means. Nothing here
   invents a friendlier name for a call site.

   STATE IS NEVER COLOUR ALONE. Each decision and result carries a glyph and
   the word itself, with colour as the third cue. A projector washes out the
   tint, a greyscale screenshot loses it entirely, and a colour-blind reader
   never had it - in all three cases the word and the glyph still read. */

const AUDIT_MARK = {
  ALLOWED:           ["good", "+"],
  DENIED:            ["bad", "x"],
  APPROVAL_REQUIRED: ["warn", "?"],
  BLOCKED:           ["bad", "#"],
  INCONCLUSIVE:      ["warn", "~"],
  OK:                ["good", "✓"],
  PASSED:            ["good", "✓"],
  ACCEPTED:          ["good", "✓"],
  DONE:              ["good", "✓"],
  VERIFIED:          ["good", "✓"],
  FAILED:            ["bad", "✕"],
  REJECTED:          ["bad", "✕"],
  DIGEST_MISMATCH:   ["bad", "✕"],
  NO_VIABLE_MODEL:   ["bad", "✕"],
  DIVERGED:          ["warn", "!"],
  SIMULATED:         ["warn", "!"],
  UNVERIFIED:        ["warn", "!"],
  REPAIRED:          ["warn", "!"],
  REPLANNED:         ["warn", "!"],
  BUDGET_EXHAUSTED:  ["bad", "✕"],
  PENDING:           ["warn", "…"],
  AWAITING_APPROVAL: ["warn", "…"],
  RUNNING:           ["blue", "▶"],
};

/* Which rows each filter keeps. Defined as predicates over the projected row
   rather than over the raw event name, so a new event that lands in one of
   these categories is included the day it is added. */
const AUDIT_FILTERS = {
  all: () => true,
  gate: (r) => r.decision !== "-",
  denied: (r) => r.decision === "DENIED" || r.result === "BLOCKED"
              || r.result === "REJECTED",
  failed: (r) => r.result === "FAILED" || r.result === "DIGEST_MISMATCH"
              || r.result === "NO_VIABLE_MODEL" || r.result === "BUDGET_EXHAUSTED",
};

let auditRows = [];
let auditOpen = null;      // seq of the expanded row, so it survives a refresh

function verdictCell(value) {
  const cell = el("td");
  if (!value || value === "-") {
    cell.appendChild(el("span", "muted", "-"));
    return cell;
  }
  const mark = AUDIT_MARK[value] || ["", "·"];
  const tag = el("span", "verdict " + mark[0]);
  tag.appendChild(el("span", "verdict-glyph", mark[1]));
  tag.appendChild(el("span", null, value));
  cell.appendChild(tag);
  return cell;
}

/* ISO timestamps are what the chain stores; the console shows the clock time
   and keeps the full value in the detail. A table of 40 identical date
   prefixes is 40 columns of noise. */
function auditClock(ts) {
  const at = String(ts || "");
  const t = at.indexOf("T");
  return t === -1 ? at : at.slice(t + 1, t + 13);
}

function renderAuditConsole() {
  const host = $("audit-rows");
  clear(host);

  const keep = AUDIT_FILTERS[segmentedValue($("audit-filter"), "audit") || "all"];
  const rows = auditRows.filter(keep);

  if (!rows.length) {
    const empty = el("tr");
    const cell = el("td", "empty", "no activity matches this filter");
    cell.colSpan = 8;
    empty.appendChild(cell);
    host.appendChild(empty);
    return;
  }

  for (const row of rows) {
    const tr = el("tr", "console-row" + (row.mapped ? "" : " unmapped"));
    tr.tabIndex = 0;
    tr.appendChild(el("td", "mono muted col-seq", String(row.seq).padStart(4, "0")));
    tr.appendChild(el("td", "mono col-time", auditClock(row.ts)));

    const who = el("td");
    who.appendChild(el("span", "mono", row.actor));
    // The role the POLICY ENGINE evaluated against - not a display name. A row
    // without one shows nothing rather than a default that would read as a
    // claim about a decision the chain never recorded.
    if (row.role && row.role !== "-") {
      who.appendChild(el("span", "role-tag", row.role));
    }
    tr.appendChild(who);

    const op = el("td");
    op.appendChild(el("span", "op mono", row.operation));
    if (row.target) op.appendChild(el("span", "op-target mono", row.target));
    if (!row.mapped) op.appendChild(el("span", "badge warn", "unmapped event"));
    tr.appendChild(op);

    tr.appendChild(el("td", "mono label-cell", row.label || "-"));
    tr.appendChild(el("td", "mono muted", row.side_effect || "-"));
    tr.appendChild(verdictCell(row.decision));
    tr.appendChild(verdictCell(row.result));

    const toggle = () => {
      auditOpen = auditOpen === row.seq ? null : row.seq;
      renderAuditConsole();
    };
    tr.addEventListener("click", toggle);
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
    host.appendChild(tr);

    if (auditOpen === row.seq) {
      tr.classList.add("hl");
      host.appendChild(auditDetail(row));
    }
  }
}

function auditDetail(row) {
  const tr = el("tr", "console-detail");
  const cell = el("td");
  cell.colSpan = 8;

  const facts = el("dl", "kv");
  const pairs = [
    ["event", row.event],
    ["timestamp", row.ts],
    ["run", row.run || "-"],
    ["step", row.step === null || row.step === undefined ? "-" : String(row.step)],
    ["egress observed", String(row.egress_observed || 0)],
    ["record digest", row.sha256],
    ["previous digest", row.prev_sha256],
  ];
  for (const [key, value] of pairs) {
    facts.appendChild(el("dt", null, key));
    facts.appendChild(el("dd", "mono", value));
  }
  cell.appendChild(facts);

  // The payload verbatim. This is the audited content: summarising it here
  // would put a paraphrase next to a digest that covers the original.
  cell.appendChild(el("h4", "sec", "recorded payload"));
  cell.appendChild(el("pre", "code", JSON.stringify(row.payload, null, 2)));

  tr.appendChild(cell);
  return tr;
}

segmented($("audit-filter"), "audit", renderAuditConsole);

async function refreshAuditTail() {
  let body;
  /* The PROJECTED tail: one row per record with the call site, the policy
     context and the outcome already resolved server-side. The panel does not
     interpret event names itself - two interpretations of one record is how a
     console starts disagreeing with the log it displays. */
  try { body = await api("/api/audit/console?n=60"); } catch (e) { return; }
  if (!payloadChanged("audit", body)) return;

  auditRows = body.rows;
  renderAuditConsole();

  /* An event this build has no mapping for is shown by its raw name AND said
     out loud here. A console that renders unknown events as ordinary rows is
     one that quietly stops describing the system. */
  const unmapped = $("audit-unmapped");
  unmapped.classList.toggle("hide", !body.note);
  if (body.note) unmapped.textContent = body.note;
  caveat("audit-mapping", body.note, "sovereignty");

  // The chain, drawn as a chain. Each block carries its own digest and points at
  // the one before it; the point of the picture is that they are linked, so
  // editing any record breaks every block to its right.
  const chain = $("sv-chain");
  clear(chain);
  const blocks = body.rows.slice(0, 7).reverse();
  blocks.forEach((rec, index) => {
    if (index) chain.appendChild(el("div", "chain-link"));
    const block = el("div", "chain-block" + (index === blocks.length - 1 ? " head" : ""));
    block.appendChild(el("div", null, "#" + rec.seq));
    block.appendChild(el("div", "muted", (rec.sha256 || "").slice(0, 8)));
    block.title = rec.event + " ← " + (rec.prev_sha256 || "").slice(0, 12);
    chain.appendChild(block);
  });
}

$("btn-canary").addEventListener("click", async () => {
  const button = $("btn-canary");
  const out = $("canary-verdict");
  const probes = $("canary-probes");
  button.disabled = true;
  out.classList.remove("hide");
  out.className = "notice";
  out.textContent = "attempting DNS and TCP egress ...";
  clear(probes);

  try {
    const body = await post("/api/sovereignty/canary", {});
    // BLOCKED is proof. INCONCLUSIVE is not a softer kind of proof - a timeout
    // and an unplugged cable are indistinguishable - so it renders amber and
    // says why, rather than a tick that implies enforcement was demonstrated.
    // `ok` alone is not enough: the API reports ok AND conclusive separately
    // precisely so a UI cannot collapse them.
    const verdict = !body.conclusive ? "INCONCLUSIVE"
                  : body.ok ? "BLOCKED" : "ALLOWED";
    const tone = verdict === "BLOCKED" ? "good"
               : verdict === "ALLOWED" ? "bad" : "warn";
    out.className = "notice " + tone;
    /* NOT ARMED is its own word, and it is the state most machines are in.
       "INCONCLUSIVE" describes a probe that came back ambiguous; this box never
       even had a policy that could have refused one, which is a different fact
       and a different thing to do about it. Collapsing the two is what let the
       panel imply a firewall existed. */
    const armed = body.enforcement || {};
    const label = (!armed.present && verdict !== "ALLOWED") ? "NOT ARMED" : verdict;
    out.textContent = label + " — " + (body.headline || "");

    for (const probe of body.probes || []) {
      const line = el("div", "logline");
      line.appendChild(el("span", "badge " +
        (probe.verdict === "BLOCKED" ? "good" : probe.verdict === "ALLOWED" ? "bad" : "warn"),
        probe.verdict));
      line.appendChild(el("span", "mono", probe.proto + " " + probe.target));
      line.appendChild(el("span", "ellipsis muted tiny", probe.detail || ""));
      probes.appendChild(line);
    }

    /* The sentinel already says this computer cannot prove traffic is blocked.
       A failed canary is the SAME fact arriving by a second route, and listing
       it twice makes the panel look like it has two problems when it has one.
       So the canary only speaks when it has something the sentinel does not:
       that something actually got out. */
    caveat("canary", verdict === "ALLOWED"
      ? "A test message reached the internet from this computer. Traffic is NOT "
        + "being blocked - stop and investigate before using this for real work."
      : null, "sovereignty");
  } catch (err) {
    out.className = "notice bad";
    out.textContent = "canary failed: " + err.message;
  } finally {
    button.disabled = false;
    refreshAuditTail();
  }
});

$("btn-verify").addEventListener("click", async () => {
  const out = $("canary-verdict");
  out.classList.remove("hide");
  try {
    const body = await api("/api/audit/verify");
    out.className = "notice " + (body.ok ? "good" : "bad");
    out.textContent = (body.ok ? "chain intact — " : "CHAIN BROKEN — ") + body.detail;
  } catch (err) {
    out.className = "notice bad";
    out.textContent = err.message;
  }
});

// ------------------------------------------------------- fabric + router

async function refreshRegistry() {
  let body;
  try { body = await api("/api/models"); } catch (e) { return; }

  const resident = new Set(((lastVram && lastVram.resident) || []).map(
    (r) => (typeof r === "string" ? r : r.model_id || r.id)));

  // Residency comes from the sovereignty poll, not this payload, so it has to
  // be part of the signature or a model going resident would never repaint.
  // The filter checkbox is part of what is rendered, so it belongs in the
  // signature - otherwise ticking it repaints nothing until the next poll
  // happens to differ.
  if (!payloadChanged("models",
      [body, [...resident].sort(), $("rt-show-all").checked])) return;

  const host = $("rt-registry");
  clear(host);

  /* Only what this computer can actually run, unless asked otherwise.
     Fourteen rows of which seven say "file not on this PC" buries the seven
     that work - and the question this table answers is "what can I use", not
     "what is in the config file". */
  const showAll = $("rt-show-all").checked;
  /* `file_present`, not the provenance enum. A missing file is SIMULATED on
     the mock backend and UNVERIFIED on llamacpp - and UNVERIFIED is ALSO what
     a file that is present but has no recorded digest reports. Filtering on
     the enum showed all fourteen rows the moment the backend changed. */
  const usable = (m) => m.file_present;
  const shown = showAll ? body.models : body.models.filter(usable);

  if (!shown.length) {
    const row = el("tr");
    const cell = el("td", "empty",
      "No model file is on this computer yet. Add one above, or tick the box to "
      + "see what is configured.");
    cell.colSpan = 7;
    row.appendChild(cell);
    host.appendChild(row);
  }

  for (const model of shown) {
    const tr = el("tr", resident.has(model.id) ? "hl" : null);
    tr.appendChild(el("td", "mono", model.id));
    tr.appendChild(el("td", "muted", model.role));
    tr.appendChild(el("td", "num", model.tier));
    tr.appendChild(el("td", "num", model.vram_gb.toFixed(1)));
    tr.appendChild(el("td", "num", (model.context / 1024).toFixed(0) + "k"));

    const provenance = el("td");
    const [tone, label] = WEIGHTS_STATE[model.provenance]
      || ["warn", model.provenance];
    const mark = el("span", "badge " + tone, label);
    // The enum value itself, for anyone matching a row against the audit
    // chain. Renamed on screen only.
    mark.title = model.provenance;
    provenance.appendChild(mark);
    tr.appendChild(provenance);

    const res = el("td");
    if (resident.has(model.id)) res.appendChild(el("span", "badge blue", "resident"));
    tr.appendChild(res);

    host.appendChild(tr);
  }
  /* "14 registered, 14 SIMULATED" told a reader almost nothing. The number
     that matters is how many can actually run. */
  const installed = body.models.filter((m) => m.file_present).length;
  const absent = body.count - installed;
  set("rt-registry-sub", installed + " ready to use"
    + (absent ? "  ·  " + absent + " configured with no file here" : ""));
}

$("form-route").addEventListener("submit", async (event) => {
  event.preventDefault();
  const out = $("route-out");
  clear(out);

  /* An empty box is a user who has not typed yet, not an error. Sending it
     anyway returned FastAPI's raw validation JSON - "string_too_short",
     "loc":["body","prompt"] - which tells a reader nothing they can act on. */
  const prompt = $("route-prompt").value.trim();
  if (!prompt) {
    out.appendChild(el("div", "notice", "Type a task above to see how it would be routed."));
    $("route-prompt").focus();
    return;
  }

  out.appendChild(el("p", "muted tiny", "scoring stages 1-3, no VRAM touched..."));
  try {
    renderRoute(await post("/api/route/explain", {
      prompt,
      has_images: $("route-images").checked,
      /* Auto sends nothing and the classifier decides, exactly as before the
         toggle existed. The other two pin the class, which is the fastest way
         to show a room that the same sentence routes to a different model
         depending on what is being asked for. */
      forced_class: FORCED_CLASS[segmentedValue($("route-mode"), "mode")] || null,
    }));
  } catch (err) {
    clear(out);
    out.appendChild(el("div", "notice bad", err.message));
  }
});

/* The mode toggle in UI terms -> the router's own task class. Mirrors
   WorkMode.forced_class in types.py; auto deliberately maps to nothing, because
   "let the classifier decide" is the absence of a forced class, not a class. */
const FORCED_CLASS = {
  auto: null,
  code: "code_generation",
  document: "official_drafting",
};

$("rt-show-all").addEventListener("change", refreshRegistry);

segmented($("route-mode"), "mode", () => {
  // Re-explain immediately on a flip. The whole value of this control is the
  // side-by-side comparison, and making the operator press Explain again turns
  // a one-second demonstration into three clicks.
  if ($("route-prompt").value.trim()) {
    $("form-route").dispatchEvent(new Event("submit", { cancelable: true }));
  }
});

function renderRoute(body) {
  const out = $("route-out");
  clear(out);

  const decision = body.decision;
  const head = el("div", "inline");
  head.appendChild(el("span", "badge " + (decision.chosen ? "good" : "bad"),
    decision.chosen || "no viable model"));
  head.appendChild(el("span", "badge blue", decision.class));
  head.appendChild(el("span", "badge", "p=" + decision.p));
  head.appendChild(el("span", "badge", decision.elapsed_ms + " ms"));
  out.appendChild(head);
  out.appendChild(el("p", "tiny muted", decision.reason));

  const survivors = decision.candidates.filter((c) => !c.eliminated_by);

  if (survivors.length) {
    out.appendChild(el("h3", "sec", "score"));
    const rows = el("div");
    Charts.barRows(rows, survivors.map((c) => ({
      label: c.model,
      value: c.score === null ? 0 : c.score,
      tone: c.model === decision.chosen ? "green" : "blue",
    })));
    out.appendChild(rows);

    // The winner's terms, drawn as a diverging bar. Rendering the score as one
    // number hides the entire argument: quality pushes right, the costs of
    // loading and running push left, and the net is what decided it.
    const winner = survivors.find((c) => c.model === decision.chosen) || survivors[0];
    if (winner && winner.terms) {
      out.appendChild(el("h3", "sec", "why " + winner.model + " won"));
      const terms = el("div");
      Charts.divergingBars(terms, winner.terms);
      out.appendChild(terms);
    }
  }

  // Stage 1 removals are shown with their reasons. A model with no vision tower
  // is not a bad choice for a scan, it is not a choice - and the panel has to
  // show it was excluded rather than merely out-scored.
  const eliminated = decision.candidates.filter((c) => c.eliminated_by);
  if (eliminated.length) {
    out.appendChild(el("h3", "sec",
      `eliminated before scoring (${eliminated.length} of ${decision.candidates.length})`));
    const list = el("ul", "plain");
    for (const cand of eliminated) {
      const li = el("li");
      li.appendChild(el("span", "mono", cand.model));
      li.appendChild(el("span", "ellipsis muted tiny", cand.eliminated_by));
      list.appendChild(li);
    }
    out.appendChild(list);
  }

  // The quality column is what the router ranks on. Until the eval harness has
  // run against real weights those numbers are seeds, and the page says so
  // before anyone has to ask where 0.84 came from.
  const provenance = body.quality_provenance || "seeded";
  const badge = $("rt-provenance");
  badge.className = "badge " + (provenance === "measured" ? "good" : "warn");
  badge.textContent = "quality: " + provenance;
  caveat("quality", provenance === "measured" ? null :
    "The model scores used to pick a model are estimates, not measurements (" +
    provenance + "). The router ranks models on numbers no eval harness has " +
    "produced yet.", "router");
  if (body.quality_warning) {
    out.appendChild(el("div", "notice warn", body.quality_warning));
  }
}

// ---------------------------------------------------------------- runs (L5)

let source = null;       // EventSource
let pollTimer = null;    // fallback poller

function streamState(tone, text) {
  const node = $("run-stream-state");
  node.className = "badge " + (tone || "");
  node.textContent = text;
}

function stopStream() {
  if (source) { source.close(); source = null; }
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

/* Poll the timeline instead of streaming. Not a silent downgrade: the header
   says "polling" for as long as this is what is happening, because a panel that
   looks live and is not is worse than one that admits it refreshes every 2s. */
function startPolling(runId, why) {
  if (pollTimer) return;
  streamState("warn", "checking every 2s - " + why);
  pollTimer = setInterval(async () => {
    try {
      const timeline = await api("/api/runs/" + runId + "/timeline");
      renderRun(timeline);
      if (["done", "failed", "budget_exhausted", "awaiting_approval"]
          .includes(timeline.status)) {
        clearInterval(pollTimer); pollTimer = null;
        streamState("", timeline.status);
      }
    } catch (e) { /* the sovereignty poller reports a gateway that is down */ }
  }, POLL_MS);
}

function watch(runId) {
  stopStream();

  if (typeof EventSource === "undefined") {
    startPolling(runId, "this browser has no EventSource");
    return;
  }

  source = new EventSource("/api/runs/" + runId + "/stream");
  streamState("good", "watching live");

  source.addEventListener("snapshot", (e) => renderRun(JSON.parse(e.data)));

  // Every run event is a hint to re-read the server's state rather than a patch
  // to apply locally. Slightly more traffic; no way for the panel to drift out
  // of agreement with the run it is displaying.
  const refresh = async () => {
    try { renderRun(await api("/api/runs/" + runId + "/timeline")); } catch (e) {}
  };
  for (const kind of ["step_started", "step_passed", "step_failed", "replanned",
                      "awaiting_approval", "budget_exhausted", "done", "failed"]) {
    source.addEventListener(kind, refresh);
  }

  source.addEventListener("dropped", (e) => {
    const detail = JSON.parse(e.data).detail || {};
    streamState("warn", "fell behind by " + detail.count + " - caught up");
    refresh();
  });

  source.addEventListener("error", (e) => {
    // A run whose thread died. Distinct from a transport error, which arrives
    // via onerror below and carries no data.
    if (!e.data) return;
    const detail = JSON.parse(e.data).detail || {};
    $("run-error").classList.remove("hide");
    $("run-error").textContent = "the run thread failed: " + (detail.reason || "");
  });

  source.addEventListener("stream_end", () => {
    stopStream();
    refresh();
    api("/api/runs/" + runId + "/timeline")
      .then((t) => streamState(t.status === "done" ? "good" : "", t.status))
      .catch(() => streamState("", "stream closed"));
  });

  source.onerror = () => {
    // The browser retries on its own, but if the connection never establishes
    // the panel must not sit on a spinner. Fall back, and say why.
    if (source && source.readyState === EventSource.CLOSED) {
      stopStream();
      startPolling(runId, "the event stream closed");
    }
  };
}

const STEP_MARK = {
  pending: ["", "·"], running: ["running", "▶"], passed: ["passed", "✓"],
  failed: ["failed", "✕"], awaiting_approval: ["awaiting_approval", "?"],
  skipped: ["", "–"],
};

function renderRun(timeline) {
  // A task exists now, so the panel stops being a prompt and starts being a
  // report: the idle line goes, the identity and the two badges appear.
  $("run-idle").classList.add("hide");
  $("run-id").classList.remove("hide");
  $("run-status").classList.remove("hide");
  $("run-label").classList.remove("hide");
  $("run-id").textContent = timeline.run_id + "  ·  " + timeline.goal;

  const status = timeline.status;
  const tone = status === "done" ? "good"
             : status === "failed" || status === "budget_exhausted" ? "bad"
             : status === "awaiting_approval" ? "warn" : "blue";
  const statusBadge = $("run-status");
  statusBadge.className = "badge " + tone;
  statusBadge.textContent = status;
  $("run-label").textContent = timeline.label;

  /* Which toggle produced THIS run, not which toggle is currently selected. A
     historical run opened from the list was planned under whatever mode was
     chosen then, and showing the live control's position over it would
     misattribute the result. */
  const mode = timeline.mode || "auto";
  const modeBadge = $("out-mode");
  modeBadge.className = "badge " + (mode === "auto" ? "" : "blue");
  modeBadge.textContent = mode;

  refreshOutputs(timeline);
  refreshRouting(timeline);

  const b = timeline.budget || {};
  kv($("run-budget"), [
    ["steps", `${b.steps_used} / ${b.max_steps}`],
    ["tool calls", `${b.tool_calls_used} / ${b.max_tool_calls}`],
    ["wall clock", `${b.elapsed_s || 0}s / ${b.max_wall_s}s`],
  ]);

  $("run-error").classList.toggle("hide", !timeline.error);
  if (timeline.error) $("run-error").textContent = timeline.error;

  const list = $("steps");
  clear(list);
  for (const step of timeline.steps) {
    const [cls, mark] = STEP_MARK[step.status] || ["", "·"];
    const li = el("li", "step " + cls);
    li.appendChild(el("div", "step-node", mark));

    const head = el("div", "step-head");
    head.appendChild(el("span", "step-intent", step.intent));
    if (step.tool) head.appendChild(el("span", "badge blue", step.tool));
    if (step.attempts > 1) head.appendChild(el("span", "badge warn", "x" + step.attempts));
    if (step.status === "awaiting_approval") {
      const button = el("button", "tiny", "Approve");
      button.addEventListener("click", () => approve(timeline.run_id, step.idx));
      head.appendChild(button);
    } else if (step.requires_approval) {
      head.appendChild(el("span", "badge warn", "needs approval"));
    }
    li.appendChild(head);

    // The acceptance test is the load-bearing part of a plan, so it is on screen
    // next to the step rather than behind a click. A plan without one is a list
    // of intentions, and this is where a viewer can see there is one.
    if (step.acceptance) {
      const accept = el("div", "step-accept");
      accept.appendChild(el("b", null, "accept: "));
      accept.appendChild(document.createTextNode(step.acceptance));
      li.appendChild(accept);
    }

    const obs = (step.observations || [])[step.observations.length - 1];
    if (obs && (obs.stdout || obs.stderr)) {
      li.appendChild(el("div", "step-out", (obs.stderr || obs.stdout).slice(0, 900)));
    }
    list.appendChild(li);
  }
  if (!timeline.steps.length) {
    list.appendChild(el("div", "empty", "no steps"));
  }

  const artifacts = $("artifacts");
  clear(artifacts);
  if (!timeline.artifacts.length) {
    artifacts.appendChild(el("li", "muted", "none yet"));
  }
  for (const art of timeline.artifacts) {
    const li = el("li");
    /* The name is a link to the bytes. A panel that can name an approval note
       and show its digest, but cannot open it, leaves "real deliverables rather
       than chat replies" as a claim - the reader has to go digging in var/ to
       check. Same origin, so this downloads from this box and nowhere else. */
    const href = "/api/runs/" + encodeURIComponent(timeline.run_id)
      + "/artifacts/" + art.path.split(/[\/]/).map(encodeURIComponent).join("/");
    const link = el("a", "ellipsis mono", art.path);
    link.href = href;
    link.download = art.path.split(/[\/]/).pop();
    link.title = "download " + art.path;
    li.appendChild(link);
    li.appendChild(el("span", "muted tiny", (art.sha256 || "").slice(0, 10)));
    if (art.approved_by) li.appendChild(el("span", "badge good", "signed"));
    artifacts.appendChild(li);
  }
}

/* ------------------------------------------------- run output (code | doc) */

/* Two toggles over one run, because they are two different ways of reading.
   Code is read as source next to what it printed when it ran; a document is
   read by opening it. The server decides which side every artifact belongs to
   (agent/outputs.py) - the split is a property of the run, not a guess this
   file makes from a filename, so the API and the panel can never disagree
   about where a deliverable went.

   An empty side always renders the server's `note` rather than nothing. A blank
   panel reads as "still loading", and the most important case here - a run
   started in Code mode that produced no code - is exactly the one where that
   would hide a failure. */

let outputsInFlight = false;


function refreshOutputs(timeline) {
  /* Reading a .docx back off disk on every SSE event, four times a run, for a
     panel whose contents did not change, would also reset the reader's scroll
     mid-run. Re-read only when something that can affect the output moved: the
     status, the artifact set, or a code.exec step's state. */
  const signature = [
    timeline.status,
    (timeline.artifacts || []).map((a) => a.path + ":" + (a.sha256 || "") +
      ":" + (a.approved_by || "")).join(","),
    (timeline.steps || []).filter((s) => s.tool === "code.exec")
      .map((s) => s.idx + ":" + s.status + ":" + (s.observations || []).length).join(","),
  ].join("|");
  if (!payloadChanged("outputs:" + timeline.run_id, signature)) return;
  if (outputsInFlight) return;

  outputsInFlight = true;
  api("/api/runs/" + encodeURIComponent(timeline.run_id) + "/outputs")
    .then(renderOutputs)
    .catch(() => {
      /* The panel must not keep showing the previous run's deliverables while
         this one's could not be fetched. Say what happened instead. */
      for (const pair of [[$("out-code"), "code"], [$("out-document"), "documents"]]) {
        clear(pair[0]);
        pair[0].appendChild(el("div", "notice bad",
          "the " + pair[1] + " for this run could not be read from the gateway"));
      }
    })
    .finally(() => { outputsInFlight = false; });
}

/* WHICH MODEL DID WHAT, in the order the system actually did it.

   The first question anyone asks watching a run is whether one model is doing
   everything or a different one is picked per task, and the page could not
   answer it. Every routing decision is already recorded against the run; this
   reads them back oldest-first, because it is a story: the prompt was
   classified, a model was chosen for that class, then each step was judged and
   routed again on its own merits.

   Polled on the same signature rule as the outputs panel - a run with four
   steps makes a handful of decisions, and re-fetching them on every SSE frame
   would reset the reader's scroll for no new information. */
let routingInFlight = false;

function refreshRouting(timeline) {
  const signature = [
    timeline.status,
    (timeline.steps || []).map((s) => s.idx + ":" + s.status).join(","),
  ].join("|");
  if (!payloadChanged("routing:" + timeline.run_id, signature)) return;
  if (routingInFlight) return;

  routingInFlight = true;
  api("/api/runs/" + encodeURIComponent(timeline.run_id) + "/routing")
    .then(renderRouting)
    .catch(() => {
      set("routing-summary", "unavailable");
      clear($("routing-steps"));
      $("routing-steps").appendChild(el("div", "notice warn",
        "the routing record for this run could not be read from the gateway"));
    })
    .finally(() => { routingInFlight = false; });
}

function renderRouting(body) {
  const host = $("routing-steps");
  clear(host);
  set("routing-how", body.how_it_works || "");

  const decisions = body.decisions || [];
  if (!decisions.length) {
    set("routing-summary", "not routed yet");
    host.appendChild(el("div", "empty",
      "Nothing has been routed yet. The moment you press Run, the prompt goes "
      + "to the router and the model it picks appears here."));
    return;
  }

  /* The answer to the question, in words, before any table. "one model for all
     4 decisions" and "2 models across 6 decisions" are different facts about
     the system and a reader should not have to count rows to tell them apart. */
  const used = body.models_used || [];
  set("routing-summary", body.one_model_for_everything
    ? "one model, " + decisions.length + " decision(s)"
    : used.length + " models, " + decisions.length + " decision(s)");

  for (const d of decisions) {
    const row = el("div", "route-row");

    const stage = el("span", "route-stage", d.stage || "-");
    stage.title = d.stage === "plan"
      ? "the decision that produced the whole plan"
      : "the decision made while judging this step";
    row.appendChild(stage);

    /* The class the classifier detected, then the model it led to. The arrow
       is the whole point of the card: the task decides the model, not the
       other way round. */
    row.appendChild(el("span", "route-class", (d.task_class || "?").toLowerCase()));
    row.appendChild(el("span", "route-arrow", "->"));
    row.appendChild(el("span", "badge blue", d.model || "no model"));

    if (d.escalated_from) {
      const up = el("span", "badge warn", "escalated from " + d.escalated_from);
      up.title = "the first answer did not verify, so a stronger model was tried";
      row.appendChild(up);
    }

    row.appendChild(el("span", "spacer"));
    row.appendChild(el("span", "muted tiny",
      Math.round(d.confidence * 100) + "% - " + (d.elapsed_ms || 0) + " ms"));
    host.appendChild(row);

    if (d.considered && d.considered.length > 1) {
      host.appendChild(el("div", "route-considered",
        "considered: " + d.considered.join(", ")));
    }
  }
}

let outputsShownFor = null;    // which run the visible tab was chosen for

function renderOutputs(body) {
  $("out-code-count").textContent = body.code.count;
  $("out-doc-count").textContent = body.documents.count;

  renderCode($("out-code"), body.code);
  renderDocuments($("out-document"), body.documents);

  /* Open on the side this run is actually about, ONCE per run.
     Opening a document run on an empty Code tab shows "this run wrote no
     source files" as the first thing a reader sees about a run that produced
     exactly what it was asked for. After that first choice the tab is the
     reader's - re-selecting it on every 2s refresh would drag them back from
     whichever side they were reading. */
  if (outputsShownFor !== body.run_id) {
    outputsShownFor = body.run_id;
    const preferred = body.mode === "code" ? "code"
                    : body.mode === "document" ? "document"
                    : body.documents.count ? "document"
                    : body.code.count ? "code"
                    : null;
    if (preferred) {
      const tab = $("out-tabs").querySelector('[data-out="' + preferred + '"]');
      if (tab && !tab.classList.contains("active")) tab.click();
    }
  }

  /* An execution that was not sandboxed is a sovereignty caveat, not a footnote
     on one card: it means Python from a model ran against this machine's real
     filesystem. It is hoisted to the banner like every other one. */
  const unsandboxed = body.code.items.filter(
    (item) => item.kind === "executed" && item.sandboxed === false).length;
  /* ONE key, not one per run. The panel shows a single run at a time and the
     text names which one, so keying per run would leave a warning about a run
     nobody is looking at standing in the banner for the rest of the session. */
  caveat("sandbox", unsandboxed
    ? unsandboxed + " step(s) in run " + body.run_id + " executed model-written "
      + "Python WITHOUT a container. The dev fallback is on; results from those "
      + "steps are not evidence of a sandbox."
    : null, "runs");
}

function copyButton(text) {
  const button = el("button", "tiny ghost", "copy");
  button.type = "button";
  button.addEventListener("click", async () => {
    try {
      /* The page is served over http from loopback, which browsers treat as a
         secure context, so the clipboard API is available. The fallback is for
         the case where it is not - a denied permission must not leave the
         button silently doing nothing. */
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const area = el("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
      }
      button.textContent = "copied";
      setTimeout(() => { button.textContent = "copy"; }, 1500);
    } catch (e) {
      toast("This browser refused clipboard access - select the text instead",
            "warn", 6000);
    }
  });
  return button;
}

function renderCode(host, side) {
  clear(host);
  if (!side.items.length) {
    host.appendChild(el("div", "empty", side.note || "no code"));
    return;
  }

  for (const item of side.items) {
    const block = el("div", "out-item");
    const head = el("div", "out-head");

    if (item.kind === "executed") {
      head.appendChild(el("span", "out-name", "step " + item.step_idx + " - " + item.intent));
      head.appendChild(el("span", "badge blue", item.language));
      /* The verdict, then the exit code - in that order and never the other
         way round. An Observation defaults exit_code to 0, so a step that never
         reached the sandbox reports exit 0; badging on that alone puts a green
         tick above the stderr explaining why it did not run. */
      if (!item.ran) {
        head.appendChild(el("span", "badge", "not run yet"));
      } else if (item.ok === false) {
        head.appendChild(el("span", "badge bad", "failed"));
      } else if (item.exit_code === 0) {
        head.appendChild(el("span", "badge good", "exit 0"));
      } else {
        head.appendChild(el("span", "badge bad", "exit " + item.exit_code));
      }
      /* Three states, never two. `null` means the step has not run, which is
         not the same claim as "ran outside a container" and must not be drawn
         the same way. */
      if (item.sandboxed === true) head.appendChild(el("span", "badge good", "sandboxed"));
      else if (item.sandboxed === false) head.appendChild(el("span", "badge bad", "NOT SANDBOXED"));
    } else {
      const link = el("a", "out-name mono", item.path);
      link.href = item.download;
      link.download = item.name;
      head.appendChild(link);
      head.appendChild(el("span", "badge blue", item.language));
      head.appendChild(el("span", "muted tiny", (item.sha256 || "").slice(0, 10)));
    }

    head.appendChild(el("span", "spacer"));
    if (item.source) head.appendChild(copyButton(item.source));
    block.appendChild(head);

    if (item.source) {
      block.appendChild(el("pre", "code", item.source));
      if (item.truncated) {
        block.appendChild(el("p", "tiny warn",
          "shown up to the display limit - download the file for the rest"));
      }
    } else {
      block.appendChild(el("div", "notice warn",
        item.source_note || "the source is not available"));
    }

    if (item.kind === "executed") {
      const printed = (item.stderr || "") + (item.stdout || "");
      block.appendChild(el("h4", "sec", item.stderr ? "stderr" : "stdout"));
      block.appendChild(el("pre", "code out-stdout", printed.trim() ||
        (item.ran ? "(the program printed nothing)" : "(not run yet)")));
      /* The tool's own reading of the failure - a missing entry point, a
         toolchain this image does not carry. A compiler message is written for
         someone who already knows the language; this says which of the two
         problems it is. */
      if (item.hint) block.appendChild(el("div", "notice warn", item.hint));
    }
    host.appendChild(block);
  }
}

/* How the preview was obtained decides how strongly it may be worded. Reading
   the bytes back proves what is IN the file; echoing the text the run asked to
   write proves only what was requested, and a failed write would look perfect
   under that label if it were not said out loud. */
const PREVIEW_SOURCE = {
  file: ["good", "read back out of the file"],
  requested: ["warn", "this is the text the run ASKED to write - the file itself could not be read back, so it is not proof of what is in it"],
};

function renderDocuments(host, side) {
  clear(host);
  if (!side.items.length) {
    host.appendChild(el("div", "empty", side.note || "no documents"));
    return;
  }

  for (const item of side.items) {
    const block = el("div", "out-item");
    const head = el("div", "out-head");

    const link = el("a", "out-name mono", item.path);
    link.href = item.download;
    link.download = item.name;
    link.title = "download " + item.path;
    head.appendChild(link);
    head.appendChild(el("span", "badge blue", item.format));
    head.appendChild(el("span", "muted tiny", (item.sha256 || "").slice(0, 10)));
    if (item.approved_by) {
      head.appendChild(el("span", "badge good", "signed by " + item.approved_by));
    }
    head.appendChild(el("span", "spacer"));
    if (item.preview) head.appendChild(copyButton(item.preview));
    block.appendChild(head);

    if (item.preview) {
      const origin = PREVIEW_SOURCE[item.preview_source] ||
        ["warn", "the origin of this text is unknown"];
      block.appendChild(el("p", "tiny " + origin[0], origin[1]));
      block.appendChild(el("pre", "doc-preview", item.preview));
      if (item.truncated) {
        block.appendChild(el("p", "tiny warn",
          "shown up to the display limit - download the file for the rest"));
      }
    } else {
      block.appendChild(el("div", "notice warn",
        item.preview_note || "this box cannot show a preview of this file"));
    }
    host.appendChild(block);
  }
}

const RUN_MODE_HINT = {
  auto: "the classifier picks the model and the plan decides the deliverable",
  code: "routed as a coding task; the plan must write a program and run it in the sandbox",
  document: "routed as a drafting task; the plan must end in a Word, Excel or PowerPoint file",
};

segmented($("out-tabs"), "out", (which) => {
  $("out-code").classList.toggle("hide", which !== "code");
  $("out-document").classList.toggle("hide", which !== "document");
});

/* A resting state before any run is opened. An empty pane under two tabs and a
   zero count reads as "loading" or "broken"; this says which it is. */
for (const [pane, kind] of [[$("out-code"), "code"], [$("out-document"), "document"]]) {
  pane.appendChild(el("div", "empty",
    "Start a task above, or pick one from Earlier tasks, to see its "
    + kind + " here."));
}

/* Choosing what a run should DELIVER also chooses what to show when it lands.
   Starting a document run and being left staring at the code tab is the panel
   contradicting the toggle that started it. */
segmented($("run-mode"), "mode", (mode) => {
  set("run-mode-hint", RUN_MODE_HINT[mode] || "");
  suggestions($("run-examples"), $("run-goal"), RUN_EXAMPLES[mode] || RUN_EXAMPLES.auto);
  if (mode !== "auto") {
    const tab = $("out-tabs").querySelector("[data-out=\"" + mode + "\"]");
    if (tab) tab.click();
  }
});

async function approve(runId, stepIdx) {
  const by = prompt("Approve step " + stepIdx + " as (your name):", "operator");
  if (!by) return;
  await post("/api/runs/" + runId + "/approve", { step_idx: stepIdx, approved_by: by });
  watch(runId);          // approving resumes it; re-attach so the rest streams
  refreshRuns();
}

$("form-run").addEventListener("submit", async (event) => {
  event.preventDefault();
  const goal = $("run-goal").value.trim();
  if (!goal) return;
  stopStream();
  streamState("blue", "planning...");
  $("run-error").classList.add("hide");
  $("run-submit").disabled = true;
  try {
    const planOnly = $("run-plan-only").checked;
    const timeline = await post("/api/runs", {
      goal, execute: !planOnly, stream: !planOnly,
      mode: segmentedValue($("run-mode"), "mode") || "auto",
    });
    renderRun(timeline);
    if (planOnly) streamState("", "planned, not run");
    else watch(timeline.run_id);
    refreshRuns();
  } catch (err) {
    streamState("bad", "failed");
    $("run-error").classList.remove("hide");
    $("run-error").textContent = err.message;
  } finally {
    $("run-submit").disabled = false;
  }
});

/* Opening one run, from a click or from the URL.

   `?run=run_1a2b3c` deep-links a run the way `?theme=` deep-links a theme and
   `#runs` deep-links a view. It is what makes a finished run something an
   engineer can put in a mail to a colleague on the same box - and without it
   the only route to a run's deliverables is to find it in a list and click,
   which is no route at all from a report or a run sheet. */
async function openRun(runId) {
  show("runs");
  try {
    renderRun(await api("/api/runs/" + encodeURIComponent(runId) + "/timeline"));
  } catch (err) {
    $("run-error").classList.remove("hide");
    $("run-error").textContent = "run " + runId + " could not be opened: " + err.message;
    return;
  }
  watch(runId);
}

// Read once, at startup. A malformed or unknown id lands in the panel's own
// error line rather than silently showing the previous run.
(function openRunFromTheUrl() {
  const asked = (location.search.match(/[?&]run=([\w-]{1,64})/) || [])[1];
  if (asked) openRun(asked);
})();

async function refreshRuns() {
  let body;
  try { body = await api("/api/runs?limit=10"); } catch (e) { return; }

  for (const [host, limit] of [[$("run-history"), 10], [$("ov-runs"), 5]]) {
    clear(host);
    if (!body.runs.length) {
      host.appendChild(el("li", "muted", "no runs yet"));
      continue;
    }
    for (const run of body.runs.slice(0, limit)) {
      const li = el("li");
      const tone = run.status === "done" ? "good"
                 : run.status === "failed" || run.status === "budget_exhausted" ? "bad"
                 : run.status === "awaiting_approval" ? "warn" : "";
      li.appendChild(el("span", "ellipsis", run.goal));
      li.appendChild(el("span", "badge " + tone, run.status));
      li.style.cursor = "pointer";
      li.addEventListener("click", () => openRun(run.id));
      host.appendChild(li);
    }
  }
}

// -------------------------------------------------------------- corpus (L3)

/* One tone per way a page was read, and NEVER two the same: NATIVE and OCR were
   both green, so the coverage bar showed two identical blocks and the split it
   exists to communicate was invisible. The order is a confidence order - a real
   text layer is better evidence than a transcription of a picture of text. */
/* Provenance, in words a plant engineer can act on. The enum is exactly right
   in the code and exactly wrong on a table: SIMULATED does not mean "this model
   is fake", it means "its weight file is not on this computer". */
const WEIGHTS_STATE = {
  VERIFIED: ["good", "installed, checked"],
  UNVERIFIED: ["warn", "installed, not approved"],
  MISMATCH: ["bad", "FILE CHANGED"],
  SIMULATED: ["warn", "file not on this PC"],
};

const EXTRACTION_TONE = {
  NATIVE: "green",       // the document's own text layer
  OCR: "blue",           // read off an image by tesseract
  VLM: "violet",         // read by a vision model
  SIMULATED: "amber",    // synthetic, excluded from grounded answers
  UNREADABLE: "red",     // nothing could read it, and it is not in the index
};

async function refreshCorpus() {
  let body;
  try { body = await api("/api/kb"); } catch (e) { return; }
  if (!payloadChanged("kb", body)) return;
  const corpus = body.corpus;

  set("kb-docs", corpus.documents);
  set("kb-chunks", corpus.chunks);
  set("kb-gaps", corpus.unreadable_pages,
    corpus.unreadable_pages ? "bad" : "good");
  set("kpi-gaps", corpus.unreadable_pages, corpus.unreadable_pages ? "bad" : "good");
  set("kpi-gaps-note", corpus.documents
    ? `${corpus.pages} pages across ${corpus.documents} documents`
    : "no corpus ingested");

  const state = $("kb-state");
  state.className = "badge " + (body.index.degraded ? "warn" : "good");
  state.textContent = body.index.degraded ? "word-match only" : "full search";

  const segments = Object.entries(corpus.extraction || {}).map(([kind, count]) => ({
    label: kind, value: count, tone: EXTRACTION_TONE[kind] || "muted",
  }));
  Charts.stackedBar($("kb-coverage"), segments);
  Charts.stackedBar($("ov-coverage"), segments);
  set("ov-corpus-sub", corpus.documents
    ? `${corpus.documents} docs · ${corpus.chunks} chunks` : "nothing ingested");

  /* The engine's own words go to the console and the caveat; this line is for
     the person looking at the coverage bar, who needs to know whether their
     next search will work and what changes that. */
  $("kb-detail").textContent = body.index.degraded
    ? "Searching by word and tag number works normally. Searching by meaning is "
      + "off until the embedding model is installed (" + (body.index.detail || "")
      + ")."
    : "Searching by word and by meaning are both on.";

  const image = body.image_path;
  kv($("kb-imagepath"), [
    // Plain names for the three ways a page can be read, in the order they
    // are tried. "OCR engine: none" is accurate and means nothing to the
    // person who just uploaded a scan and got no text out of it.
    ["typed text in a PDF", "always readable", "ok"],
    ["text recognition (OCR)", image.ocr_engine || "not installed",
      image.ocr_engine ? "ok" : "bad"],
    ["vision model for photos", (image.vision_model || "not installed") +
      (image.vision_is_simulated ? " (SIMULATED)" : ""),
      image.vision_is_simulated || !image.vision_model ? "bad" : "ok"],
    ["scans and photos", image.mode === "UNREADABLE" ? "cannot be read"
                                                      : image.mode,
     image.mode === "UNREADABLE" ? "bad" : "ok"],
  ]);
  /* Never only the raw status. A row reading "image path: UNREADABLE" is a
     true statement that leaves the reader with nowhere to go; the sentence
     under it names the two things that would change it. */
  $("kb-imagepath-detail").textContent = image.available
    ? (image.detail || "")
    : ((image.detail || "") + ". A PDF with a real text layer still reads "
       + "perfectly - this only affects pictures of text. To fix: install "
       + "Tesseract OCR, or stage a vision model; then re-add those documents.");

  // The gap count is the number a judge should be looking at, so it goes in the
  // banner alongside the sovereignty caveats rather than only in this view.
  caveat("corpus", corpus.unreadable_pages
    ? corpus.warning
    : (body.index.degraded && corpus.chunks
        ? "SEARCH IS WORD-MATCH ONLY. Meaning-based search is switched off "
          + "because the model that turns text into numbers is not installed, "
          + "so a question worded differently from the document may not find "
          + "it. Word and tag-number search is unaffected. Detail: "
          + body.index.detail
          + ". To fix: install the embedding model, then re-add the documents."
        : null), "corpus");

  const docsHost = $("kb-docs-list");
  const ovHost = $("ov-docs");
  clear(docsHost); clear(ovHost);
  if (!corpus.documents) {
    const row = el("tr");
    const cell = el("td", "empty", "no documents — run: kavach kb ingest MRPL_DEMO_CORPUS");
    cell.colSpan = 5;
    row.appendChild(cell);
    docsHost.appendChild(row);
    const row2 = el("tr");
    const cell2 = el("td", "empty", "nothing ingested");
    cell2.colSpan = 3;
    row2.appendChild(cell2);
    ovHost.appendChild(row2);
    return;
  }

  let listing;
  try { listing = await api("/api/kb/documents"); } catch (e) { return; }
  for (const doc of listing.documents) {
    const tr = el("tr");
    tr.appendChild(el("td", "mono", doc.id));
    const label = el("td");
    label.appendChild(el("span", "badge", doc.label));
    tr.appendChild(label);
    tr.appendChild(el("td", "num", doc.pages));
    tr.appendChild(el("td", "num", doc.readable_pages));
    const gaps = el("td");
    if (doc.unreadable_pages.length) {
      gaps.appendChild(el("span", "badge bad",
        "p." + doc.unreadable_pages.join(",")));
    } else {
      gaps.appendChild(el("span", "badge good", "complete"));
    }
    tr.appendChild(gaps);
    docsHost.appendChild(tr);

    const short = el("tr");
    short.appendChild(el("td", "mono", doc.id));
    const l2 = el("td");
    l2.appendChild(el("span", "badge", doc.label));
    short.appendChild(l2);
    const r2 = el("td", "num");
    r2.appendChild(el("span",
      doc.unreadable_pages.length ? "bad" : "ok",
      `${doc.readable_pages}/${doc.pages}`));
    short.appendChild(r2);
    ovHost.appendChild(short);
  }
}

$("form-ask").addEventListener("submit", async (event) => {
  event.preventDefault();
  const out = $("ask-out");
  clear(out);

  const question = $("ask-q").value.trim();
  if (!question) {
    out.appendChild(el("div", "notice", "Ask something the corpus might answer."));
    $("ask-q").focus();
    return;
  }

  out.appendChild(el("p", "muted tiny", "retrieving, routing, then checking every citation..."));
  try {
    renderAnswer(await post("/api/kb/answer", {
      question,
      role: currentRole(),
    }));
  } catch (err) {
    clear(out);
    out.appendChild(el("div", "notice bad", err.message));
  }
});

function renderAnswer(body) {
  const out = $("ask-out");
  clear(out);

  const head = el("div", "inline");
  head.appendChild(el("span", "badge " + (body.ok ? "good" : "bad"),
    body.ok ? "GROUNDED" : "NO GROUNDED ANSWER"));
  head.appendChild(el("span", "badge", body.label));
  if (body.model) head.appendChild(el("span", "badge blue", body.model));
  if (body.simulated) head.appendChild(el("span", "badge warn", "SIMULATED PROSE"));
  if (body.grounding) {
    head.appendChild(el("span", "badge",
      body.grounding.resolved + "/" + body.grounding.cited + " citations resolve"));
  }
  out.appendChild(head);

  if (body.ok) {
    const answer = el("div", "answer");
    answer.style.marginTop = "10px";
    // Citations are highlighted so a reader can see at a glance that every claim
    // carries one - the whole point of §07, and invisible in a wall of text.
    for (const part of String(body.answer).split(/(\[[^\]]+\])/g)) {
      if (/^\[.+\]$/.test(part)) answer.appendChild(el("span", "cite", part));
      else answer.appendChild(document.createTextNode(part));
    }
    out.appendChild(answer);
    out.appendChild(el("p", "tiny muted", body.headline));
  } else {
    out.appendChild(el("div", "notice " + (body.rejected_answer ? "bad" : "warn"),
      body.headline));
    if (body.rejected_answer) {
      out.appendChild(el("p", "tiny muted",
        "what the model produced, withheld because it is not grounded:"));
      out.appendChild(el("div", "answer rejected", body.rejected_answer));
    }
  }

  // An unresolved citation is the failure this whole plane exists to catch, so
  // it is named individually rather than summarised.
  const unresolved = (body.grounding && body.grounding.unresolved) || [];
  if (unresolved.length) {
    out.appendChild(el("h3", "sec", "citations that do not resolve"));
    const list = el("ul", "plain");
    for (const item of unresolved) {
      const li = el("li");
      li.appendChild(el("span", "cite", item.raw));
      li.appendChild(el("span", "badge bad", item.verdict));
      li.appendChild(el("span", "ellipsis muted tiny", item.detail));
      list.appendChild(li);
    }
    out.appendChild(list);
  }

  const retrieval = body.retrieval || {};
  if (retrieval.withheld) {
    out.appendChild(el("div", "notice warn",
      retrieval.withheld + " passage(s) withheld — " + retrieval.withheld_reason));
  }
  if (retrieval.coverage && retrieval.coverage.warning) {
    out.appendChild(el("div", "notice warn", retrieval.coverage.warning));
  }

  if ((body.sources || []).length) {
    out.appendChild(el("h3", "sec", "sources"));
    const list = el("ul", "plain");
    for (const src of body.sources) {
      const li = el("li");
      li.appendChild(el("span", "cite", src.citation));
      li.appendChild(el("span", "badge", src.label));
      li.appendChild(el("span", "badge " +
        (EXTRACTION_TONE[src.extraction] === "green" ? "good" : "warn"),
        src.extraction));
      li.appendChild(el("span", "ellipsis muted tiny", src.excerpt));
      list.appendChild(li);
    }
    out.appendChild(list);
  }
}

// ---------------------------------------------------------------- startup

/* Corpus upload. fetch() with FormData and no Content-Type header - the browser
   sets the multipart boundary, and setting it by hand is the classic way to get
   a 422 that looks like a server bug. Same origin, so the file never leaves this
   machine: it is read here, hashed into the audit chain here, indexed here. */
/* Uploading is a JOB you watch, not a request that hangs.

   Reading a scanned page can take minutes: a vision model load, an image
   encode, a transcription. The old handler waited inside one fetch for all of
   it, which meant a spinner that could not say what it was waiting for and -
   because the ingest held uvicorn's only event loop - a gateway that stopped
   answering anything at all. People reloaded, and started a second ingest of
   the same file.

   Now the upload returns immediately with a job id and this polls it. What is
   shown is what the server actually knows: a stage, and a page count. Never a
   percentage - a bar filled from elapsed time stalls at 90% on exactly the
   page that is taking the time, which is worse than no bar. */

const INGEST_STAGES = [
  ["reading", "Reading the document"],
  ["indexing", "Adding it to the search index"],
  ["done", "Ready to search"],
];

let ingestPoll = null;

function renderIngest(job) {
  const host = $("kb-progress");
  host.classList.remove("hide");
  clear(host);

  const head = el("div", "ingest-head");
  const spinner = el("span", "spinner" + (job.running ? "" : " hide"));
  head.appendChild(spinner);
  head.appendChild(el("span", "ingest-name mono", job.filename));
  head.appendChild(el("span", "spacer"));
  head.appendChild(el("span", "tiny muted", fmtAge(job.elapsed_s) + " so far"));
  host.appendChild(head);

  /* The stages, as steps. Which one is current is the honest signal - the
     slow one is always "Reading", and saying so is what stops a person
     wondering whether it is stuck. */
  const steps = el("div", "ingest-steps");
  const reached = INGEST_STAGES.findIndex(([key]) => key === job.stage);
  INGEST_STAGES.forEach(([key, label], index) => {
    const failed = job.stage === "failed";
    const state = failed && index === 0 ? "bad"
                : index < reached ? "done"
                : index === reached ? "now" : "";
    const step = el("div", "ingest-step " + state);
    step.appendChild(el("span", "ingest-dot", state === "done" ? "✓"
                                            : state === "bad" ? "✕" : ""));
    step.appendChild(el("span", null, label));
    steps.appendChild(step);
  });
  host.appendChild(steps);

  const detail = el("div", "ingest-detail " + (job.stage === "failed" ? "bad" : ""));
  detail.textContent = job.error || job.detail || "";
  host.appendChild(detail);

  // A page count is the one true measure of progress here, so it is shown
  // whenever the server has one and omitted entirely when it does not.
  if (job.pages > 1 && job.running) {
    const bar = el("div", "ingest-bar");
    const fill = el("span");
    fill.style.width = Math.round((job.page / job.pages) * 100) + "%";
    bar.appendChild(fill);
    host.appendChild(bar);
    host.appendChild(el("div", "tiny muted",
      "page " + job.page + " of " + job.pages));
  }
}

function watchIngest(jobId) {
  if (ingestPoll) clearInterval(ingestPoll);

  /* ONE SLOW REPLY IS NOT A LOST GATEWAY.

     Loading a 6 GB vision model saturates the disk and the CPU for a minute or
     two, and a /jobs poll in the middle of that can take seconds or time out.
     Giving up on the first miss declared "lost contact" at precisely the
     moment the work was going hardest - and the read was still running fine.
     Five consecutive failures is about twenty seconds of silence, which a
     gateway that is actually gone will never recover from and a busy one
     always does. */
  let misses = 0;
  const GIVE_UP_AFTER = 5;

  /* `clearInterval` stops FUTURE ticks; it does nothing to the ones already
     in flight. With a 1 s interval and an 8 s timeout there are half a dozen
     requests outstanding at any moment, so when the gateway went away every
     one of them rejected, every one ran the give-up branch, and the operator
     got six identical red toasts stacked up the screen.

     This latch is what makes "give up" happen once. Checked at the top of the
     tick as well, so a late rejection cannot restart a watch that has already
     finished. */
  let stopped = false;
  const stop = () => {
    stopped = true;
    if (ingestPoll) { clearInterval(ingestPoll); ingestPoll = null; }
    $("kb-upload").disabled = false;
  };

  const tick = async () => {
    if (stopped) return;
    let job;
    try {
      job = await api("/api/kb/jobs/" + encodeURIComponent(jobId));
      misses = 0;
    } catch (e) {
      if (stopped) return;

      if (e.status === 404) {
        // The reader restarted and took its in-flight jobs with it. Nothing is
        // reading this document, so the panel must stop pretending otherwise.
        stop();
        $("kb-progress").classList.add("hide");
        set("kb-upload-state", "the reader restarted - this read did not finish");
        toast("The reader restarted while that document was being read, so it "
              + "was not added. Upload it again.", "warn", 10000);
        return;
      }

      misses += 1;
      if (misses < GIVE_UP_AFTER) {
        set("kb-upload-state",
            "still reading - the box is busy loading a model (" + misses + ")");
        return;
      }
      stop();
      set("kb-upload-state", "lost contact with the reader");
      toast("Lost contact with the gateway while reading the document. The "
            + "read may still be running - check My documents in a minute.",
            "bad", 12000);
      return;
    }
    if (stopped) return;

    renderIngest(job);
    if (job.running) return;

    stop();
    $("kb-file").value = "";

    if (job.stage === "failed") {
      set("kb-upload-state", "could not be read");
      toast(job.filename + " could not be read: " + (job.error || ""), "bad", 12000);
    } else {
      const gaps = ((job.result || {}).corpus || {}).unreadable_pages || 0;
      set("kb-upload-state", "added to the corpus");
      /* The finished notification the reader asked for. It stays up longer
         than a normal toast because the whole point is that they walked away
         while it was working. */
      toast(job.filename + " is ready to search"
            + (gaps ? " - " + gaps + " page(s) could not be read" : "")
            + "  (" + fmtAge(job.elapsed_s) + ")",
            gaps ? "warn" : "good", 12000);
    }
    refreshCorpus();
  };

  tick();
  ingestPoll = setInterval(tick, 1000);
}

$("kb-upload").addEventListener("click", async () => {
  const input = $("kb-file");
  const file = input.files && input.files[0];
  if (!file) { set("kb-upload-state", "choose a file first"); return; }

  const form = new FormData();
  form.append("file", file);
  const label = $("kb-label").value;
  const query = "?background=1" + (label ? "&label=" + encodeURIComponent(label) : "");

  set("kb-upload-state", "sending...");
  $("kb-upload").disabled = true;
  try {
    const res = await fetch("/api/kb/upload" + query, { method: "POST", body: form });
    const body = await res.json().catch(() => null);
    if (!res.ok) {
      /* Say what the box said. "415 unsupported" with the list of formats is
         actionable; "upload failed" sends the operator to the logs. */
      const why = (body && body.detail) ? body.detail : ("failed: " + res.status);
      set("kb-upload-state", why);
      toast(why, "bad", 9000);
      $("kb-upload").disabled = false;
      return;
    }
    set("kb-upload-state", "reading it now - you can leave this page open");
    watchIngest(body.job.id);
  } catch (e) {
    set("kb-upload-state", "the gateway did not answer");
    toast("Upload failed: " + e.message, "bad", 9000);
    $("kb-upload").disabled = false;
  }
});

/* A reload during a long read must not lose the progress: the job is on the
   server, so the page picks it back up. */
(async function resumeIngest() {
  try {
    const body = await api("/api/kb/jobs");
    // Only jobs this gateway is actually running. A finished or lost one must
    // not put a spinner on screen at page load.
    if (body.running && body.running.length) {
      $("kb-upload").disabled = true;
      set("kb-upload-state", "still reading a document from earlier");
      watchIngest(body.running[0].id);
    }
  } catch (e) { /* nothing in flight, or the gateway is not up yet */ }
})();

/* One set of examples per mode. The chips are the fastest route to a working
   demo, and offering a "write it as a .docx" chip while the toggle says Code
   is the panel arguing with itself - the operator follows the chip, gets a
   document out of a code run, and concludes the toggle does nothing. */
const RUN_EXAMPLES = {
  /* The first chip is the brief's own end-to-end example: read the scanned
     inspection report, pull the findings, hand back an approval note somebody
     can sign. It names doc.write and the .docx explicitly, because the tool a
     model does not think of is a tool it does not call. */
  auto: [
    { short: "approval note (Word)",
      value: "Search the corpus for the inspection report findings on the pump, "
           + "then draft an approval note as a Word file named approval_note.docx "
           + "using doc.write" },
    { short: "calc sheet (Excel)",
      value: "Search the corpus for the wall thickness readings, then write them "
           + "as a spreadsheet named calculations.xlsx using doc.write" },
    { short: "pressure drop report",
      value: "Calculate the pressure drop and write it to a report file" },
    { short: "budget exhaustion",
      value: "do a lot of things" },
  ],
  code: [
    { short: "NPSH script",
      value: "Write npsh.py that computes NPSH available for a pump from suction "
           + "pressure, vapour pressure, static head and friction loss, then run "
           + "it on P-101A's data and print the margin" },
    { short: "orifice sizing",
      value: "Write orifice.py that sizes an orifice plate for a given flow rate "
           + "and differential pressure, then run it and print the bore" },
    { short: "unit-tested calc",
      value: "Write pressure_drop.py with a Darcy-Weisbach function and asserts "
           + "covering laminar and turbulent flow, then run it" },
  ],
  document: [
    { short: "approval note (Word)",
      value: "Search the corpus for the inspection report findings on the pump, "
           + "then draft an approval note as a Word file named approval_note.docx" },
    { short: "calc sheet (Excel)",
      value: "Search the corpus for the wall thickness readings, then write them "
           + "as a spreadsheet named calculations.xlsx" },
    { short: "briefing (PowerPoint)",
      value: "Search the corpus for the hot work permit requirements and build a "
           + "toolbox-talk briefing named hot_work.pptx" },
  ],
};

suggestions($("run-examples"), $("run-goal"), RUN_EXAMPLES.auto);
set("run-mode-hint", RUN_MODE_HINT.auto);
suggestions($("ask-examples"), $("ask-q"), [
  { short: "trip setpoint",
    value: "What is the suction pressure trip setpoint for P-101A?" },
  { short: "hot work permit",
    value: "What are the gas testing thresholds before hot work?" },
  { short: "nothing in the corpus",
    value: "What is the torque spec for the compressor coupling?" },
]);
suggestions($("route-examples"), $("route-prompt"), [
  { short: "code", value: "Script the NPSH-available calculation and unit-test it" },
  { short: "summary", value: "Summarise this 40-page SOP and list the safety interlocks" },
  { short: "scan", value: "Read this scanned inspection report" },
]);

async function tick() {
  await refreshSovereignty();
  await refreshAuditTail();
  await refreshCorpus();
  await refreshRegistry();
}

tick();
refreshRuns();
setInterval(tick, POLL_MS);
setInterval(refreshRuns, POLL_MS * 3);

refreshCapabilities();
setInterval(refreshCapabilities, POLL_MS * 5);

/* ------------------------------------------------- sidebar and the role */

/* The menu closes, and stays closed.
   Remembered per browser so a venue machine set up once opens the way it was
   left. localStorage can throw in a private window or with site data blocked,
   so every access is guarded - a page that will not render because it could
   not remember a menu state is worse than a menu in the wrong state. */
function setNav(open) {
  document.querySelector(".shell").classList.toggle("nav-closed", !open);
  /* The icon shows what pressing it does NEXT, not what the menu is doing now.
     Open menu -> a cross, because the next press closes it. Closed menu -> the
     three bars, because the next press opens it. It was wired the other way
     round, which meant the button told you the state you were already looking
     at and then did the opposite of what it drew. */
  $("nav-toggle").classList.toggle("nav-open-icon", open);
  $("nav-toggle").setAttribute(
    "title", open ? "hide the menu" : "show the menu");
  $("nav-toggle").setAttribute("aria-expanded", String(open));
  try { localStorage.setItem("kavach-nav", open ? "open" : "closed"); } catch (e) {}
}

$("nav-toggle").addEventListener("click", () => {
  setNav(document.querySelector(".shell").classList.contains("nav-closed"));
});

(function restoreNav() {
  let stored = null;
  try { stored = localStorage.getItem("kavach-nav"); } catch (e) {}
  // Narrow screens start closed whatever was stored: 220px of menu on a 900px
  // window leaves no room for the panel the menu is pointing at.
  setNav(stored ? stored === "open" : window.innerWidth > 1024);
})();

/* WHO IS USING THE BOX.

   This is not cosmetic and it is not a login. The policy engine already
   decides, per role, what may be read and what needs sign-off
   (`PolicyEngine.check_tool`); this control is where a person states which
   role they are, and it is sent with every question they ask of their
   documents. The note beside it says what that means in one line, because
   "auditor" tells a user nothing about what they are about to be refused.

   It is deliberately a plain select and not an identity claim: on an
   air-gapped box with a loopback-only gateway, the person at the keyboard is
   the person at the keyboard. Anything stronger would be theatre. */
const ROLE_NOTE = {
  engineer: "reads INTERNAL + CONFIDENTIAL",
  operator: "reads INTERNAL only",
  admin: "reads everything, incl. RESTRICTED",
  auditor: "reads everything, changes nothing",
};

function currentRole() {
  return $("role").value || "engineer";
}

let modelFolderScanned = false;

function applyRole(role) {
  set("role-note", ROLE_NOTE[role] || "");
  // The corpus question box carries the role to the policy engine. Keeping a
  // second copy of the choice in that panel is how the two drift apart and a
  // question gets asked as somebody the user is not.
  const asking = $("ask-role");
  if (asking && asking.value !== role) asking.value = role;
  // The question row states the role rather than offering a second control
  // for it. Two pickers for one fact is how a question gets asked as somebody
  // the user is not.
  set("ask-as", "asking as " + role);
  // The add-a-model card is gated on this too. The server refuses a non-admin
  // regardless; this keeps the panel telling the same story as the gateway.
  if ($("add-model-gate")) addModelGate();
  try { localStorage.setItem("kavach-role", role); } catch (e) {}
}

$("role").addEventListener("change", () => applyRole(currentRole()));

(function restoreRole() {
  let stored = null;
  try { stored = localStorage.getItem("kavach-role"); } catch (e) {}
  // ?role=admin deep-links a role the way ?theme= deep-links a theme: a
  // workstation set up for one job opens in it. It selects a role, it does not
  // grant one - every privileged endpoint checks the role it is sent and
  // records the attempt either way.
  const asked = (location.search.match(/[?&]role=(\w+)/) || [])[1];
  const wanted = ROLE_NOTE[asked] ? asked : stored;
  if (wanted && ROLE_NOTE[wanted]) $("role").value = wanted;
  applyRole(currentRole());
})();

/* ----------------------------------------------------- adding a model (L1) */

/* The brief asks that new open-weight models be addable later without
   redesigning the system. This panel is that path.

   THE GATE IS THE SERVER'S, NOT THIS FILE'S. The card dims for a non-admin
   because /api/models/stage refuses a non-admin, and it says so. Hiding the
   controls would be a UI that merely looks locked; showing them enabled and
   failing on click would be a UI that lies the other way. Both the refusal and
   the staging are written to the audit chain with the role that was claimed. */

function addModelGate() {
  const admin = currentRole() === "admin";
  $("add-model-denied").classList.toggle("hide", admin);
  $("add-model-body").classList.toggle("hide", !admin);
  const gate = $("add-model-gate");
  gate.className = "badge " + (admin ? "good" : "");
  gate.textContent = admin ? "you are an administrator" : "administrator only";

  /* Scan once when the card first opens, so the folder path and anything
     already sitting in it are on screen without a click. An administrator who
     copied a file in a minute ago should not have to guess that a button will
     find it. Once only - re-scanning on every role change would re-read the
     folder for no reason. */
  if (admin && !modelFolderScanned) {
    modelFolderScanned = true;
    scanModelFolder();
  }
}

async function scanModelFolder() {
  const host = $("model-found");
  clear(host);
  host.appendChild(el("p", "muted tiny", "reading the models folder..."));
  let body;
  try {
    body = await api("/api/models/unregistered");
  } catch (err) {
    clear(host);
    host.appendChild(el("div", "notice bad", err.message));
    return;
  }

  set("model-dir", body.model_dir);
  clear(host);
  if (!body.files.length) {
    host.appendChild(el("div", "empty",
      "No unregistered .gguf files in that folder. Copy one in and scan again."));
    return;
  }

  for (const found of body.files) {
    const row = el("div", "out-item");
    const head = el("div", "out-head");
    head.appendChild(el("span", "out-name mono", found.file));
    head.appendChild(el("span", "badge", found.size_gb + " GB"));
    head.appendChild(el("span", "spacer"));

    const look = el("button", "tiny ghost", "Look at it first");
    look.addEventListener("click", () => inspectModel(found.file));
    head.appendChild(look);

    const add = el("button", "tiny", "Add it");
    add.addEventListener("click", () => stageModel(found.file, add));
    head.appendChild(add);

    row.appendChild(head);
    host.appendChild(row);
  }
}

/* A dry run before anything is written to a file that is under version
   control. The architecture line is the one that matters: a GGUF this build's
   llama.cpp cannot load is far better caught here than by a failed load
   halfway through a demo. */
async function inspectModel(file) {
  const out = $("model-stage-out");
  clear(out);
  out.appendChild(el("p", "muted tiny", "reading the file header..."));
  let body;
  try {
    body = await post("/api/models/inspect", { file, role: currentRole() });
  } catch (err) {
    clear(out);
    out.appendChild(el("div", "notice bad", err.message));
    return;
  }

  clear(out);
  const card = el("div", "out-item");
  const head = el("div", "out-head");
  head.appendChild(el("span", "out-name mono", body.gguf.file));
  head.appendChild(el("span", "badge blue", body.gguf.architecture || "no architecture"));
  head.appendChild(el("span", "badge", body.gguf.size_gb + " GB"));
  card.appendChild(head);

  const rows = el("dl", "kv");
  for (const [key, value] of [
    ["it would be called", body.proposed.id],
    ["context length", body.gguf.context_length || "not declared"],
    ["estimated graphics memory", body.proposed.vram_gb + " GB"],
    ["added to profile", body.profile],
  ]) {
    rows.appendChild(el("dt", null, key));
    rows.appendChild(el("dd", "mono", String(value)));
  }
  card.appendChild(rows);

  for (const warning of body.gguf.warnings || []) {
    card.appendChild(el("div", "notice warn", warning));
  }
  card.appendChild(el("p", "tiny muted", body.note));

  const confirm = el("button", null, "Add it");
  confirm.addEventListener("click", () => stageModel(body.gguf.file, confirm));
  card.appendChild(confirm);
  out.appendChild(card);
}

async function stageModel(file, button) {
  const out = $("model-stage-out");
  if (button) button.disabled = true;
  clear(out);
  out.appendChild(el("p", "muted tiny", "checking the file and registering it..."));

  let body;
  try {
    body = await post("/api/models/stage", { file, role: currentRole() });
  } catch (err) {
    clear(out);
    out.appendChild(el("div", "notice bad", err.message));
    if (button) button.disabled = false;
    return;
  }

  clear(out);
  const done = el("div", "notice good");
  done.appendChild(el("b", null, body.id + " is registered. "));
  done.appendChild(document.createTextNode(
    body.added_to_profile
      ? "It is in this profile's model list, so the router can choose it now."
      : "It is NOT in this profile's model list, so the router will not choose "
        + "it until it is added there."));
  out.appendChild(done);

  /* Everything that is still a human's job, in the panel. A model registered
     on estimated memory and seeded quality scores is a routable model, not a
     finished one, and a panel that only said "done" would be the reason
     somebody trusts its ranking. */
  const list = el("ul", "plain");
  for (const step of body.next_steps || []) list.appendChild(el("li", null, step));
  out.appendChild(list);

  scanModelFolder();
  refreshRegistry();
}

/* The drop zone. `dragover` must be cancelled or the browser navigates to the
   file instead of handing it over - the default action for a dropped file is
   to open it. */
(function wireDropzone() {
  const zone = $("model-drop");
  const picker = $("model-file");

  zone.addEventListener("click", () => picker.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); picker.click(); }
  });
  picker.addEventListener("change", () => {
    if (picker.files.length) uploadModel(picker.files[0]);
  });

  for (const kind of ["dragenter", "dragover"]) {
    zone.addEventListener(kind, (e) => {
      e.preventDefault();
      zone.classList.add("over");
    });
  }
  for (const kind of ["dragleave", "drop"]) {
    zone.addEventListener(kind, (e) => {
      e.preventDefault();
      zone.classList.remove("over");
    });
  }
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) uploadModel(file);
  });
})();

async function uploadModel(file) {
  const out = $("model-stage-out");
  clear(out);

  if (!file.name.toLowerCase().endsWith(".gguf")) {
    out.appendChild(el("div", "notice bad",
      file.name + " is not a .gguf file. This box loads GGUF weights - "
      + "download the GGUF build of the model."));
    return;
  }

  const gb = file.size / (1024 * 1024 * 1024);
  const progress = el("div", "notice");
  progress.textContent = "Sending " + file.name + " (" + gb.toFixed(2)
    + " GB) to this computer's models folder. Leave this tab open.";
  out.appendChild(progress);
  if (gb > 2) {
    out.appendChild(el("p", "tiny muted",
      "A file this size will take a while through the browser. Copying it into "
      + "the models folder with File Explorer and pressing Scan the folder does "
      + "the same job in a fraction of the time."));
  }

  const form = new FormData();
  form.append("file", file);
  form.append("role", currentRole());

  /* XMLHttpRequest rather than fetch, for the one thing fetch still cannot do:
     report upload progress. On a 4 GB file a page with no progress is a page
     the operator assumes has hung, and the usual next move is to reload it
     halfway through. */
  const request = new XMLHttpRequest();
  request.open("POST", "/api/models/upload");
  request.upload.addEventListener("progress", (e) => {
    if (!e.lengthComputable) return;
    progress.textContent = "Sending " + file.name + " - "
      + Math.round((e.loaded / e.total) * 100) + "% of " + gb.toFixed(2) + " GB";
  });
  request.addEventListener("load", () => {
    let body = null;
    try { body = JSON.parse(request.responseText); } catch (e) {}
    if (request.status >= 200 && request.status < 300 && body) {
      clear(out);
      const done = el("div", "notice good",
        body.id + " was uploaded and registered.");
      out.appendChild(done);
      const list = el("ul", "plain");
      for (const step of body.next_steps || []) list.appendChild(el("li", null, step));
      out.appendChild(list);
      scanModelFolder();
      refreshRegistry();
      return;
    }
    clear(out);
    const detail = body && body.detail ? body.detail : request.statusText;
    out.appendChild(el("div", "notice bad",
      typeof detail === "string" ? detail : JSON.stringify(detail)));
  });
  request.addEventListener("error", () => {
    clear(out);
    out.appendChild(el("div", "notice bad",
      "The upload failed before it finished. Nothing was added - a partial "
      + "file is discarded rather than left in the models folder."));
  });
  request.send(form);
}

/* --------------------------------------------- capabilities and the probe */

/* WHAT THIS BOX CAN DO, REPORTED BEFORE IT IS ASKED.

   The house rule is that a degraded thing must never look like a working one.
   Code execution was the last place it was not applied: with no Docker, the
   only way to find out was to start a task and watch it fail four minutes
   later. Now it is on the Home page, with the fix beside it. */

const CAPABILITY_ROWS = [
  ["knowledge_base", "Search your documents"],
  ["documents", "Write Word, Excel and PowerPoint"],
  ["model_weights", "AI models installed"],
  ["text_recognition", "Read scans and photos"],
  ["sandbox", "Run code safely"],
  ["policy_engine", "Who may do what"],
  ["audit_chain", "Tamper-proof record"],
  ["sovereignty_checks", "Offline proof"],
];

/* Its own fetch, on its own clock.

   /api/sovereignty is "everything the sovereignty panel renders in one poll"
   and capabilities are not one of those things - hanging them off that payload
   made the render silently no-op, because the field was never there. And they
   change when somebody starts Docker, not every two seconds, so this polls on
   a slower tick than the live readings do. */
async function refreshCapabilities() {
  try {
    renderCapabilities(await api("/api/capabilities"));
  } catch (e) { /* the sovereignty poller already reports a gateway that is down */ }
}

function renderCapabilities(caps) {
  const host = $("capabilities");
  if (!host || !caps) return;
  if (!payloadChanged("capabilities", caps)) return;
  clear(host);

  let ready = 0;
  const list = el("div", "cap-list");

  for (const [key, label] of CAPABILITY_ROWS) {
    const entry = caps[key] || {};
    const ok = entry.available !== false;
    if (ok) ready += 1;

    const row = el("div", "cap-row" + (ok ? "" : " off"));
    const mark = el("span", "cap-mark " + (ok ? "good" : "warn"), ok ? "\u2713" : "!");
    row.appendChild(mark);

    const text = el("div", "cap-text");
    text.appendChild(el("div", "cap-name", label));

    /* The detail line is what makes this a capability panel rather than a row
       of ticks: when something is off it says what to do, and when it is on it
       says what is backing it. */
    let detail = "";
    if (key === "sandbox") {
      detail = ok ? "isolated container, no network interface"
                  : (entry.fix || "not available");
    } else if (key === "model_weights") {
      detail = ok ? entry.installed + " of " + entry.configured + " have their file"
                  : (entry.fix || "no model file on this computer");
    } else if (key === "text_recognition") {
      detail = ok ? (entry.detail || "available") : (entry.fix || entry.detail || "");
    }
    if (detail) text.appendChild(el("div", "cap-detail", detail));
    row.appendChild(text);
    list.appendChild(row);
  }

  host.appendChild(list);
  set("cap-sub", ready + " of " + CAPABILITY_ROWS.length + " ready");

  /* The one capability whose absence changes what a user can ask for goes to
     the honesty banner as well. The others are either on or explained in
     place; this one silently removes a headline feature. */
  caveat("sandbox-capability",
    caps.code_execution === false
      ? "CODE EXECUTION IS OFF. There is no sandbox on this computer, so the "
        + "assistant will refuse to run code rather than run it unprotected. "
        + "Everything else works. " + ((caps.sandbox || {}).fix || "")
      : null, "overview");
}

$("btn-probe").addEventListener("click", async () => {
  const button = $("btn-probe");
  const verdict = $("probe-verdict");
  const out = $("probe-out");

  button.disabled = true;
  set("probe-state", "running");
  verdict.classList.remove("hide");
  verdict.className = "notice";
  verdict.textContent = "running code inside the sandbox that tries to fetch a web page...";
  out.classList.add("hide");

  try {
    const body = await post("/api/tools/sandbox/probe", {});

    /* Three outcomes, and only one of them is proof.
       reached_network true is a BREACH and must be the loudest thing on the
       page. sandboxed false means the code ran on the host, so whatever it
       reported proves nothing about a sandbox that was not there. */
    const breach = body.reached_network === true;
    const unsandboxed = body.sandboxed === false;

    verdict.className = "notice " + (breach || unsandboxed ? "bad" : "good");
    verdict.textContent = breach
      ? "BREACH: code inside the sandbox reached the outside world. Do not present this build."
      : unsandboxed
        ? "INCONCLUSIVE: this ran WITHOUT a container, so it proves nothing about "
          + "a sandbox. " + (body.headline || "")
        : (body.headline || "blocked: the sandbox has no network interface");

    set("probe-state", breach ? "REACHED" : unsandboxed ? "not sandboxed" : "blocked");
    $("probe-state").className = "badge " + (breach || unsandboxed ? "bad" : "good");

    out.classList.remove("hide");
    out.textContent = (body.stdout || "") + (body.stderr || "");
  } catch (err) {
    verdict.className = "notice bad";
    verdict.textContent = err.message;
    set("probe-state", "failed");
    $("probe-state").className = "badge bad";
  } finally {
    button.disabled = false;
  }
});

/* ------------------------------------------- engineering drawings & vision */

let currentDrawingFile = null;
let currentDrawingScale = null;
let currentDrawingBase64 = null;
let latestDrawingReport = null;
let currentOriginalImageSrc = null;
let currentAnnotatedImageSrc = null;

async function checkVisionStatus() {
  try {
    const data = await api("/api/vision/status");
    const badge = $("eng-vlm-badge");
    if (!badge) return;
    if (data.ready) {
      badge.textContent = `${data.model_id} active`;
      badge.className = "badge good";
    } else if (data.model_id) {
      badge.textContent = `${data.model_id} standby`;
      badge.className = "badge warn";
    } else {
      badge.textContent = "OCR / Geometry mode";
      badge.className = "badge";
    }
  } catch (e) {
    // Ignore transient poll error
  }
}

let engineeringInitialized = false;

function initEngineeringView() {
  if (engineeringInitialized) return;
  const uploadBtn = $("eng-upload-btn");
  const fileInput = $("eng-file-input");
  const fileLabel = $("eng-file-label");
  const analyzeBtn = $("eng-analyze-btn");
  const previewImg = $("eng-preview-img");
  const previewContainer = $("eng-preview-container");
  const calibrateBtn = $("eng-calibrate-btn");
  const statusText = $("eng-status-text");

  if (!uploadBtn || !fileInput) return;
  engineeringInitialized = true;

  uploadBtn.addEventListener("click", () => fileInput.click());

  function handleFile(file) {
    if (!file) return;
    const isImage = (file.type && file.type.startsWith("image/")) ||
                    /\.(png|jpe?g|webp|bmp|tiff?|gif)$/i.test(file.name);
    if (!isImage) {
      toast("Please select an image file (PNG, JPG, TIFF, BMP)", "warn");
      return;
    }
    currentDrawingFile = file;
    if (fileLabel) {
      fileLabel.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    }
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      currentDrawingBase64 = e.target.result;
      currentOriginalImageSrc = e.target.result;
      currentAnnotatedImageSrc = null;
      if (previewImg) previewImg.src = e.target.result;
      if (previewContainer) previewContainer.style.display = "block";
      const btnOrig = $("eng-btn-original");
      const btnAnno = $("eng-btn-annotated");
      if (btnOrig && btnAnno) {
        btnOrig.classList.add("active");
        btnAnno.classList.remove("active");
        set("eng-view-mode-label", "Viewing: Original Drawing");
      }
    };
    reader.readAsDataURL(file);
  }

  // Inspection View Switcher
  const btnAnno = $("eng-btn-annotated");
  const btnOrig = $("eng-btn-original");
  if (btnAnno && btnOrig) {
    btnAnno.addEventListener("click", () => {
      if (currentAnnotatedImageSrc && previewImg) {
        previewImg.src = currentAnnotatedImageSrc;
        btnAnno.classList.add("active");
        btnOrig.classList.remove("active");
        set("eng-view-mode-label", "Viewing: Live Detection Overlay");
      } else {
        toast("Run analysis first to generate live CV detection overlay", "warn");
      }
    });
    btnOrig.addEventListener("click", () => {
      if (currentOriginalImageSrc && previewImg) {
        previewImg.src = currentOriginalImageSrc;
        btnOrig.classList.add("active");
        btnAnno.classList.remove("active");
        set("eng-view-mode-label", "Viewing: Original Drawing");
      }
    });
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag and drop
  const dropzone = uploadBtn.parentElement;
  if (dropzone) {
    ["dragenter", "dragover"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--accent)";
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--line)";
      });
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    });
  }

  // Calibrate scale
  if (calibrateBtn) {
    calibrateBtn.addEventListener("click", async () => {
      const mm = parseFloat($("eng-known-mm").value);
      const px = parseFloat($("eng-measured-px").value);
      if (!mm || !px || mm <= 0 || px <= 0) {
        toast("Enter positive numbers for both mm and px", "warn");
        return;
      }
      try {
        const res = await post("/api/engineering/calibrate", {
          nominal_mm: mm,
          pixel_distance: px,
        });
        currentDrawingScale = res.scale_mm_per_px;
        $("eng-cal-feedback").textContent = `Calibrated: ${res.scale_ratio_str}`;
        $("eng-kpi-scale").textContent = res.scale_ratio_str;
        toast("Scale calibrated successfully", "good");
      } catch (err) {
        toast(err.message, "bad");
      }
    });
  }

  // Analyze drawing
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      // Fallback: if currentDrawingFile is null but fileInput has a file selected, use it
      if (!currentDrawingFile && fileInput.files && fileInput.files[0]) {
        handleFile(fileInput.files[0]);
      }

      if (!currentDrawingFile) {
        toast("Please select an engineering drawing image first", "warn");
        fileInput.click();
        return;
      }

      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyzing Drawing (OpenCV + Vision Model)...";
      if (statusText) {
        statusText.classList.remove("hide");
        statusText.textContent = "⏳ Extracting geometry, OCR dimensions, ASME Y14.5 rules & local vision model reasoning...";
      }

      const tableBody = $("eng-dims-table");
      if (tableBody) {
        clear(tableBody);
        const row = el("tr");
        const td = el("td", "muted center", "⏳ Analyzing drawing: detecting Hough lines, circles, OCR dimension callouts, and querying local vision model...");
        td.colSpan = 6;
        row.appendChild(td);
        tableBody.appendChild(row);
      }

      const standard = $("eng-standard-select") ? $("eng-standard-select").value : "ASME Y14.5";
      const formData = new FormData();
      formData.append("file", currentDrawingFile);
      formData.append("standard", standard);
      if (currentDrawingScale) {
        formData.append("scale_mm_per_px", currentDrawingScale);
      }

      try {
        const res = await fetch("/api/engineering/analyze", {
          method: "POST",
          body: formData,
        });
        const report = await res.json();
        if (!res.ok) throw new Error(report.detail || "Analysis failed");
        latestDrawingReport = report;

        // Visual Annotated Overlay
        if (report.annotated_image) {
          currentAnnotatedImageSrc = report.annotated_image;
          if (previewImg) previewImg.src = currentAnnotatedImageSrc;
          const bAnno = $("eng-btn-annotated");
          const bOrig = $("eng-btn-original");
          if (bAnno && bOrig) {
            bAnno.classList.add("active");
            bOrig.classList.remove("active");
            set("eng-view-mode-label", "Viewing: Live Detection Overlay");
          }
        }

        // Live Telemetry steps
        const telemContainer = $("eng-telemetry-container");
        const telemList = $("eng-telemetry-list");
        if (telemContainer && telemList && report.telemetry_steps) {
          telemContainer.classList.remove("hide");
          clear(telemList);
          set("eng-telemetry-total", `${report.elapsed_ms}ms total execution`);
          report.telemetry_steps.forEach((st) => {
            const li = el("li");
            li.innerHTML = `<span><span style="color:var(--good); font-weight:bold;">✓</span> <b>[Step ${st.step}] ${st.name}</b> <span class="mono muted">(${st.elapsed_ms}ms)</span>: <span class="muted">${st.detail}</span></span>`;
            telemList.appendChild(li);
          });
        }

        // 1. KPIs
        set("eng-kpi-dims", report.dimensions_count || 0);
        set("eng-kpi-holes", (report.geometry && report.geometry.circles_count) || 0);
        const rulesPass = report.rules_evaluation && report.rules_evaluation.overall_compliant;
        set("eng-kpi-rules", rulesPass ? "PASS" : "WARNINGS", rulesPass ? "good" : "warn");
        if (report.scale_calibration && report.scale_calibration.is_calibrated) {
          set("eng-kpi-scale", report.scale_calibration.scale_ratio_str);
        }

        // 2. Dimensions Table
        if (tableBody) {
          clear(tableBody);
          const dims = report.dimensions || [];
          if (dims.length === 0) {
            const row = el("tr");
            const td = el("td", "muted center", "No explicit dimensions extracted; geometry detected.");
            td.colSpan = 7;
            row.appendChild(td);
            tableBody.appendChild(row);
          } else {
            dims.forEach((d) => {
              const row = el("tr");
              row.appendChild(el("td", "bold mono", d.raw_text));
              row.appendChild(el("td", null, d.feature_type));
              row.appendChild(el("td", "num bold", `${d.nominal} ${d.unit}`));
              const loc = d.bbox ? `(${d.bbox[0]}, ${d.bbox[1]})` : "-";
              row.appendChild(el("td", "num mono muted", loc));
              const low = d.tolerance ? `${d.tolerance.lower_limit} ${d.unit}` : "-";
              const high = d.tolerance ? `${d.tolerance.upper_limit} ${d.unit}` : "-";
              const band = d.tolerance ? `±${(d.tolerance.tolerance_band / 2).toFixed(3)}` : "General";
              row.appendChild(el("td", "num", low));
              row.appendChild(el("td", "num", high));
              row.appendChild(el("td", "mono", band));
              tableBody.appendChild(row);
            });
          }
        }

        // 3. Rules Engine list
        const rulesList = $("eng-rules-list");
        if (rulesList) {
          clear(rulesList);
          const checks = (report.rules_evaluation && report.rules_evaluation.checks) || [];
          checks.forEach((c) => {
            const li = el("li");
            const icon = c.passed && c.severity === "PASS" ? "✅" : (c.severity === "INFO" ? "ℹ️" : (c.severity === "WARNING" ? "⚠️" : "❌"));
            li.innerHTML = `<span>${icon} <b>[${c.rule_id}] ${c.title}</b>: ${c.message}</span>`;
            rulesList.appendChild(li);
          });
        }

        // 4. Tolerance Stack
        const stackCard = $("eng-stack-card");
        if (stackCard) {
          if (report.tolerance_stack) {
            const st = report.tolerance_stack;
            stackCard.innerHTML = `
              <div><b>Cumulative Nominal</b>: <span class="mono">${st.nominal_sum_mm} mm</span> (${st.dimension_count} dimensions)</div>
              <div style="margin-top:4px"><b>Worst-Case Stack</b>: <span class="mono">[${st.worst_case_limits_mm[0]} mm, ${st.worst_case_limits_mm[1]} mm]</span> (Band: ${st.worst_case_band_mm} mm)</div>
              <div style="margin-top:4px"><b>Statistical Stack (RSS)</b>: <span class="mono">[${st.rss_limits_mm[0]} mm, ${st.rss_limits_mm[1]} mm]</span> (Band: ${st.rss_band_mm} mm)</div>
            `;
          } else {
            stackCard.innerHTML = `<span class="muted">No linear chain dimensions available for stack-up calculation.</span>`;
          }
        }

        // 5. VLM Explanation
        const vlmBox = $("eng-vlm-box");
        if (vlmBox) {
          vlmBox.textContent = report.vlm_explanation || "No explanation generated.";
        }

        toast(`Analysis completed in ${report.elapsed_ms}ms`, "good");

        // Progressive Background VLM Reasoning
        if (currentDrawingFile) {
          (async () => {
            const vlmNote = el("div", "muted tiny", "🧠 Querying Local Vision Model (Qwen2.5-VL) for deep engineering assessment...");
            vlmNote.style.marginTop = "10px";
            vlmNote.style.fontStyle = "italic";
            if (vlmBox) vlmBox.appendChild(vlmNote);

            try {
              const rForm = new FormData();
              rForm.append("file", currentDrawingFile);
              rForm.append("standard", standard);
              const rRes = await fetch("/api/engineering/reason", {
                method: "POST",
                body: rForm,
              });
              if (rRes.ok) {
                const reasonData = await rRes.json();
                if (reasonData.vlm_explanation && vlmBox) {
                  vlmBox.textContent = reasonData.vlm_explanation;
                  toast("Local Vision Model assessment ready", "good");
                }
              }
            } catch (e) {
              // Keep deterministic summary if background VLM is interrupted
            }
          })();
        }
      } catch (err) {
        toast(err.message, "bad");
      } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Run Measurement & Rules Engine";
        if (statusText) statusText.classList.add("hide");
      }
    });
  }

  // Interactive Drawing Q&A Assistant
  const chatForm = $("eng-chat-form");
  const chatInput = $("eng-chat-input");
  const chatHistory = $("eng-chat-history");
  const chatSubmit = $("eng-chat-submit");

  function appendChatMessage(sender, text, meta) {
    if (!chatHistory) return;
    const emptyNotice = $("eng-chat-empty");
    if (emptyNotice) emptyNotice.remove();

    const bubble = el("div");
    bubble.style.padding = "8px 12px";
    bubble.style.borderRadius = "6px";
    bubble.style.fontSize = "12px";
    bubble.style.lineHeight = "1.5";

    if (sender === "user") {
      bubble.style.background = "var(--bg-2)";
      bubble.style.borderLeft = "3px solid var(--accent)";
      const bTitle = el("div", "bold tiny", "You asked:");
      const bText = el("div", null, text);
      bubble.appendChild(bTitle);
      bubble.appendChild(bText);
    } else {
      bubble.style.background = "var(--bg-0)";
      bubble.style.border = "1px solid var(--line)";
      const head = el("div", "inline", "AI Engineering Assistant");
      head.style.fontWeight = "bold";
      head.style.fontSize = "11px";
      if (meta) {
        const spanMeta = el("span", "badge", meta);
        spanMeta.style.marginLeft = "8px";
        spanMeta.style.fontSize = "10px";
        head.appendChild(spanMeta);
      }
      bubble.appendChild(head);
      const bText = el("div", "tiny", text);
      bText.style.marginTop = "4px";
      bText.style.whiteSpace = "pre-wrap";
      bubble.appendChild(bText);
    }
    chatHistory.appendChild(bubble);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  async function handleAskQuestion(questionText) {
    if (!questionText || !questionText.trim()) return;
    const q = questionText.trim();
    if (chatInput) chatInput.value = "";

    appendChatMessage("user", q);

    if (chatSubmit) chatSubmit.disabled = true;
    const loadingId = "eng-chat-loading-" + Date.now();
    const loadingBubble = el("div", "muted tiny", "🧠 Reasoning over drawing features, dimensions & ASME standards...");
    loadingBubble.id = loadingId;
    loadingBubble.style.padding = "6px 12px";
    if (chatHistory) {
      chatHistory.appendChild(loadingBubble);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    try {
      const payload = {
        question: q,
        image_base64: currentDrawingBase64 || null,
        drawing_context: latestDrawingReport || null,
      };
      const res = await api("/api/engineering/ask", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const loadElem = $(loadingId);
      if (loadElem) loadElem.remove();

      if (res && res.answer) {
        appendChatMessage("assistant", res.answer, `${res.model_id} · ${res.elapsed_ms}ms`);
      } else {
        appendChatMessage("assistant", "No response received from local assistant.");
      }
    } catch (err) {
      const loadElem = $(loadingId);
      if (loadElem) loadElem.remove();
      appendChatMessage("assistant", `Error: ${err.message || "Failed to query assistant"}`);
    } finally {
      if (chatSubmit) chatSubmit.disabled = false;
    }
  }

  if (chatForm) {
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      if (chatInput) handleAskQuestion(chatInput.value);
    });
  }

  document.querySelectorAll(".eng-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      handleAskQuestion(btn.textContent);
    });
  });

  checkVisionStatus();
}

document.addEventListener("DOMContentLoaded", initEngineeringView);
if (document.readyState !== "loading") initEngineeringView();
