/* Inline SVG chart primitives (L6, §09).
   Zero external libraries, theme-adaptive, offline-safe.
*/

"use strict";

const NS = "http://www.w3.org/2000/svg";

/* Chart colours come from the STYLESHEET, never from a copy of it.

   This file used to carry its own light and dark palettes, hardcoded. Two
   palettes for one design system drift the moment either is touched - and the
   drift is silent, because a chart drawn in last season's blue still looks
   like a chart. Reading the CSS custom properties means the tokens in app.css
   are the only definition of what green means here, in both themes.

   Cached per theme: a sparkline asks for a tone once per point, and
   getComputedStyle is a layout read. The cache key is the theme attribute
   itself, so flipping the toggle invalidates it without any wiring between
   this file and the theme switcher.
*/

const TONE_ALIAS = {
  // Names the charts use -> the token that defines them. `muted` and `line`
  // are the unfilled track and the axis rule, which are surfaces rather than
  // status colours.
  green: "--green",
  amber: "--amber",
  red: "--red",
  blue: "--blue",
  violet: "--violet",
  accent: "--accent",
  muted: "--bg-3",
  line: "--line-strong",
  ink: "--ink",
};

let toneCache = { theme: null, values: {} };

function getTone(name) {
  const theme = document.documentElement.getAttribute("data-theme") || "dark";
  if (toneCache.theme !== theme) toneCache = { theme, values: {} };
  if (toneCache.values[name] !== undefined) return toneCache.values[name];

  const token = TONE_ALIAS[name] || TONE_ALIAS.accent;
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(token).trim();

  /* A token that resolves to nothing would paint an invisible chart - an empty
     stroke is not an error in SVG, it is a line you cannot see. Falling back
     to the ink colour keeps the shape readable and makes the miss obvious
     rather than silent. */
  const resolved = value || getComputedStyle(document.documentElement)
    .getPropertyValue("--ink").trim() || "#888";
  toneCache.values[name] = resolved;
  return resolved;
}

const TONE = new Proxy({}, {
  get(target, prop) {
    return getTone(String(prop));
  }
});

function svgEl(tag, attrs) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function wipe(host) { while (host.firstChild) host.removeChild(host.firstChild); }

/* Every chart on a polled view is rebuilt from scratch each tick.
   Includes active theme in signature so toggling dark/light redraws immediately.
*/
function unchanged(host, signature) {
  const theme = document.documentElement.getAttribute("data-theme") || "dark";
  const full = theme + "|" + signature;
  if (host.dataset.chartSig === full && host.firstChild) return true;
  host.dataset.chartSig = full;
  return false;
}

function emptyState(host, message) {
  wipe(host);
  const div = document.createElement("div");
  div.className = "empty";
  div.textContent = message;
  host.appendChild(div);
}

/* ------------------------------------------------------------------ donut
   Used for the VRAM meter.
*/
function donut(host, { value, max, label, sublabel, tone = "accent", size = 150 }) {
  if (unchanged(host, ["donut", value, max, label, sublabel, tone, size].join("|"))) return;
  if (!max || max <= 0) {
    emptyState(host, sublabel || "no budget to report");
    return;
  }
  wipe(host);

  const stroke = 13;
  const r = (size - stroke) / 2 - 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const fraction = Math.max(0, Math.min(1, value / max));

  const svg = svgEl("svg", {
    class: "chart", viewBox: `0 0 ${size} ${size}`,
    height: size, role: "img",
    "aria-label": `${label}: ${value} of ${max}`,
  });

  svg.appendChild(svgEl("circle", {
    cx, cy, r, fill: "none", stroke: TONE.muted, "stroke-width": stroke,
  }));

  const arc = svgEl("circle", {
    cx, cy, r, fill: "none", stroke: TONE[tone] || TONE.accent,
    "stroke-width": stroke, "stroke-linecap": "round",
    "stroke-dasharray": `${circumference} ${circumference}`,
    "stroke-dashoffset": circumference * (1 - fraction),
    transform: `rotate(-90 ${cx} ${cy})`,
  });
  arc.style.transition = "stroke-dashoffset .45s ease";
  svg.appendChild(arc);

  const value_text = svgEl("text", {
    x: cx, y: cy - 1, "text-anchor": "middle", class: "donut-value",
  });
  value_text.textContent = label;
  svg.appendChild(value_text);

  if (sublabel) {
    const sub = svgEl("text", {
      x: cx, y: cy + 16, "text-anchor": "middle", class: "donut-label",
    });
    sub.textContent = sublabel;
    svg.appendChild(sub);
  }

  host.appendChild(svg);
}

/* -------------------------------------------------------------- sparkline
   A short client-side history the page keeps itself.
*/
function sparkline(host, values, { tone = "green", height = 46 } = {}) {
  if (unchanged(host, ["spark", (values || []).join(","), tone, height].join("|"))) return;
  if (!values || values.length < 2) {
    emptyState(host, "collecting...");
    return;
  }
  wipe(host);

  const width = 100;
  const max = Math.max(1, ...values);
  const step = width / (values.length - 1);
  const points = values.map((v, i) => [
    i * step,
    height - 4 - (v / max) * (height - 10),
  ]);

  const line = points.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`)
    .join(" ");
  const area = `${line} L${width},${height} L0,${height} Z`;

  const svg = svgEl("svg", {
    class: "chart", viewBox: `0 0 ${width} ${height}`,
    height, preserveAspectRatio: "none", role: "img",
  });

  svg.appendChild(svgEl("path", {
    d: area, fill: TONE[tone], "fill-opacity": .12, stroke: "none",
  }));
  svg.appendChild(svgEl("path", {
    d: line, fill: "none", stroke: TONE[tone],
    "stroke-width": 2, "vector-effect": "non-scaling-stroke",
  }));
  host.appendChild(svg);
}

/* ------------------------------------------------------------ stacked bar
   The corpus coverage bar.
*/
function stackedBar(host, segments, { height = 26, legend = true } = {}) {
  if (unchanged(host, ["stack", JSON.stringify(segments || []), height, legend].join("|"))) return;
  const total = (segments || []).reduce((sum, s) => sum + s.value, 0);
  if (!total) {
    emptyState(host, "nothing ingested yet");
    return;
  }
  wipe(host);

  const svg = svgEl("svg", {
    class: "chart", viewBox: `0 0 100 ${height}`,
    height, preserveAspectRatio: "none", role: "img",
  });

  let x = 0;
  for (const segment of segments) {
    if (!segment.value) continue;
    const w = (segment.value / total) * 100;
    const rect = svgEl("rect", {
      x, y: 0, width: Math.max(w - 0.35, 0.4), height,
      rx: 1.2, fill: TONE[segment.tone] || TONE.muted,
    });
    const title = svgEl("title", {});
    title.textContent = `${segment.label}: ${segment.value}`;
    rect.appendChild(title);
    svg.appendChild(rect);
    x += w;
  }
  host.appendChild(svg);

  if (!legend) return;
  const box = document.createElement("div");
  box.className = "legend";
  for (const segment of segments) {
    if (!segment.value) continue;
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = TONE[segment.tone] || TONE.muted;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(
      `${segment.label} ${segment.value} (${Math.round(segment.value / total * 100)}%)`
    ));
    box.appendChild(item);
  }
  host.appendChild(box);
}

/* --------------------------------------------------------- diverging bars
   The router's score breakdown.
*/
function divergingBars(host, terms, { height = 22 } = {}) {
  const entries = Object.entries(terms || {}).filter(([k]) => !k.startsWith("_"));
  if (!entries.length) {
    emptyState(host, "not scored");
    return;
  }
  wipe(host);

  const span = Math.max(0.001, ...entries.map(([, v]) => Math.abs(v)));
  const mid = 50;
  const scale = 46 / span;

  const svg = svgEl("svg", {
    class: "chart", viewBox: `0 0 100 ${entries.length * height}`,
    height: entries.length * height, preserveAspectRatio: "none", role: "img",
  });

  entries.forEach(([name, value], index) => {
    const y = index * height + 5;
    const w = Math.abs(value) * scale;
    const positive = value >= 0;
    const rect = svgEl("rect", {
      x: positive ? mid : mid - w, y, width: Math.max(w, 0.3),
      height: height - 10, rx: 1,
      fill: positive ? TONE.green : TONE.red, opacity: .85,
    });
    const title = svgEl("title", {});
    title.textContent = `${name}: ${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
    rect.appendChild(title);
    svg.appendChild(rect);
  });

  svg.appendChild(svgEl("line", {
    x1: mid, y1: 0, x2: mid, y2: entries.length * height,
    stroke: TONE.line, "stroke-width": .4,
  }));
  host.appendChild(svg);

  const box = document.createElement("div");
  box.className = "legend";
  for (const [name, value] of entries) {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = value >= 0 ? TONE.green : TONE.red;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(
      `${name} ${value >= 0 ? "+" : ""}${value.toFixed(3)}`
    ));
    box.appendChild(item);
  }
  host.appendChild(box);
}

/* --------------------------------------------------------------- bar rows
   A labelled horizontal bar list: model scores, per-class confidence.
*/
function barRows(host, rows, { format } = {}) {
  if (!rows || !rows.length) {
    emptyState(host, "nothing to rank");
    return;
  }
  wipe(host);
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 0.0001);

  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "bar-row";

    const name = document.createElement("div");
    name.className = "name";
    if (row.tone === "green") name.style.color = TONE.green;
    name.textContent = row.label;
    line.appendChild(name);

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, (row.value / max) * 100)}%`;
    fill.style.background = TONE[row.tone] || TONE.accent;
    track.appendChild(fill);
    line.appendChild(track);

    const value = document.createElement("div");
    value.className = "val";
    value.textContent = format ? format(row.value) : row.value.toFixed(3);
    line.appendChild(value);

    host.appendChild(line);
  }
}

window.Charts = { donut, sparkline, stackedBar, divergingBars, barRows, TONE, emptyState };
