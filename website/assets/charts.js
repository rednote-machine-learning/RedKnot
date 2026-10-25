// Tiny SVG chart helpers for the RedKnot site. No dependencies, so pages stay editable by hand.
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function text(parent, x, y, str, cls, extra = {}) {
  const t = el("text", { x, y, class: cls, ...extra }, parent);
  t.textContent = str;
  return t;
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const color = (c) => (c.startsWith("--") ? cssVar(c) : c);

// Rounded data-end, square at the baseline.
function barPath(x, y0, w, y1, r = 4) {
  const top = Math.min(y0, y1), h = Math.abs(y1 - y0);
  r = Math.min(r, h, w / 2);
  if (y1 <= y0) // grows upward
    return `M${x},${y0}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${y0}Z`;
  return `M${x},${y0}V${y1 - r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 - r}V${y0}Z`;
}

let tip;
function tooltip() {
  if (!tip) { tip = document.createElement("div"); tip.className = "tooltip"; document.body.appendChild(tip); }
  return tip;
}
function hover(node, html) {
  node.style.cursor = "default";
  node.addEventListener("mousemove", (e) => {
    const t = tooltip(); t.innerHTML = html; t.style.opacity = 1;
    t.style.left = Math.min(e.clientX + 14, innerWidth - 270) + "px"; t.style.top = e.clientY + 14 + "px";
  });
  node.addEventListener("mouseleave", () => (tooltip().style.opacity = 0));
}

function legend(host, series) {
  if (series.length < 2) return;
  const d = document.createElement("div");
  d.className = "chart-legend";
  d.innerHTML = series.map((s) => `<span style="--c:${color(s.color)}">${s.name}</span>`).join("");
  host.appendChild(d);
}

// Every chart gets a plain table so no value depends on reading color.
function dataTable(host, header, rows) {
  const d = document.createElement("details");
  d.innerHTML = `<summary>Show the numbers</summary><table><tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr>${rows
    .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
    .join("")}</table>`;
  host.appendChild(d);
}

function niceTicks(max, n = 5) {
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

/**
 * Grouped vertical bars.
 * opts: { categories, series:[{name,color,values}], yTitle, xTitle, yMax, fmt, labelSeries:[idx], refLine:{y,label} }
 */
export function bars(host, opts) {
  const { categories, series, fmt = (v) => v, yTitle = "", xTitle = "" } = opts;
  const W = 700, H = opts.height || 300, m = { l: 58, r: 12, t: 22, b: (xTitle ? 58 : 38) + (categories.some((c) => c.includes("\n")) ? 15 : 0) };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const all = series.flatMap((s) => s.values).filter((v) => v != null);
  const ticks = niceTicks(opts.yMax || Math.max(...all) * 1.12);
  const yMax = ticks[ticks.length - 1];
  const y = (v) => m.t + ih - (v / yMax) * ih;
  legend(host, series);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria || yTitle }, host);
  const g = el("g", { class: "ax" }, svg);
  for (const t of ticks) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: cssVar("--grid"), "stroke-width": 1 }, g);
    text(g, m.l - 8, y(t) + 4, opts.tickFmt ? opts.tickFmt(t) : t, "tick", { "text-anchor": "end" });
  }
  text(svg, 14, m.t + ih / 2, yTitle, "axis-title", { transform: `rotate(-90 14 ${m.t + ih / 2})`, "text-anchor": "middle" });
  if (xTitle) text(svg, m.l + iw / 2, H - 6, xTitle, "axis-title", { "text-anchor": "middle" });

  const band = iw / categories.length;
  const bw = Math.min(opts.barWidth || 24, (band * 0.72) / series.length - 2);
  const groupW = series.length * bw + (series.length - 1) * 2;
  categories.forEach((c, i) => {
    const gx = m.l + i * band + (band - groupW) / 2;
    c.split("\n").forEach((line, k) =>
      text(svg, m.l + i * band + band / 2, m.t + ih + 20 + k * 15, line, k ? "tick" : "cat", { "text-anchor": "middle" }));
    series.forEach((s, j) => {
      const v = s.values[i];
      if (v == null) return;
      const x = gx + j * (bw + 2);
      const p = el("path", { d: barPath(x, y(0), bw, y(v)), fill: color(s.color) }, svg);
      const hit = el("rect", { x: x - 3, y: m.t, width: bw + 6, height: ih, fill: "transparent" }, svg);
      hover(hit, `<b>${s.name}</b><br>${c.replace("\n", " · ")}: ${fmt(v)}`);
      if (!opts.labelSeries || opts.labelSeries.includes(j))
        text(svg, x + bw / 2, y(v) - 6, fmt(v), "val", { "text-anchor": "middle" });
      p.setAttribute("aria-label", `${s.name}, ${c.replace("\n", " ")}: ${fmt(v)}`);
    });
  });
  if (opts.refLine) {
    const ry = y(opts.refLine.y);
    el("line", { x1: m.l, x2: W - m.r, y1: ry, y2: ry, stroke: cssVar("--ink-2"), "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: cssVar("--ink-3"), "stroke-width": 1 }, svg);
  dataTable(host, ["", ...series.map((s) => s.name)], categories.map((c, i) => [c.replace("\n", " · "), ...series.map((s) => (s.values[i] == null ? "–" : fmt(s.values[i])))]));
}

/**
 * 100% horizontal stacked bars (two parts).
 * opts: { rows:[{name, a, b, note}], aName, bName, aColor, bColor }
 */
export function split(host, opts) {
  const { rows } = opts;
  const W = 700, rowH = 40, m = { l: 150, r: 20, t: 8, b: 34 };
  const H = m.t + rows.length * rowH + m.b, iw = W - m.l - m.r;
  legend(host, [{ name: opts.aName, color: opts.aColor }, { name: opts.bName, color: opts.bColor }]);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria || "" }, host);
  [0, 25, 50, 75, 100].forEach((t) => {
    const x = m.l + (t / 100) * iw;
    el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b + 4, stroke: cssVar("--grid") }, svg);
    text(svg, x, H - m.b + 20, t + "%", "tick", { "text-anchor": "middle" });
  });
  rows.forEach((r, i) => {
    const yy = m.t + i * rowH + 9, h = 22;
    text(svg, m.l - 12, yy + 16, r.name, "cat", { "text-anchor": "end" });
    const wa = (r.a / 100) * iw;
    const a = el("rect", { x: m.l, y: yy, width: wa - 1, height: h, rx: 4, fill: color(opts.aColor) }, svg);
    const b = el("rect", { x: m.l + wa + 1, y: yy, width: iw - wa - 1, height: h, rx: 4, fill: color(opts.bColor) }, svg);
    hover(a, `<b>${r.name}</b><br>${opts.aName}: ${r.a}%`);
    hover(b, `<b>${r.name}</b><br>${opts.bName}: ${(100 - r.a).toFixed(1)}%`);
    if (wa > 56) text(svg, m.l + 10, yy + 16, r.a.toFixed(1) + "%", "val", { style: "fill:#fff" });
    else text(svg, m.l + wa + 8, yy + 16, r.a.toFixed(1) + "%", "val", { style: `fill:${cssVar("--red-deep")}` });
    text(svg, W - m.r - 8, yy + 16, (100 - r.a).toFixed(1) + "%", "val", { "text-anchor": "end", style: "fill:#fff" });
  });
  dataTable(host, ["Model", opts.aName, opts.bName], rows.map((r) => [r.name, r.a + "%", (100 - r.a).toFixed(1) + "%"]));
}

/**
 * Horizontal diverging bars around zero.
 * opts: { rows:[{name, v}], fmt, xTitle, range }
 */
export function diverge(host, opts) {
  const { rows, fmt = (v) => v } = opts;
  const W = 700, rowH = 38, m = { l: 170, r: 60, t: 8, b: 48 };
  const H = m.t + rows.length * rowH + m.b, iw = W - m.l - m.r;
  const R = opts.range || 15;
  const x = (v) => m.l + ((v + R) / (2 * R)) * iw;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria || "" }, host);
  for (let t = -R; t <= R; t += R / 3) {
    el("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b + 4, stroke: cssVar("--grid") }, svg);
    text(svg, x(t), H - m.b + 20, (t > 0 ? "+" : "") + Math.round(t), "tick", { "text-anchor": "middle" });
  }
  text(svg, m.l + iw / 2, H - 6, opts.xTitle || "", "axis-title", { "text-anchor": "middle" });
  rows.forEach((r, i) => {
    const yy = m.t + i * rowH + 8, h = 22;
    text(svg, m.l - 12, yy + 16, r.name, "cat", { "text-anchor": "end", "font-weight": r.bold ? 700 : 400 });
    const x0 = x(0), x1 = x(r.v);
    const c = r.v >= 0 ? cssVar("--s-redknot") : cssVar("--s-base");
    const w = Math.max(Math.abs(x1 - x0), 2);
    const rect = el("rect", { x: Math.min(x0, x1), y: yy, width: w, height: h, rx: 4, fill: c }, svg);
    hover(rect, `<b>${r.name}</b><br>${fmt(r.v)}`);
    text(svg, r.v >= 0 ? x1 + 8 : x1 - 8, yy + 16, fmt(r.v), "val", { "text-anchor": r.v >= 0 ? "start" : "end" });
  });
  el("line", { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b + 4, stroke: cssVar("--ink-2"), "stroke-width": 1.5 }, svg);
  dataTable(host, ["", "Change"], rows.map((r) => [r.name, fmt(r.v)]));
}

// Deterministic pseudo-random so the picture is the same on every load.
function rng(seed) {
  return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
}

/**
 * Heads x tokens grid. Each row is one attention head, each column one token.
 * mode "dense": every cell computed. mode "redknot": global rows full, local rows = sink + window.
 * mode "token": each head's own important tokens, plus the union row a token-level method must recompute.
 * mode "head": same important tokens, but only the global rows are recomputed.
 */
export function headGrid(host, opts = {}) {
  const heads = opts.heads || 8, toks = opts.tokens || 48;
  const global = new Set(opts.global || [2, 6]);
  const sink = opts.sink ?? 2, win = opts.window || 8;
  const docs = opts.docs || null; // [{label, len}]
  const cell = 14, gap = 2, lw = 112, top = docs ? 26 : 6;
  const unionRow = opts.union;
  const rowsN = heads + (unionRow ? 1 : 0);
  const W = lw + toks * (cell + gap) + 8;
  const H = top + rowsN * (cell + gap) + (unionRow ? 12 : 0) + 6;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria || "attention heads by tokens" }, host);

  if (docs) {
    let c = 0;
    docs.forEach((d) => {
      const x0 = lw + c * (cell + gap), x1 = lw + (c + d.len) * (cell + gap) - gap;
      el("line", { x1: x0, x2: x1, y1: top - 8, y2: top - 8, stroke: cssVar("--ink-3"), "stroke-width": 1.5 }, svg);
      text(svg, (x0 + x1) / 2, top - 13, d.label, "tick", { "text-anchor": "middle" });
      c += d.len;
    });
  }

  const r = rng(opts.seed || 7);
  const important = [];
  for (let h = 0; h < heads; h++) {
    const s = new Set();
    for (let t = 0; t < toks; t++) if (r() < 0.16) s.add(t);
    important.push(s);
  }
  const union = new Set(important.flatMap((s) => [...s]));

  const rects = [];
  for (let h = 0; h < rowsN; h++) {
    const isUnion = unionRow && h === heads;
    const yy = top + h * (cell + gap) + (isUnion ? 12 : 0);
    const label = isUnion ? "recompute" : global.has(h) ? `head ${h} · global` : `head ${h} · local`;
    text(svg, lw - 8, yy + 11, label, "tick", { "text-anchor": "end", "font-weight": isUnion || global.has(h) ? 600 : 400 });
    for (let t = 0; t < toks; t++) {
      const x = lw + t * (cell + gap);
      const rc = el("rect", { x, y: yy, width: cell, height: cell, rx: 3 }, svg);
      rects.push({ rc, h, t, isUnion });
    }
  }

  const pale = "#f6eef0", lightRed = cssVar("--s-local"), red = cssVar("--red"), deep = cssVar("--red-deep");
  function paint(mode) {
    for (const { rc, h, t, isUnion } of rects) {
      let f = pale;
      if (mode === "dense") f = red;
      else if (mode === "redknot") {
        if (global.has(h)) f = red;
        else if (t < sink || t >= toks - win) f = red;
        else f = pale;
      } else if (mode === "token") {
        if (isUnion) f = union.has(t) ? deep : pale;
        else f = important[h].has(t) ? red : pale;
      } else if (mode === "head") {
        if (isUnion) f = pale;
        else if (global.has(h)) f = red;
        else f = important[h].has(t) ? lightRed : pale;
      }
      rc.setAttribute("fill", f);
    }
  }
  paint(opts.mode || "redknot");
  return { paint, unionShare: union.size / toks };
}

// Wire a two-or-more-button toggle to a callback.
export function toggle(host, options, onPick) {
  const box = document.createElement("div");
  box.className = "toggle";
  options.forEach(([key, label], i) => {
    const b = document.createElement("button");
    b.textContent = label;
    if (i === 0) b.className = "on";
    b.onclick = () => {
      box.querySelectorAll("button").forEach((x) => (x.className = ""));
      b.className = "on";
      onPick(key);
    };
    box.appendChild(b);
  });
  host.prepend(box);
}

// Copy buttons for BibTeX blocks.
export function copyButtons() {
  document.querySelectorAll("pre.bib").forEach((pre) => {
    const b = document.createElement("button");
    b.className = "copy";
    b.textContent = "Copy";
    b.onclick = () => navigator.clipboard.writeText(pre.querySelector("code").innerText).then(() => (b.textContent = "Copied"));
    pre.appendChild(b);
  });
}
