// engine.js — Pyodide boot + solve + field rendering/animation.
// Exposes window.EM = { ready, solve(sceneDict), showResult(data,kind), play/stop }.

const EM = (window.EM = {
  ready: false,
  _solveFn: null,
  _data: null,
  _anim: null,
  _view: { zoom: 1, ox: 0, oy: 0 },   // field-view pan/zoom (screen px offset + factor)
});

const statusEl = () => document.getElementById("status");

// Pyodide runs in a Web Worker (worker.js) so a solve never blocks the main
// thread -- the editor stays interactive and a progress bar can animate.
const _worker = new Worker("worker.js?v=33");   // bump with worker.js changes
const _pending = new Map();
let _solveSeq = 0;
EM._onProgress = null;   // editor sets this to drive the progress bar / status

_worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") statusEl().textContent = m.text;
  else if (m.type === "ready") {
    EM.ready = true; statusEl().textContent = "Ready.";
    document.dispatchEvent(new Event("em-ready"));
  } else if (m.type === "progress") {
    if (EM._onProgress) EM._onProgress(m.text);
  } else if (m.type === "result") {
    const p = _pending.get(m.id); if (p) { _pending.delete(m.id); p.resolve(m.json); }
  } else if (m.type === "error") {
    const p = _pending.get(m.id);
    if (p) { _pending.delete(m.id); p.reject(new Error(m.text)); }
    else statusEl().textContent = "Boot error: " + m.text;
  }
};
_worker.onerror = (e) => { statusEl().textContent = "Worker error: " + (e.message || e); };

// Solve a scene dict (built by the editor) in the worker. Returns a Promise of
// the parsed result object (typed arrays for the hot paths).
EM.solve = function (sceneDict) {
  return new Promise((resolve, reject) => {
    const id = ++_solveSeq;
    _pending.set(id, { resolve, reject });
    _worker.postMessage({ type: "solve", id, scene: sceneDict });
  }).then((json) => {
    const data = JSON.parse(json);
    data.nodes = Float64Array.from(data.nodes);
    data.tris = Int32Array.from(data.tris);
    data.region = Int32Array.from(data.region);
    data.a_re = Float64Array.from(data.a_re);
    data.a_im = Float64Array.from(data.a_im);
    data.javg_re = Float64Array.from(data.javg_re);  // complex terminal avg density I/A
    data.javg_im = Float64Array.from(data.javg_im);
    data.loss_density = Float64Array.from(data.loss_density);  // per-element W/m^3 (time-avg)
    EM._data = data;
    EM._edges = computeEdges(data);  // material/conductor interface edges
    EM._util = null;                 // invalidate any cached period-sum map
    EM._view = { zoom: 1, ox: 0, oy: 0 };   // reset pan/zoom for the new solution
    EM._thermal = null;                      // EM changed -> thermal is stale
    return data;
  });
};

// Steady thermal solve in the worker, reusing the cached EM loss (no EM re-run).
EM.solveThermal = function (params) {
  return new Promise((resolve, reject) => {
    const id = ++_solveSeq;
    _pending.set(id, { resolve, reject });
    _worker.postMessage({ type: "thermal", id, params });
  }).then((json) => {
    const data = JSON.parse(json);
    if (data.error) throw new Error(data.error);
    data.T = Float64Array.from(data.T);
    if (data.vair) data.vair = Float64Array.from(data.vair);
    EM._thermal = data;
    return data;
  });
};

function _tbar(ctx, W, H, vmax, vmin, unit, fmt) {   // °C / m/s colorbar
  const bw = 16, bx = W - 70, by = 26, bh = Math.max(40, H - 70);
  for (let i = 0; i < bh; i++) {
    const col = inferno(1 - i / bh);
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; ctx.fillRect(bx, by + i, bw, 1);
  }
  ctx.strokeStyle = "#333"; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by + 0.5, bw, bh);
  ctx.fillStyle = "#222"; ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  for (const [yy, v] of [[by, vmax], [by + bh / 2, (vmax + vmin) / 2], [by + bh, vmin]])
    ctx.fillText(fmt(v), bx + bw + 5, yy);
  ctx.textBaseline = "alphabetic"; ctx.fillText(unit, bx - 2, by - 8);
}

// Thermal views: 'temp' (T on solids), 'air' (axial airspeed on the air),
// 'ir' (inter-busbar IR rays over faint geometry).
EM.drawThermal = function () {
  const d = EM._data, th = EM._thermal, cv = document.getElementById("thermalCanvas");
  if (!cv || !d || !th) return;
  const W = (cv.width = cv.clientWidth), H = (cv.height = cv.clientHeight);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const ext = d.extent, sc = 0.92 * Math.min(W, H) / (2 * ext);
  const X = (x) => W / 2 + x * sc, Y = (y) => H / 2 + y * sc;
  const n = d.tris, nodes = d.nodes, region = d.region;
  const mode = EM._thermalView || "temp";
  const tri = (e) => [n[e], n[e + 1], n[e + 2]];
  const fill = (e, val, colorOf) => {
    const [i0, i1, i2] = tri(e);
    const x0 = X(nodes[i0 * 2]), y0 = Y(nodes[i0 * 2 + 1]);
    const x1 = X(nodes[i1 * 2]), y1 = Y(nodes[i1 * 2 + 1]);
    const x2 = X(nodes[i2 * 2]), y2 = Y(nodes[i2 * 2 + 1]);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.closePath();
    if (colorOf) fillGouraud(ctx, x0, y0, x1, y1, x2, y2, val[i0], val[i1], val[i2], colorOf);
    else ctx.fill();
  };

  if (mode === "air") {
    const v = th.vair, vmax = Math.max(th.vmax, 1e-6);
    const colorOf = (s) => inferno(s / vmax);
    for (let e = 0; e < n.length; e += 3) {
      if (region[e / 3] >= 10) { ctx.fillStyle = "#3a3d42"; fill(e, null, null); }   // solids: grey
      else fill(e, v, colorOf);                                                       // air: speed
    }
  } else if (mode === "temp") {
    const tmin = th.Tamb, tmax = Math.max(th.Tmax, tmin + 1e-3);
    const colorOf = (t) => inferno((t - tmin) / (tmax - tmin));
    for (let e = 0; e < n.length; e += 3) if (region[e / 3] >= 10) fill(e, th.T, colorOf);
  } else { // ir / surface: faint solids so the overlay reads clearly
    ctx.fillStyle = "#e6e6e6";
    for (let e = 0; e < n.length; e += 3) if (region[e / 3] >= 10) fill(e, null, null);
  }

  // conductor outlines
  ctx.strokeStyle = "rgba(15,15,15,0.9)"; ctx.lineWidth = 1.3; ctx.beginPath();
  for (const [a, b] of EM._edges || []) {
    ctx.moveTo(X(nodes[a * 2]), Y(nodes[a * 2 + 1])); ctx.lineTo(X(nodes[b * 2]), Y(nodes[b * 2 + 1]));
  }
  ctx.stroke();

  if (mode === "ir" && th.ir_lines && th.ir_lines.length) {
    const wmax = th.ir_wmax || 1;
    for (const [x1, y1, x2, y2, w] of th.ir_lines) {
      const f = w / wmax;
      ctx.strokeStyle = `rgba(255,80,20,${Math.min(0.85, 0.15 + 0.6 * f)})`;
      ctx.lineWidth = 0.3 + 1.2 * f;
      ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke();
    }
  }

  // surface power transfer: each surface segment colored by how much heat it
  // sheds (|q''|, W/m^2) -- bright = a face dumping a lot of heat to the air.
  if (mode === "surface" && th.surf_segments && th.surf_segments.length) {
    const qmax = th.surf_qmax || 1;
    ctx.lineCap = "round";
    for (const s of th.surf_segments) {
      const f = Math.min(1, Math.abs(s[4]) / qmax);
      const col = inferno(f);
      ctx.strokeStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
      ctx.lineWidth = 1.5 + 3.5 * f;
      ctx.beginPath(); ctx.moveTo(X(s[0]), Y(s[1])); ctx.lineTo(X(s[2]), Y(s[3])); ctx.stroke();
    }
    ctx.lineCap = "butt";
  }

  if (mode === "air") _tbar(ctx, W, H, th.vmax, 0, "m/s", (x) => x.toFixed(2));
  else if (mode === "temp") _tbar(ctx, W, H, Math.max(th.Tmax, th.Tamb + 1e-3), th.Tamb, "°C", (x) => x.toFixed(0));
  else if (mode === "surface") _tbar(ctx, W, H, th.surf_qmax || 0, 0, "W/m²",
    (x) => (Math.abs(x) >= 1000 ? (x / 1000).toFixed(1) + "k" : x.toFixed(0)));
  updateThermalReadout();
};

// ---- thermal hover value readout (mirrors the EM field readout) -----------
// Sample the temperature / airspeed at a thermal-canvas pixel for the active
// view. Temperature is solved on the solids only; airspeed on the air only.
EM.thermalValueAt = function (px, py) {
  const d = EM._data, th = EM._thermal, cv = document.getElementById("thermalCanvas");
  if (!d || !th || !cv) return null;
  const mode = EM._thermalView || "temp";
  if (mode === "3d") return null;                  // 3D extrusion has its own transform
  const W = cv.width, H = cv.height, ext = d.extent, sc = 0.92 * Math.min(W, H) / (2 * ext);
  const xm = (px - W / 2) / sc, ym = (py - H / 2) / sc;   // thermal canvas has no pan/zoom
  if (mode === "surface") {   // sample the nearest surface segment's outgoing flux
    if (!th.surf_segments) return null;
    let best = null, bd = Infinity;
    for (const s of th.surf_segments) {
      const dx = s[2] - s[0], dy = s[3] - s[1], L2 = dx * dx + dy * dy || 1e-12;
      let tt = ((xm - s[0]) * dx + (ym - s[1]) * dy) / L2; tt = Math.max(0, Math.min(1, tt));
      const qx = s[0] + tt * dx, qy = s[1] + tt * dy, dist2 = (xm - qx) ** 2 + (ym - qy) ** 2;
      if (dist2 < bd) { bd = dist2; best = s; }
    }
    if (!best || bd > (0.025 * ext) ** 2) return null;   // only near a surface
    return { text: `q″ = ${fmtVal(best[4])} W/m²${best[4] < 0 ? " (in)" : ""}` };
  }
  const hit = triAt(d, xm, ym);
  if (!hit) return null;
  const solid = d.region[hit.t] >= 10;
  const n = d.tris, i0 = n[hit.t * 3], i1 = n[hit.t * 3 + 1], i2 = n[hit.t * 3 + 2];
  const interp = (arr) => hit.w0 * arr[i0] + hit.w1 * arr[i1] + hit.w2 * arr[i2];
  if (mode === "air") {
    if (solid || !th.vair) return { text: "solid" };
    return { text: `v = ${fmtVal(interp(th.vair))} m/s` };
  }
  // temp / ir: temperature lives on the solids
  if (!solid) return { text: "air" };
  return { text: `T = ${fmtVal(interp(th.T))} °C` };
};
function updateThermalReadout() {
  const el = document.getElementById("thermalReadout"), h = EM._thermalHover;
  if (!el) return;
  if (!h || !h.active || !EM._data || !EM._thermal) { el.style.display = "none"; return; }
  const r = EM.thermalValueAt(h.px, h.py);
  if (!r) { el.style.display = "none"; return; }
  el.textContent = r.text;
  el.style.left = h.ox + "px"; el.style.top = h.oy + "px"; el.style.display = "block";
}
(function bindThermalHover() {
  const cv = document.getElementById("thermalCanvas"),
        view = document.getElementById("thermalView");
  if (!cv || !view) return;
  EM._thermalHover = { active: false, px: 0, py: 0, ox: 0, oy: 0 };
  cv.addEventListener("mousemove", (e) => {
    if (EM._thermalView === "3d") { EM._thermalHover.active = false; updateThermalReadout(); return; }
    const r = cv.getBoundingClientRect(), o = view.getBoundingClientRect();
    EM._thermalHover.px = (e.clientX - r.left) * (cv.width / r.width);
    EM._thermalHover.py = (e.clientY - r.top) * (cv.height / r.height);
    EM._thermalHover.ox = e.clientX - o.left + 14; EM._thermalHover.oy = e.clientY - o.top + 14;
    EM._thermalHover.active = true;
    updateThermalReadout();
  });
  cv.addEventListener("mouseleave", () => { EM._thermalHover.active = false; updateThermalReadout(); });
})();

// edges shared by two triangles of different region = conductor / material outline
function computeEdges(d) {
  const N = d.nodes.length / 2, t = d.tris, region = d.region;
  const map = new Map();
  for (let e = 0; e < t.length; e += 3) {
    const r = region[e / 3], v = [t[e], t[e + 1], t[e + 2]];
    for (let k = 0; k < 3; k++) {
      const a = v[k], b = v[(k + 1) % 3], lo = Math.min(a, b), hi = Math.max(a, b);
      const key = lo * N + hi, rec = map.get(key);
      if (rec) rec.regs.push(r); else map.set(key, { a: lo, b: hi, regs: [r] });
    }
  }
  const out = [];
  for (const rec of map.values())
    if (rec.regs.length === 2 && rec.regs[0] !== rec.regs[1]) out.push([rec.a, rec.b]);
  return out;
}

// ---- field rendering on the result canvas --------------------------------
function inferno(t) {
  t = Math.max(0, Math.min(1, t));
  return [(60 + 255 * Math.pow(t, 0.6)) | 0,
          Math.max(0, 255 * Math.pow(t, 2.2) - 20) | 0,
          (120 * Math.pow(1 - t, 1.5) + 40 * t) | 0];
}
function diverging(t) {
  const a = Math.min(1, Math.abs(t));
  return t >= 0 ? [255, (255 * (1 - a)) | 0, (255 * (1 - a)) | 0]
                : [(255 * (1 - a)) | 0, (255 * (1 - a)) | 0, 255];
}
const rgb = (col) => `rgb(${col[0]},${col[1]},${col[2]})`;

// Smooth (Gouraud) fill of the current triangle path given the 3 nodal values.
// The field is linear across the element, so a single canvas linear gradient
// along the value-gradient direction reproduces it exactly (no facets).
function fillGouraud(ctx, x0, y0, x1, y1, x2, y2, v0, v1, v2, colorOf) {
  const vmin = Math.min(v0, v1, v2), vmax = Math.max(v0, v1, v2);
  if (vmax - vmin < 1e-12) { ctx.fillStyle = rgb(colorOf(v0)); ctx.fill(); return; }
  const det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
  if (Math.abs(det) < 1e-9) { ctx.fillStyle = rgb(colorOf((v0 + v1 + v2) / 3)); ctx.fill(); return; }
  const gx = ((v1 - v0) * (y2 - y0) - (v2 - v0) * (y1 - y0)) / det;   // value gradient
  const gy = ((v2 - v0) * (x1 - x0) - (v1 - v0) * (x2 - x0)) / det;
  const gmag = Math.hypot(gx, gy);
  if (gmag < 1e-12) { ctx.fillStyle = rgb(colorOf((v0 + v1 + v2) / 3)); ctx.fill(); return; }
  const dx = gx / gmag, dy = gy / gmag;
  const s0 = x0 * dx + y0 * dy, s1 = x1 * dx + y1 * dy, s2 = x2 * dx + y2 * dy;
  const smin = Math.min(s0, s1, s2), smax = Math.max(s0, s1, s2);
  const gr = ctx.createLinearGradient(dx * smin, dy * smin, dx * smax, dy * smax);
  for (let f = 0; f <= 1.0001; f += 0.25) {
    gr.addColorStop(Math.min(1, f), rgb(colorOf(vmin + f * (vmax - vmin))));
  }
  ctx.fillStyle = gr; ctx.fill();
}

// [unit, diverging?, display-scale factor]
const FIELD_META = {
  J: ["A/mm²", true, 1e-6], B: ["T", false, 1], A: ["Wb/m", true, 1],
  Jn: ["x avg", true, 1], P: ["W/m³", false, 1],
};

EM.DESCRIPTIONS = {
  J: "Current density J_z(t) [A/mm2]. Instantaneous axial current, Re(J e^{j*phi}). " +
     "Red = current out of the page (+z), blue = into the page (-z), white ~ 0. " +
     "At AC it crowds toward surfaces and edges facing other phases (skin & proximity effect). " +
     "Light-blue lines are flux lines (contours of A_z).",
  B: "Flux density |B(t)| [T]. Magnitude of the in-plane magnetic field. " +
     "Brighter = stronger; it peaks at conductor surfaces and between opposing currents and " +
     "decays with distance. The light-blue flux lines run parallel to B.",
  A: "Vector potential A_z(t) [Wb/m]. Its contours ARE the magnetic field lines - closer spacing " +
     "means a stronger field. Diverging colors show the sign of the instantaneous value.",
  Jn: "Current density relative to the terminal's average AT THIS INSTANT: J_z(t) / (i(t)/A). " +
      "Animate it. White = 1 (equal to the instantaneous average), red >1 = above, blue <1 = below. " +
      "Because eddy currents lag, the pattern shifts through the cycle (near the current zero-crossing " +
      "the average ~ 0, so it briefly saturates). For the steady picture select 'J / avg over period'. " +
      "Air is white.",
  util: "Period-summed utilization: RMS |J| over one cycle / (I/A) = the steady under/over-utilization " +
      "map. Selecting it sums the instantaneous current over a full period (you'll see it build up, then " +
      "settle). White = 1 (carrying its fair share), red >1 = over-utilized, blue <1 = under-utilized / " +
      "'slow' copper. Air is white. Re-select or Animate to recompute.",
  P: "Ohmic loss density p = ½|J|²/σ [W/m³], time-averaged over the cycle - where the busbar heats up. " +
     "The terminal voltage gradient V̇/L is uniform per phase, but skin & proximity effect crowd J (and " +
     "loss ~ J²) onto surfaces and facing edges, so the loss density is far from uniform. Integrated over " +
     "the cross-section it gives the total loss in W/m. Air carries no current and is white.",
};

function drawColorbar(ctx, W, H, kind, scale) {
  const norm = kind === "Jn";
  const meta = FIELD_META[kind], unit = meta[0], factor = meta[2];
  const div = norm ? true : meta[1] === true;
  const bw = 16, bx = W - 70, by = 26, bh = Math.max(40, H - 70);
  for (let i = 0; i < bh; i++) {
    const f = 1 - i / bh;                       // top = 1, bottom = 0
    const col = div ? diverging(2 * f - 1) : inferno(f);
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.fillRect(bx, by + i, bw, 1);
  }
  ctx.strokeStyle = "#333"; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by + 0.5, bw, bh);
  ctx.fillStyle = "#222"; ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  const fmt = (v) => (v === 0 ? "0" : Math.abs(v) >= 100 || Math.abs(v) < 0.1 ? v.toExponential(1) : v.toFixed(2));
  let ticks;
  if (norm) ticks = [[by, 2], [by + bh / 2, 1], [by + bh, 0]];   // centered at 1 (fair share)
  else if (div) ticks = [[by, scale * factor], [by + bh / 2, 0], [by + bh, -scale * factor]];
  else ticks = [[by, scale * factor], [by + bh / 2, scale * factor / 2], [by + bh, 0]];
  for (const [y, v] of ticks) ctx.fillText(fmt(v), bx + bw + 5, y);
  ctx.textBaseline = "alphabetic";
  ctx.fillText(unit, bx - 2, by - 8);
}

function fieldScale(d, kind) {
  if (kind === "Jn") return 1;  // centered-at-1 ratio, drawn directly
  let m = 1e-30;
  const n = d.J_re.length;
  if (kind === "J") for (let e = 0; e < n; e++) m = Math.max(m, Math.hypot(d.J_re[e], d.J_im[e]));
  else if (kind === "A") for (let e = 0; e < n; e++) m = Math.max(m, Math.hypot(d.Az_re[e], d.Az_im[e]));
  else if (kind === "P") for (let e = 0; e < d.loss_density.length; e++) m = Math.max(m, d.loss_density[e]);
  else for (let e = 0; e < n; e++)
    m = Math.max(m, Math.hypot(d.Bx_re[e], d.Bx_im[e]) + Math.hypot(d.By_re[e], d.By_im[e]));
  return m;
}

EM.drawFrame = function (phi, accUtil) {
  const d = EM._data;
  if (!d) return;
  // "util" dropdown entry = the steady period-sum utilization map (computed by
  // sumOverPeriod and cached in EM._util); render it via the accUtil path.
  const sel = document.getElementById("field").value;
  if (!accUtil && sel === "util" && EM._util && EM._util.length === d.tris.length / 3) accUtil = EM._util;
  const kind = accUtil ? "Jn" : sel;
  const cv = document.getElementById("fieldCanvas");
  const W = (cv.width = cv.clientWidth), H = (cv.height = cv.clientHeight);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  EM._lastPhi = phi;
  const ext = d.extent, sc = 0.92 * Math.min(W, H) / (2 * ext);
  const vz = EM._view.zoom, vox = EM._view.ox, voy = EM._view.oy;
  // y-down to match the Konva editor's screen coordinates (otherwise off-centre
  // or rotated geometry appears vertically flipped vs. what was drawn). Pan/zoom
  // (vox,voy,vz) let you scroll into the solution; the colorbar stays fixed.
  const X = (x) => W / 2 + vox + x * sc * vz, Y = (y) => H / 2 + voy + y * sc * vz;
  const c = Math.cos(phi), s = Math.sin(phi), scale = fieldScale(d, kind);
  const n = d.tris, nodes = d.nodes, NE = n.length / 3, NN = nodes.length / 2;

  // 1) per-element field value + validity (air -> white) + value->colour map
  const ev = new Float64Array(NE), valid = new Uint8Array(NE);
  let colorOf;
  if (accUtil) {
    for (let t = 0; t < NE; t++) { ev[t] = accUtil[t]; valid[t] = accUtil[t] >= 0 ? 1 : 0; }
    colorOf = (v) => diverging(Math.max(-1, Math.min(1, v - 1)));
  } else if (kind === "J") {   // current density: no current in air -> hard edge at the bar
    for (let t = 0; t < NE; t++) {
      ev[t] = d.J_re[t] * c - d.J_im[t] * s;
      valid[t] = d.region[t] >= 10 ? 1 : 0;   // conductor tags are >= 10; air -> white
    }
    colorOf = (v) => diverging(v / scale);
  } else if (kind === "Jn") {   // instantaneous J / average-at-this-instant (i(t)/A)
    for (let t = 0; t < NE; t++) {
      const ar = d.javg_re[t], ai = d.javg_im[t];
      if (ar === 0 && ai === 0) { valid[t] = 0; ev[t] = 1; }
      else {
        const amp = Math.hypot(ar, ai), fl = 0.12 * amp;
        let denom = ar * c - ai * s;
        if (Math.abs(denom) < fl) denom = (denom < 0 ? -1 : 1) * fl;
        ev[t] = (d.J_re[t] * c - d.J_im[t] * s) / denom; valid[t] = 1;
      }
    }
    colorOf = (v) => diverging(Math.max(-1, Math.min(1, v - 1)));
  } else if (kind === "A") {
    for (let t = 0; t < NE; t++) { ev[t] = d.Az_re[t] * c - d.Az_im[t] * s; valid[t] = 1; }
    colorOf = (v) => diverging(v / scale);
  } else if (kind === "P") {   // ohmic loss density (time-averaged, phase-independent)
    for (let t = 0; t < NE; t++) { ev[t] = d.loss_density[t]; valid[t] = d.region[t] >= 10 ? 1 : 0; }
    colorOf = (v) => inferno(v / scale);
  } else {  // |B|
    for (let t = 0; t < NE; t++) {
      const bx = d.Bx_re[t] * c - d.Bx_im[t] * s, by = d.By_re[t] * c - d.By_im[t] * s;
      ev[t] = Math.hypot(bx, by); valid[t] = 1;
    }
    colorOf = (v) => inferno(v / scale);
  }

  // 2) average element values onto nodes (area-weighted, valid elements only)
  const nv = new Float64Array(NN), nw = new Float64Array(NN);
  for (let e = 0; e < n.length; e += 3) {
    const t = e / 3; if (!valid[t]) continue;
    const i0 = n[e], i1 = n[e + 1], i2 = n[e + 2];
    const ax = nodes[i0 * 2], ay = nodes[i0 * 2 + 1], bx = nodes[i1 * 2], by = nodes[i1 * 2 + 1];
    const cx = nodes[i2 * 2], cy = nodes[i2 * 2 + 1];
    const w = Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) + 1e-30;  // ~2*area
    nv[i0] += w * ev[t]; nw[i0] += w;
    nv[i1] += w * ev[t]; nw[i1] += w;
    nv[i2] += w * ev[t]; nw[i2] += w;
  }
  for (let i = 0; i < NN; i++) if (nw[i] > 0) nv[i] /= nw[i];

  // 3) smooth (Gouraud) fill per triangle; air -> white
  for (let e = 0; e < n.length; e += 3) {
    const t = e / 3, i0 = n[e], i1 = n[e + 1], i2 = n[e + 2];
    const x0 = X(nodes[i0 * 2]), y0 = Y(nodes[i0 * 2 + 1]);
    const x1 = X(nodes[i1 * 2]), y1 = Y(nodes[i1 * 2 + 1]);
    const x2 = X(nodes[i2 * 2]), y2 = Y(nodes[i2 * 2 + 1]);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.closePath();
    if (!valid[t]) { ctx.fillStyle = "#fff"; ctx.fill(); }
    else fillGouraud(ctx, x0, y0, x1, y1, x2, y2, nv[i0], nv[i1], nv[i2], colorOf);
  }

  // flux lines: contours of instantaneous A_z = Re(a e^{j phi}) (marching triangles)
  const ar = d.a_re, ai = d.a_im, nN = ar.length;
  let amp = 1e-30;
  const vv = new Float64Array(nN);
  for (let i = 0; i < nN; i++) { vv[i] = ar[i] * c - ai[i] * s; amp = Math.max(amp, Math.hypot(ar[i], ai[i])); }
  ctx.strokeStyle = "rgba(90,170,255,0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const LEV = 14;
  for (let li = 1; li < LEV; li++) {
    const L = -amp + (2 * amp * li) / LEV;
    for (let e = 0; e < n.length; e += 3) {
      const id = [n[e], n[e + 1], n[e + 2]];
      const va = vv[id[0]] - L, vb = vv[id[1]] - L, vc = vv[id[2]] - L;
      const pts = [];
      const edge = (i, j, vi, vj) => {
        if ((vi < 0) !== (vj < 0)) {
          const t = vi / (vi - vj);
          pts.push([nodes[id[i] * 2] + t * (nodes[id[j] * 2] - nodes[id[i] * 2]),
                    nodes[id[i] * 2 + 1] + t * (nodes[id[j] * 2 + 1] - nodes[id[i] * 2 + 1])]);
        }
      };
      edge(0, 1, va, vb); edge(1, 2, vb, vc); edge(2, 0, vc, va);
      if (pts.length === 2) { ctx.moveTo(X(pts[0][0]), Y(pts[0][1])); ctx.lineTo(X(pts[1][0]), Y(pts[1][1])); }
    }
  }
  ctx.stroke();

  // conductor / material outlines on top
  ctx.strokeStyle = "rgba(15,15,15,0.9)";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (const [a, b] of EM._edges || []) {
    ctx.moveTo(X(nodes[a * 2]), Y(nodes[a * 2 + 1]));
    ctx.lineTo(X(nodes[b * 2]), Y(nodes[b * 2 + 1]));
  }
  ctx.stroke();

  drawColorbar(ctx, W, H, kind, scale);

  // keep the rendered field around so the hover readout can sample it (and stay
  // live during animation); nv/valid are exactly what was drawn this frame.
  EM._nv = nv; EM._valid = valid; EM._lastKind = kind; EM._isUtil = !!accUtil;
  updateHoverReadout();
};

// ---- hover value readout -------------------------------------------------
const fmtVal = (v) => {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000 || a < 0.01) return v.toExponential(2);
  return v.toFixed(a >= 100 ? 0 : a >= 1 ? 2 : 3);
};
function readoutText(kind, val, air, isUtil) {
  if (kind === "J") return air ? "Jz = 0 (air)" : `Jz = ${fmtVal(val * 1e-6)} A/mm²`;
  if (kind === "B") return `|B| = ${fmtVal(val)} T`;
  if (kind === "A") return `Az = ${fmtVal(val)} Wb/m`;
  if (kind === "P") return air ? "p = 0 (air)" : `p = ${fmtVal(val)} W/m³`;
  if (kind === "Jn") return air ? "air" : `${isUtil ? "util" : "J/avg"} = ${fmtVal(val)}×`;
  return "";
}
// element containing world point (px,py) + its barycentric weights, or null.
function triAt(d, px, py) {
  const n = d.tris, nodes = d.nodes, NE = n.length / 3, eps = 1e-7;
  for (let t = 0; t < NE; t++) {
    const i0 = n[t * 3], i1 = n[t * 3 + 1], i2 = n[t * 3 + 2];
    const ax = nodes[i0 * 2], ay = nodes[i0 * 2 + 1], bx = nodes[i1 * 2], by = nodes[i1 * 2 + 1];
    const cx = nodes[i2 * 2], cy = nodes[i2 * 2 + 1];
    const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (Math.abs(det) < 1e-30) continue;
    const w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det;
    const w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det;
    const w2 = 1 - w0 - w1;
    if (w0 >= -eps && w1 >= -eps && w2 >= -eps) return { t, w0, w1, w2 };
  }
  return null;
}
// value at canvas pixel (px,py) for the field as currently drawn.
EM.valueAt = function (px, py) {
  const d = EM._data, nv = EM._nv, valid = EM._valid;
  if (!d || !nv) return null;
  const cv = document.getElementById("fieldCanvas");
  const W = cv.width, H = cv.height, ext = d.extent, sc = 0.92 * Math.min(W, H) / (2 * ext);
  const vz = EM._view.zoom;
  const xm = (px - W / 2 - EM._view.ox) / (sc * vz), ym = (py - H / 2 - EM._view.oy) / (sc * vz);
  const hit = triAt(d, xm, ym);
  if (!hit) return null;
  if (!valid[hit.t]) return { text: readoutText(EM._lastKind, 0, true, EM._isUtil) };
  const n = d.tris, i0 = n[hit.t * 3], i1 = n[hit.t * 3 + 1], i2 = n[hit.t * 3 + 2];
  const val = hit.w0 * nv[i0] + hit.w1 * nv[i1] + hit.w2 * nv[i2];
  return { text: readoutText(EM._lastKind, val, false, EM._isUtil) };
};
function updateHoverReadout() {
  const el = document.getElementById("hoverReadout"), h = EM._hover;
  if (!el) return;
  if (!h || !h.active || !EM._data) { el.style.display = "none"; return; }
  const r = EM.valueAt(h.px, h.py);
  if (!r) { el.style.display = "none"; return; }
  el.textContent = r.text;
  el.style.left = h.ox + "px"; el.style.top = h.oy + "px"; el.style.display = "block";
}
EM.redraw = function () { if (EM._data) EM.drawFrame(EM._lastPhi || 0); };
(function bindFieldInteraction() {
  const cv = document.getElementById("fieldCanvas"), overlay = document.getElementById("fieldOverlay");
  if (!cv || !overlay) return;
  EM._hover = { active: false, px: 0, py: 0, ox: 0, oy: 0 };
  let pan = null;
  cv.addEventListener("mousemove", (e) => {
    const r = cv.getBoundingClientRect();
    if (pan) {   // drag = pan the solution
      EM._view.ox = pan.ox + (e.clientX - pan.x) * (cv.width / r.width);
      EM._view.oy = pan.oy + (e.clientY - pan.y) * (cv.height / r.height);
      EM.redraw();
      return;
    }
    const o = overlay.getBoundingClientRect();
    EM._hover.px = (e.clientX - r.left) * (cv.width / r.width);
    EM._hover.py = (e.clientY - r.top) * (cv.height / r.height);
    EM._hover.ox = e.clientX - o.left + 14; EM._hover.oy = e.clientY - o.top + 14;
    EM._hover.active = true;
    updateHoverReadout();
  });
  cv.addEventListener("mouseleave", () => { EM._hover.active = false; updateHoverReadout(); });
  cv.addEventListener("mousedown", (e) => {
    pan = { x: e.clientX, y: e.clientY, ox: EM._view.ox, oy: EM._view.oy };
    EM._hover.active = false; updateHoverReadout(); cv.style.cursor = "grabbing";
  });
  window.addEventListener("mouseup", () => { pan = null; cv.style.cursor = ""; });
  cv.addEventListener("dblclick", () => { EM._view = { zoom: 1, ox: 0, oy: 0 }; EM.redraw(); });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (!EM._data) return;
    const r = cv.getBoundingClientRect();
    const px = (e.clientX - r.left) * (cv.width / r.width), py = (e.clientY - r.top) * (cv.height / r.height);
    const W = cv.width, H = cv.height, ext = EM._data.extent, sc = 0.92 * Math.min(W, H) / (2 * ext);
    const v = EM._view, oldZ = v.zoom;
    let z = e.deltaY > 0 ? oldZ / 1.12 : oldZ * 1.12;
    z = Math.max(0.5, Math.min(20, z));
    const xw = (px - W / 2 - v.ox) / (sc * oldZ), yw = (py - H / 2 - v.oy) / (sc * oldZ);  // keep point under cursor
    v.zoom = z; v.ox = px - W / 2 - xw * sc * z; v.oy = py - H / 2 - yw * sc * z;
    EM.redraw();
  }, { passive: false });
})();

EM.play = function () {
  EM.stop();
  const start = performance.now();
  const tick = (now) => {
    const period = +(document.getElementById("speed")?.value || 5000);  // ms per cycle
    const phi = (2 * Math.PI * ((now - start) % period)) / period;
    EM.drawFrame(phi);
    EM._anim = requestAnimationFrame(tick);
  };
  EM._anim = requestAnimationFrame(tick);
};
EM.stop = function () { if (EM._anim) cancelAnimationFrame(EM._anim); EM._anim = null; };

// Sum the instantaneous current over one period -> steady under/over-utilization.
// Accumulates J(t)^2 frame by frame (visibly building up); converges to the RMS
// ratio |J|/(I/A): 1 = fair share, >1 over-utilized, <1 under-utilized.
EM.sumOverPeriod = function (frames = 90) {
  EM.stop();
  const d = EM._data; if (!d) return;
  const n = d.tris.length / 3;
  const acc = new Float64Array(n);
  const util = new Float64Array(n);
  let k = 0;
  const step = () => {
    const phi = (2 * Math.PI * k) / frames, c = Math.cos(phi), s = Math.sin(phi);
    for (let t = 0; t < n; t++) {
      const j = d.J_re[t] * c - d.J_im[t] * s;
      acc[t] += j * j;
      const amp = Math.hypot(d.javg_re[t], d.javg_im[t]);
      util[t] = amp > 0 ? Math.sqrt(2 * acc[t] / (k + 1)) / amp : -1;  // RMS|J| / (I/A)
    }
    EM.drawFrame(0, util);
    k++;
    if (k <= frames) EM._anim = requestAnimationFrame(step);
    else { EM._anim = null; EM._util = util; }   // cache the steady map for static redraws
  };
  EM._anim = requestAnimationFrame(step);
};
