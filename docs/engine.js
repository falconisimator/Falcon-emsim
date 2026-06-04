// engine.js — Pyodide boot + solve + field rendering/animation.
// Exposes window.EM = { ready, solve(sceneDict), showResult(data,kind), play/stop }.

const EM = (window.EM = {
  ready: false,
  _solveFn: null,
  _data: null,
  _anim: null,
});

const statusEl = () => document.getElementById("status");

async function boot() {
  const pyodide = await loadPyodide();
  statusEl().textContent = "Loading numpy / scipy…";
  await pyodide.loadPackage(["numpy", "scipy", "micropip"]);
  statusEl().textContent = "Installing emsim…";
  const wheelUrl = new URL("wheels/emsim-0.1.0-py3-none-any.whl", location.href).href;
  const micropip = pyodide.pyimport("micropip");
  // Fetch the wheel with cache bypassed and install it from the Pyodide FS, so
  // a redeploy always loads the latest build (the wheel filename is fixed, so a
  // plain URL install would otherwise serve a browser-cached copy). Falls back
  // to a direct URL install if the emfs path is unsupported.
  let ok = false;
  try {
    const resp = await fetch(wheelUrl, { cache: "no-store" });
    const bytes = new Uint8Array(await resp.arrayBuffer());
    pyodide.FS.writeFile("/emsim-0.1.0-py3-none-any.whl", bytes);
    await micropip.install("emfs:/emsim-0.1.0-py3-none-any.whl", false, false);
    ok = true;
  } catch (e) {
    console.warn("fresh-wheel install failed; falling back to cached URL", e);
  }
  if (!ok) await micropip.install(wheelUrl, false, false); // deps=False
  EM._solveFn = pyodide.runPython("from emsim.web import solve_scene\nsolve_scene");
  EM.ready = true;
  statusEl().textContent = "Ready.";
  document.dispatchEvent(new Event("em-ready"));
}

// Solve a scene dict (built by the editor). Returns parsed result object.
EM.solve = function (sceneDict) {
  const json = EM._solveFn(JSON.stringify(sceneDict));
  const data = JSON.parse(json);
  data.nodes = Float64Array.from(data.nodes);
  data.tris = Int32Array.from(data.tris);
  data.region = Int32Array.from(data.region);
  data.a_re = Float64Array.from(data.a_re);
  data.a_im = Float64Array.from(data.a_im);
  data.javg_re = Float64Array.from(data.javg_re);  // complex terminal avg density I/A
  data.javg_im = Float64Array.from(data.javg_im);
  EM._data = data;
  EM._edges = computeEdges(data);  // material/conductor interface edges
  return data;
};

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
  Jn: ["x avg", true, 1],
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
      "the average ~ 0, so it briefly saturates). Click 'Σ over period' to sum it into the steady " +
      "under/over-utilization map (RMS |J| / (I/A)). Air is white.",
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
  else for (let e = 0; e < n; e++)
    m = Math.max(m, Math.hypot(d.Bx_re[e], d.Bx_im[e]) + Math.hypot(d.By_re[e], d.By_im[e]));
  return m;
}

EM.drawFrame = function (phi, accUtil) {
  const d = EM._data;
  if (!d) return;
  const kind = accUtil ? "Jn" : document.getElementById("field").value;
  const cv = document.getElementById("fieldCanvas");
  const W = (cv.width = cv.clientWidth), H = (cv.height = cv.clientHeight);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const ext = d.extent, sc = 0.92 * Math.min(W, H) / (2 * ext);
  // y-down to match the Konva editor's screen coordinates (otherwise off-centre
  // or rotated geometry appears vertically flipped vs. what was drawn)
  const X = (x) => W / 2 + x * sc, Y = (y) => H / 2 + y * sc;
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
};

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
    else EM._anim = null;
  };
  EM._anim = requestAnimationFrame(step);
};

boot().catch((e) => { statusEl().textContent = "Boot error: " + e; console.error(e); });
