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
  const wheel = new URL("wheels/emsim-0.1.0-py3-none-any.whl", location.href).href;
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(wheel, false, false); // deps=False
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

function fieldScale(d, kind) {
  let m = 1e-30;
  const n = d.J_re.length;
  if (kind === "J") for (let e = 0; e < n; e++) m = Math.max(m, Math.hypot(d.J_re[e], d.J_im[e]));
  else if (kind === "A") for (let e = 0; e < n; e++) m = Math.max(m, Math.hypot(d.Az_re[e], d.Az_im[e]));
  else for (let e = 0; e < n; e++)
    m = Math.max(m, Math.hypot(d.Bx_re[e], d.Bx_im[e]) + Math.hypot(d.By_re[e], d.By_im[e]));
  return m;
}

EM.drawFrame = function (phi) {
  const d = EM._data;
  if (!d) return;
  const kind = document.getElementById("field").value;
  const cv = document.getElementById("fieldCanvas");
  const W = (cv.width = cv.clientWidth), H = (cv.height = cv.clientHeight);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const ext = d.extent, sc = 0.92 * Math.min(W, H) / (2 * ext);
  const X = (x) => W / 2 + x * sc, Y = (y) => H / 2 - y * sc;
  const c = Math.cos(phi), s = Math.sin(phi), scale = fieldScale(d, kind);
  const n = d.tris, nodes = d.nodes;
  for (let e = 0; e < n.length; e += 3) {
    const t = e / 3;
    let col;
    if (kind === "J") {
      col = diverging((d.J_re[t] * c - d.J_im[t] * s) / scale);
    } else if (kind === "A") {
      col = diverging((d.Az_re[t] * c - d.Az_im[t] * s) / scale);
    } else {
      const bx = d.Bx_re[t] * c - d.Bx_im[t] * s, by = d.By_re[t] * c - d.By_im[t] * s;
      col = inferno(Math.hypot(bx, by) / scale);
    }
    const i0 = n[e] * 2, i1 = n[e + 1] * 2, i2 = n[e + 2] * 2;
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.beginPath();
    ctx.moveTo(X(nodes[i0]), Y(nodes[i0 + 1]));
    ctx.lineTo(X(nodes[i1]), Y(nodes[i1 + 1]));
    ctx.lineTo(X(nodes[i2]), Y(nodes[i2 + 1]));
    ctx.closePath();
    ctx.fill();
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

boot().catch((e) => { statusEl().textContent = "Boot error: " + e; console.error(e); });
