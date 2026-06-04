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
  EM._data = data;
  return data;
};

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
