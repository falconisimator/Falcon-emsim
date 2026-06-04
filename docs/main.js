// emsim web app — boots the Python FEM core in Pyodide and renders results.
// Compute is 100% client-side; this file is served as a static asset.

const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let pyRun = null;       // Python entry point (set after load)
let current = null;     // last solve result
let anim = null;        // requestAnimationFrame handle
const TWO_PI = Math.PI * 2;

// Python module run inside Pyodide: build a 3-phase scene, solve with the
// gmsh-free backend, and return everything the page needs as JSON.
const PY_SETUP = `
import json, numpy as np
from emsim.geometry.shapes import Rectangle, Placement
from emsim.geometry.model import Conductor
from emsim.materials import COPPER
from emsim.scene import Scene
from emsim.post.fields import element_B, element_Jz
from emsim.mesh.gmsh_backend import KELVIN_TAG

def run(freq, current):
    bars = [Conductor(f"Phase {g}", Rectangle(0.01, 0.05), Placement(x, 0, 0),
                      COPPER, group=g, busbar=f"bb{i+1}")
            for i, (g, x) in enumerate(zip("ABC", (-0.03, 0.0, 0.03)))]
    sc = Scene(conductors=bars, frequency=freq, three_phase=True, line_current=current,
               boundary="dirichlet", mesh_backend="py")
    # web-tuned mesh: tighter air domain + coarser far field keeps the WASM solve fast
    ext0 = max(abs(c.placement.x) + abs(c.placement.y) + c.shape.bounding_radius()
               for c in bars)
    sc.domain_radius = 2.2 * ext0
    sc.lc_far = 0.5 * ext0
    sol = sc.solve(); res = sc.analyse(sol)
    mesh = sol.mesh
    phys = mesh.region_tag != KELVIN_TAG
    B = element_B(sol)[phys]; J = element_Jz(sol)[phys]
    ext = max(abs(c.placement.x) + abs(c.placement.y) + 1.6 * c.shape.bounding_radius()
              for c in sc.conductors)
    return json.dumps(dict(
        nodes=mesh.nodes.ravel().tolist(),
        tris=mesh.tris[phys][:, :3].ravel().tolist(),
        Bx_re=B[:,0].real.tolist(), Bx_im=B[:,0].imag.tolist(),
        By_re=B[:,1].real.tolist(), By_im=B[:,1].imag.tolist(),
        J_re=J.real.tolist(), J_im=J.imag.tolist(),
        extent=float(ext),
        conductors=[dict(name=c.name, group=c.group, I=abs(c.current),
                         phase=float(np.degrees(np.angle(c.current))),
                         loss=c.loss, share=c.share) for c in res.conductors],
        total_loss=res.total_loss,
    ))
`;

async function boot() {
  const pyodide = await loadPyodide();
  statusEl.textContent = "Loading numpy / scipy…";
  await pyodide.loadPackage(["numpy", "scipy", "micropip"]);
  statusEl.textContent = "Installing emsim…";
  const wheel = new URL("wheels/emsim-0.1.0-py3-none-any.whl", location.href).href;
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(wheel, false, false);  // keep_going, deps=False
  pyodide.runPython(PY_SETUP);
  pyRun = pyodide.globals.get("run");
  statusEl.textContent = "Ready.";
  for (const id of ["solve", "play", "static"]) document.getElementById(id).disabled = false;
  solve();
}

function solve() {
  stopAnim();
  const f = parseFloat(document.getElementById("freq").value);
  const i = parseFloat(document.getElementById("current").value);
  statusEl.textContent = "Meshing + solving…";
  setTimeout(() => {
    const t0 = performance.now();
    current = JSON.parse(pyRun(f, i));
    const ms = (performance.now() - t0).toFixed(0);
    current.nodes = Float64Array.from(current.nodes);
    current.tris = Int32Array.from(current.tris);
    statusEl.textContent = `Solved in ${ms} ms (${current.nodes.length / 2} nodes). ` +
      `Total loss ${current.total_loss.toPrecision(4)} W/m.`;
    fillTable();
    drawFrame(0);
  }, 10);
}

// --- field evaluation (instantaneous, computed in JS) ---------------------
function elementValues(phi) {
  const c = Math.cos(phi), s = Math.sin(phi);
  const kind = document.getElementById("field").value;
  const n = current.J_re.length;
  const v = new Float64Array(n);
  if (kind === "J") {
    for (let e = 0; e < n; e++) v[e] = current.J_re[e] * c - current.J_im[e] * s;
    return { v, diverging: true };
  }
  for (let e = 0; e < n; e++) {
    const bx = current.Bx_re[e] * c - current.Bx_im[e] * s;
    const by = current.By_re[e] * c - current.By_im[e] * s;
    v[e] = Math.hypot(bx, by);
  }
  return { v, diverging: false };
}

// scale across the whole period so the colour range is fixed
function fieldScale() {
  const kind = document.getElementById("field").value;
  let m = 1e-30;
  const n = current.J_re.length;
  if (kind === "J") {
    for (let e = 0; e < n; e++) m = Math.max(m, Math.hypot(current.J_re[e], current.J_im[e]));
  } else {
    for (let e = 0; e < n; e++)
      m = Math.max(m, Math.hypot(current.Bx_re[e], current.Bx_im[e]) +
                      Math.hypot(current.By_re[e], current.By_im[e]) * 0 +
                      Math.hypot(current.By_re[e], current.By_im[e]));
  }
  return m;
}

// colormaps
function inferno(t) {  // t in [0,1]
  t = Math.max(0, Math.min(1, t));
  const r = Math.min(255, 60 + 255 * Math.pow(t, 0.6));
  const g = Math.max(0, 255 * Math.pow(t, 2.2) - 20);
  const b = 120 * Math.pow(1 - t, 1.5) + 40 * t;
  return [r | 0, g | 0, b | 0];
}
function diverging(t) {  // t in [-1,1]: blue<0, white 0, red>0
  const a = Math.min(1, Math.abs(t));
  return t >= 0 ? [255, 255 * (1 - a), 255 * (1 - a)] : [255 * (1 - a), 255 * (1 - a), 255];
}

function drawFrame(phi) {
  if (!current) return;
  const W = canvas.width = canvas.clientWidth;
  const H = canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const ext = current.extent;
  const sc = 0.9 * Math.min(W, H) / (2 * ext);
  const ox = W / 2, oy = H / 2;
  const X = (x) => ox + x * sc, Y = (y) => oy - y * sc;

  const { v, diverging: div } = elementValues(phi);
  const scale = fieldScale();
  const nodes = current.nodes, tris = current.tris;
  for (let e = 0; e < tris.length; e += 3) {
    const i0 = tris[e] * 2, i1 = tris[e + 1] * 2, i2 = tris[e + 2] * 2;
    const val = v[e / 3];
    let col;
    if (div) col = diverging(val / scale);
    else col = inferno(val / scale);
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.beginPath();
    ctx.moveTo(X(nodes[i0]), Y(nodes[i0 + 1]));
    ctx.lineTo(X(nodes[i1]), Y(nodes[i1 + 1]));
    ctx.lineTo(X(nodes[i2]), Y(nodes[i2 + 1]));
    ctx.closePath();
    ctx.fill();
  }
}

// --- animation -------------------------------------------------------------
function startAnim() {
  let phi = 0;
  const tick = () => { phi = (phi + 0.08) % TWO_PI; drawFrame(phi); anim = requestAnimationFrame(tick); };
  anim = requestAnimationFrame(tick);
}
function stopAnim() { if (anim) cancelAnimationFrame(anim); anim = null; }

function fillTable() {
  const tbl = document.getElementById("results");
  tbl.hidden = false;
  const tb = tbl.querySelector("tbody");
  tb.innerHTML = "";
  for (const c of current.conductors) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${c.group}</td><td>${c.I.toFixed(0)}</td>` +
      `<td>${(c.share * 100).toFixed(0)}%</td><td>${c.loss.toPrecision(3)}</td>`;
    tb.appendChild(tr);
  }
}

document.getElementById("solve").onclick = solve;
document.getElementById("play").onclick = () => { stopAnim(); startAnim(); };
document.getElementById("static").onclick = () => { stopAnim(); drawFrame(0); };
document.getElementById("field").onchange = () => { if (current) drawFrame(0); };
window.addEventListener("resize", () => { if (current) drawFrame(0); });

boot().catch((e) => { statusEl.textContent = "Error: " + e; console.error(e); });
