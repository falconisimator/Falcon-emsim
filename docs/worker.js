// worker.js — Pyodide runtime off the main thread, so a solve never freezes the
// editor. The main thread (engine.js) posts {type:'solve', id, scene}; we post
// back {type:'status'|'ready'|'progress'|'result'|'error'}.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js");

let pyodide = null, solveFn = null, thermalFn = null;

async function boot() {
  pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/" });
  postMessage({ type: "status", text: "Loading numpy / scipy…" });
  await pyodide.loadPackage(["numpy", "scipy", "micropip"]);
  postMessage({ type: "status", text: "Installing emsim…" });
  const wheelUrl = new URL("wheels/emsim-0.1.0-py3-none-any.whl", location.href).href;
  const micropip = pyodide.pyimport("micropip");
  // fetch with cache bypassed + install from the Pyodide FS so a redeploy always
  // loads the latest build (fixed wheel filename would otherwise be cached).
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
  solveFn = pyodide.runPython("from emsim.web import solve_scene\nsolve_scene");
  thermalFn = pyodide.runPython("from emsim.web import solve_thermal\nsolve_thermal");
  postMessage({ type: "ready" });
}
boot().catch((e) => postMessage({ type: "error", text: "" + e }));

onmessage = (e) => {
  const m = e.data;
  if (m.type === "solve") {
    if (!solveFn) { postMessage({ type: "error", id: m.id, text: "runtime not ready" }); return; }
    try {
      postMessage({ type: "progress", id: m.id, text: "Meshing + solving (in-browser)…" });
      const json = solveFn(JSON.stringify(m.scene));
      postMessage({ type: "result", id: m.id, json });
    } catch (err) {
      postMessage({ type: "error", id: m.id, text: "" + err });
    }
  } else if (m.type === "thermal") {
    if (!thermalFn) { postMessage({ type: "error", id: m.id, text: "runtime not ready" }); return; }
    try {
      postMessage({ type: "progress", id: m.id, text: "Solving thermal…" });
      const json = thermalFn(JSON.stringify(m.params));
      postMessage({ type: "result", id: m.id, json });
    } catch (err) {
      postMessage({ type: "error", id: m.id, text: "" + err });
    }
  }
};
