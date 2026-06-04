// editor.js — interactive Konva geometry editor wired to the Pyodide engine.

const PPM = 4;          // pixels per millimetre at scale 1
const GRID = 5;         // grid / snap step in mm
const PHASES = ["A", "B", "C"];
const PHASE_COLOR = { A: "#d65f5f", B: "#5f9ed6", C: "#5fd68a" };
const MAT = {
  Copper: { name: "copper", sigma: 5.8e7, mu_r: 1.0 },
  Aluminium: { name: "aluminium", sigma: 3.5e7, mu_r: 1.0 },
  Steel: { name: "steel", sigma: 1.0e7, mu_r: 200.0 },
};
const snap = (v) => Math.round(v / GRID) * GRID;

let conductors = [];    // model: list of {id,name,type,w,h,r,x,y,rot,group,material,busbar,node}
let selected = null;
let uid = 1;
let editBusbar = null;  // busbar id under isolation editing, or null

// ---- Konva stage ----------------------------------------------------------
const editorDiv = document.getElementById("editor");
const stage = new Konva.Stage({ container: "editor", width: editorDiv.clientWidth, height: editorDiv.clientHeight });
const gridLayer = new Konva.Layer({ listening: false });
const layer = new Konva.Layer();
stage.add(gridLayer); stage.add(layer);
const tr = new Konva.Transformer({ rotationSnaps: [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, -15, -30, -45, -60, -75, -90, -105, -120, -135, -150, -165], rotateAnchorOffset: 24 });
layer.add(tr);

function recenter() {
  stage.width(editorDiv.clientWidth); stage.height(editorDiv.clientHeight);
  stage.scale({ x: PPM, y: PPM });
  stage.position({ x: stage.width() / 2, y: stage.height() / 2 });
  drawGrid();
}
function drawGrid() {
  gridLayer.destroyChildren();
  const wmm = stage.width() / PPM, hmm = stage.height() / PPM;
  const ext = Math.max(wmm, hmm);
  for (let x = -snap(ext); x <= ext; x += GRID) {
    gridLayer.add(new Konva.Line({ points: [x, -ext, x, ext],
      stroke: x === 0 ? "#9aa0a6" : "#e3e6ea", strokeWidth: (x % 25 === 0 ? 0.4 : 0.2) / PPM }));
  }
  for (let y = -snap(ext); y <= ext; y += GRID) {
    gridLayer.add(new Konva.Line({ points: [-ext, y, ext, y],
      stroke: y === 0 ? "#9aa0a6" : "#e3e6ea", strokeWidth: (y % 25 === 0 ? 0.4 : 0.2) / PPM }));
  }
  gridLayer.batchDraw();
}

// ---- model <-> Konva ------------------------------------------------------
function makeNode(c) {
  const common = { x: c.x, y: c.y, rotation: c.rot, draggable: true,
                   fill: PHASE_COLOR[c.group] || "#888", stroke: "#222", strokeWidth: 0.4 };
  let node;
  if (c.type === "rect")
    node = new Konva.Rect({ ...common, width: c.w, height: c.h, offsetX: c.w / 2, offsetY: c.h / 2 });
  else
    node = new Konva.Circle({ ...common, radius: c.r });
  node.on("click tap", () => select(c));
  node.on("dblclick dbltap", () => enterIsolation(c.busbar));
  node.on("dragmove", () => { node.x(snap(node.x())); node.y(snap(node.y())); });
  node.on("dragend transformend", () => { syncFromNode(c); if (selected === c) fillProps(c); });
  node.on("transform", () => applyTransform(c));
  c.node = node;
  layer.add(node);
  return node;
}
function applyTransform(c) {           // live during resize: fold scale into size
  const n = c.node;
  if (c.type === "rect") {
    const w = Math.max(GRID, n.width() * n.scaleX()), h = Math.max(GRID, n.height() * n.scaleY());
    n.scaleX(1); n.scaleY(1); n.width(w); n.height(h); n.offset({ x: w / 2, y: h / 2 });
  } else {
    const r = Math.max(GRID, n.radius() * n.scaleX());
    n.scaleX(1); n.scaleY(1); n.radius(r);
  }
}
function syncFromNode(c) {              // commit Konva node back to the model (snapped)
  const n = c.node;
  c.x = snap(n.x()); c.y = snap(n.y()); c.rot = Math.round(n.rotation());
  n.x(c.x); n.y(c.y);
  if (c.type === "rect") {
    c.w = snap(n.width()); c.h = snap(n.height());
    n.width(c.w); n.height(c.h); n.offset({ x: c.w / 2, y: c.h / 2 });
  } else {
    c.r = snap(n.radius()); n.radius(c.r);
  }
  layer.batchDraw();
}

function phaseOfBusbar(bb) {
  const m = conductors.find((c) => c.busbar === bb);
  return m ? m.group : "A";
}
function addConductor(type, group) {
  // inside isolation, new shapes join the edited busbar (same phase)
  const busbar = editBusbar || "bb" + uid;
  const c = { id: uid, name: "C" + uid, type, w: 40, h: 10, r: 10, x: 0, y: 0, rot: 0,
              group: editBusbar ? phaseOfBusbar(editBusbar) : (group || "A"),
              material: "Copper", busbar };
  uid++;
  conductors.push(c);
  makeNode(c);
  if (editBusbar) applyIsolation();
  select(c);
  layer.batchDraw();
}

// ---- busbar isolation (double-click to edit a busbar) ---------------------
function enterIsolation(bb) {
  editBusbar = bb;
  applyIsolation();
  document.getElementById("isoBadge").textContent =
    `editing busbar (phase ${phaseOfBusbar(bb)}) — + Bar adds into it, Esc to exit`;
}
function exitIsolation() {
  editBusbar = null;
  conductors.forEach((c) => { c.node.opacity(1); c.node.draggable(true); c.node.listening(true); });
  document.getElementById("isoBadge").textContent = "";
  layer.batchDraw();
}
function applyIsolation() {
  conductors.forEach((c) => {
    const member = c.busbar === editBusbar;
    c.node.opacity(member ? 1 : 0.22);
    c.node.draggable(member);
    c.node.listening(member);
  });
  layer.batchDraw();
}

function select(c) {
  selected = c;
  tr.nodes(c ? [c.node] : []);
  // circles: keep aspect (radius); rects: free
  tr.keepRatio(c && c.type === "circle");
  tr.enabledAnchors(c && c.type === "circle"
    ? ["top-left", "top-right", "bottom-left", "bottom-right"] : undefined);
  document.getElementById("propPanel").hidden = !c;
  if (c) fillProps(c);
  layer.batchDraw();
}

stage.on("click tap", (e) => { if (e.target === stage) select(null); });
stage.on("dblclick dbltap", (e) => { if (e.target === stage) exitIsolation(); });

// ---- properties panel -----------------------------------------------------
const $ = (id) => document.getElementById(id);
function fillProps(c) {
  $("pW").value = c.type === "rect" ? c.w : c.r;
  $("pH").value = c.h;
  $("pHlabel").parentElement.style.display = c.type === "rect" ? "" : "none";
  $("pX").value = c.x; $("pY").value = c.y; $("pRot").value = c.rot;
  $("pPhase").value = c.group; $("pMat").value = c.material;
}
function readProps() {
  const c = selected; if (!c) return;
  if (c.type === "rect") { c.w = snap(+$("pW").value); c.h = snap(+$("pH").value); }
  else c.r = snap(+$("pW").value);
  c.x = snap(+$("pX").value); c.y = snap(+$("pY").value); c.rot = +$("pRot").value;
  c.group = $("pPhase").value; c.material = $("pMat").value;
  // rebuild node geometry
  const n = c.node;
  n.x(c.x); n.y(c.y); n.rotation(c.rot); n.fill(PHASE_COLOR[c.group] || "#888");
  if (c.type === "rect") { n.width(c.w); n.height(c.h); n.offset({ x: c.w / 2, y: c.h / 2 }); }
  else n.radius(c.r);
  layer.batchDraw();
}
["pW", "pH", "pX", "pY", "pRot", "pPhase", "pMat"].forEach((id) =>
  $(id).addEventListener("input", readProps));

// ---- scene serialisation (io format, metres) ------------------------------
function toSceneDict() {
  return {
    format: 1, frequency: +$("freq").value, three_phase: $("threephase").checked,
    line_current: +$("current").value, boundary: "dirichlet", order: 1,
    domain_radius: 0, lc_surface: 0, lc_far: 0, group_currents: {},
    conductors: conductors.map((c) => ({
      name: c.name,
      shape: c.type === "rect"
        ? { type: "rect", width: c.w / 1000, height: c.h / 1000 }
        : { type: "circle", radius: c.r / 1000 },
      placement: [c.x / 1000, c.y / 1000, c.rot],
      material: MAT[c.material], group: c.group, busbar: c.busbar,
    })),
  };
}
function loadSceneDict(d) {
  conductors.forEach((c) => c.node.destroy());
  conductors = []; selected = null; tr.nodes([]);
  uid = 1;
  $("freq").value = d.frequency; $("threephase").checked = d.three_phase;
  $("current").value = d.line_current;
  const matName = (m) => ({ copper: "Copper", aluminium: "Aluminium", steel: "Steel" }[m.name] || "Copper");
  for (const cd of d.conductors) {
    const s = cd.shape;
    const c = { id: uid, name: cd.name, type: s.type === "circle" ? "circle" : "rect",
      w: (s.width || 0) * 1000, h: (s.height || 0) * 1000, r: (s.radius || 0) * 1000,
      x: cd.placement[0] * 1000, y: cd.placement[1] * 1000, rot: cd.placement[2],
      group: cd.group || "A", material: matName(cd.material), busbar: cd.busbar || ("bb" + uid) };
    uid++; conductors.push(c); makeNode(c);
  }
  layer.batchDraw();
}

// ---- toolbar / solve ------------------------------------------------------
$("addBar").onclick = () => addConductor("rect", "A");
$("addRound").onclick = () => addConductor("circle", "A");
$("del").onclick = () => { if (selected) { selected.node.destroy(); conductors = conductors.filter((c) => c !== selected); select(null); layer.batchDraw(); } };

// ---- copy / paste geometry ------------------------------------------------
let clipboard = [];
function copySelection() {
  let src;
  if (editBusbar) src = conductors.filter((c) => c.busbar === editBusbar);  // whole busbar
  else if (selected) src = conductors.filter((c) => c.busbar === selected.busbar);
  else return;
  clipboard = src.map((c) => ({ ...c, node: undefined }));  // snapshot, drop node ref
  document.getElementById("isoBadge").textContent = `copied ${clipboard.length} shape(s)`;
}
function pasteClipboard() {
  if (!clipboard.length) return;
  const step = 2 * GRID;
  const remap = {};
  let first = null;
  for (const s of clipboard) {
    if (!(s.busbar in remap)) remap[s.busbar] = "bb" + uid++;  // each source busbar -> new id
    const c = { ...s, node: undefined, id: uid, name: "C" + uid,
                busbar: remap[s.busbar], x: s.x + step, y: s.y + step };
    uid++;
    conductors.push(c); makeNode(c);
    first = first || c;
  }
  exitIsolation();
  select(first);
  layer.batchDraw();
}
$("copy").onclick = copySelection;
$("paste").onclick = pasteClipboard;

document.addEventListener("keydown", (e) => {
  if (e.key === "Delete" && selected) $("del").onclick();
  if (e.key === "Escape") { exitIsolation(); select(null); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") { copySelection(); e.preventDefault(); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") { pasteClipboard(); e.preventDefault(); }
});

// ---- field view (large overlay) ------------------------------------------
function showFieldView() {
  document.getElementById("fieldOverlay").style.display = "flex";
  requestAnimationFrame(() => { EM.stop(); EM.drawFrame(0); });
}
function hideFieldView() {
  EM.stop();
  document.getElementById("fieldOverlay").style.display = "none";
}
$("editBtn").onclick = hideFieldView;
$("showField").onclick = showFieldView;

$("save").onclick = () => {
  const blob = new Blob([JSON.stringify(toSceneDict(), null, 2)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "busbar.json"; a.click();
};
$("load").onclick = () => $("loadFile").click();
$("loadFile").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const r = new FileReader(); r.onload = () => loadSceneDict(JSON.parse(r.result)); r.readAsText(f);
};

$("solve").onclick = () => {
  if (!EM.ready || !conductors.length) return;
  $("status").textContent = "Meshing + solving (in-browser)…";
  $("solve").disabled = true;
  setTimeout(() => {
    try {
      const t0 = performance.now();
      const data = EM.solve(toSceneDict());
      const ms = (performance.now() - t0).toFixed(0);
      $("resultsBox").hidden = false;
      fillResults(data);
      showFieldView();   // switch the big canvas area to the field result
      $("status").textContent = `Solved in ${ms} ms (${data.num_nodes} nodes). Total loss ${data.total_loss.toPrecision(4)} W/m.`;
    } catch (err) {
      $("status").textContent = "Solve error: " + err; console.error(err);
    }
    $("solve").disabled = false;
  }, 20);
};

function fillResults(data) {
  const tb = document.querySelector("#results tbody"); tb.innerHTML = "";
  for (const c of data.conductors) {
    const tr2 = document.createElement("tr");
    const share = c.share == null ? "-" : (c.share * 100).toFixed(0) + "%";
    tr2.innerHTML = `<td>${c.group}</td><td>${c.I.toFixed(0)}</td><td>${share}</td><td>${c.loss.toPrecision(3)}</td>`;
    tb.appendChild(tr2);
  }
  $("summary").innerHTML = data.terminals.map((t) =>
    `Term ${t.name}: V̇/L=${t.vgrad.toPrecision(3)} V/m, Z=${t.z_re.toExponential(2)}${t.z_im >= 0 ? "+" : ""}${t.z_im.toExponential(2)}j Ω/m`).join("<br>");
}

function updateInfo() {
  const el = $("fieldInfo");
  if (!el.hidden) el.textContent = (EM.DESCRIPTIONS && EM.DESCRIPTIONS[$("field").value]) || "";
}
$("info").onclick = () => { const el = $("fieldInfo"); el.hidden = !el.hidden; updateInfo(); };
$("field").onchange = () => { updateInfo(); if (EM._data) { EM.stop(); EM.drawFrame(0); } };
$("play").onclick = () => EM.play();
$("staticBtn").onclick = () => { EM.stop(); EM.drawFrame(0); };
$("sumPeriod").onclick = () => EM.sumOverPeriod();

// ---- default scene + boot --------------------------------------------------
function defaultScene() {
  PHASES.forEach((g, i) => {
    const c = { id: uid, name: "Phase " + g, type: "rect", w: 10, h: 50, r: 10,
      x: (i - 1) * 30, y: 0, rot: 0, group: g, material: "Copper", busbar: "bb" + uid };
    uid++; conductors.push(c); makeNode(c);
  });
  layer.batchDraw();
}

window.addEventListener("resize", () => {
  recenter();
  if (EM._data && document.getElementById("fieldOverlay").style.display === "flex" && !EM._anim)
    EM.drawFrame(0);
});
recenter();
defaultScene();
document.addEventListener("em-ready", () => {
  document.getElementById("solve").disabled = false;
  document.getElementById("boot").textContent = "ready — edit, then click Solve";
  if (location.search.includes("autosolve")) {
    const m = location.search.match(/field=(\w+)/);
    if (m) document.getElementById("field").value = m[1];
    document.getElementById("solve").click();
  }
});
