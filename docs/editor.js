// editor.js — interactive Konva geometry editor wired to the Pyodide engine.

const PPM = 4;          // pixels per millimetre at scale 1
const GRID = 5;         // grid / snap step in mm
const PHASES = ["A", "B", "C"];
const PHASE_COLOR = { A: "#d65f5f", B: "#5f9ed6", C: "#5fd68a" };
const PASSIVE_COLOR = "#9aa0a6";   // passive bonded (earthed, V̇/L=0)
const FLOAT_COLOR = "#b39ddb";     // passive floating (isolated, net current=0)
const colorFor = (g) => (g == null ? PASSIVE_COLOR : (PHASE_COLOR[g] || "#888"));
const nodeColor = (c) => (c.group == null ? (c.floating ? FLOAT_COLOR : PASSIVE_COLOR)
                                          : (PHASE_COLOR[c.group] || "#888"));
const MAT = {
  Copper: { name: "copper", sigma: 5.8e7, mu_r: 1.0 },
  Aluminium: { name: "aluminium", sigma: 3.5e7, mu_r: 1.0 },
  Steel: { name: "steel", sigma: 1.0e7, mu_r: 200.0 },
  Stainless: { name: "stainless", sigma: 1.4e6, mu_r: 1.0 },
};
const ROT_SNAPS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                   -15, -30, -45, -60, -75, -90, -105, -120, -135, -150, -165];
let snapOn = true;                                       // grid + angle snap for individual edits
const snap = (v) => (snapOn ? Math.round(v / GRID) * GRID : v);

let conductors = [];    // model: list of {id,name,type,w,h,r,x,y,rot,group,material,busbar,node}
let selected = null;     // representative conductor (drives the property panel)
let selGroup = [];       // all selected conductors (whole busbar, or one shape in isolation)
let dragStart = null;    // group-drag bookkeeping
let uid = 1;
let editBusbar = null;  // busbar id under isolation editing, or null

// ---- Konva stage ----------------------------------------------------------
const editorDiv = document.getElementById("editor");
const stage = new Konva.Stage({ container: "editor", width: editorDiv.clientWidth, height: editorDiv.clientHeight });
const gridLayer = new Konva.Layer({ listening: false });
const layer = new Konva.Layer();
stage.add(gridLayer); stage.add(layer);
const tr = new Konva.Transformer({ rotationSnaps: ROT_SNAPS, rotateAnchorOffset: 24 });
layer.add(tr);

const MIN_SCALE = PPM / 4, MAX_SCALE = PPM * 12;   // scroll-zoom limits
function recenter() {
  stage.width(editorDiv.clientWidth); stage.height(editorDiv.clientHeight);
  stage.scale({ x: PPM, y: PPM });
  stage.position({ x: stage.width() / 2, y: stage.height() / 2 });
  drawGrid();
}
// grid covering the current viewport (in world mm), with constant on-screen line
// width and a step that coarsens when zoomed far out so the line count stays bounded.
function drawGrid() {
  gridLayer.destroyChildren();
  const s = stage.scaleX();
  const tlx = -stage.x() / s, tly = -stage.y() / s;
  const brx = (stage.width() - stage.x()) / s, bry = (stage.height() - stage.y()) / s;
  const mx = (brx - tlx) * 0.5, my = (bry - tly) * 0.5;   // margin so small pans stay covered
  let step = GRID;
  while ((brx - tlx + 2 * mx) / step > 300) step *= 2;
  const x0 = Math.floor((tlx - mx) / step) * step, x1 = Math.ceil((brx + mx) / step) * step;
  const y0 = Math.floor((tly - my) / step) * step, y1 = Math.ceil((bry + my) / step) * step;
  for (let x = x0; x <= x1; x += step)
    gridLayer.add(new Konva.Line({ points: [x, y0, x, y1],
      stroke: x === 0 ? "#9aa0a6" : "#e3e6ea", strokeWidth: (x % 25 === 0 ? 0.4 : 0.2) / s }));
  for (let y = y0; y <= y1; y += step)
    gridLayer.add(new Konva.Line({ points: [x0, y, x1, y],
      stroke: y === 0 ? "#9aa0a6" : "#e3e6ea", strokeWidth: (y % 25 === 0 ? 0.4 : 0.2) / s }));
  gridLayer.batchDraw();
}
// scroll wheel = zoom toward the cursor
stage.on("wheel", (e) => {
  e.evt.preventDefault();
  const oldScale = stage.scaleX(), pointer = stage.getPointerPosition();
  if (!pointer) return;
  const wx = (pointer.x - stage.x()) / oldScale, wy = (pointer.y - stage.y()) / oldScale;
  let s = e.evt.deltaY > 0 ? oldScale / 1.12 : oldScale * 1.12;
  s = Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));
  stage.scale({ x: s, y: s });
  stage.position({ x: pointer.x - wx * s, y: pointer.y - wy * s });
  drawGrid();
});
// drag empty canvas = pan (shapes/handles still drag themselves)
stage.draggable(true);
stage.on("dragmove dragend", (e) => { if (e.target === stage) drawGrid(); });

// ---- model <-> Konva ------------------------------------------------------
function makeNode(c) {
  const common = { x: c.x, y: c.y, rotation: c.rot, draggable: true,
                   fill: nodeColor(c), stroke: "#222", strokeWidth: 0.4 };
  let node;
  if (c.type === "rect")
    node = new Konva.Rect({ ...common, width: c.w, height: c.h, offsetX: c.w / 2, offsetY: c.h / 2 });
  else
    node = new Konva.Circle({ ...common, radius: c.r });
  node.on("click tap", () => select(c));
  node.on("dblclick dbltap", () => enterIsolation(c.busbar));
  node.on("dragstart", () => {
    if (!selGroup.includes(c)) select(c);   // dragging selects the busbar first
    dragStart = (selGroup.length > 1 && selGroup.includes(c))
      ? { rx: node.x(), ry: node.y(), members: selGroup.map((m) => ({ m, x: m.node.x(), y: m.node.y() })) }
      : null;
  });
  node.on("dragmove", () => {
    if (dragStart) {   // move the whole busbar together — rigid, never snapped
      const dx = node.x() - dragStart.rx, dy = node.y() - dragStart.ry;
      for (const it of dragStart.members) {
        if (it.m === c) continue;
        it.m.node.x(it.x + dx); it.m.node.y(it.y + dy);
      }
      tr.forceUpdate();   // keep the selection box on the moving group
    } else {             // single shape — snap to grid (when enabled)
      node.x(snap(node.x())); node.y(snap(node.y()));
    }
  });
  node.on("dragend", () => {
    if (dragStart) { dragStart.members.forEach((it) => syncFromNode(it.m, false)); dragStart = null; }
    else syncFromNode(c, true);
    if (selected === c) fillProps(c);
    tr.forceUpdate(); updateAreas();
  });
  // rotate/scale via the transformer handles — bake the accumulated scale &
  // rotation into the model once the gesture ends (works for a whole group).
  node.on("transformend", () => {
    const grp = selGroup.length > 1 && selGroup.includes(c) ? selGroup : [c];
    grp.forEach((m) => bakeNode(m, selGroup.length <= 1));
    tr.forceUpdate();
    if (selected) fillProps(selected);
    updateAreas();
  });
  c.node = node;
  layer.add(node);
  return node;
}
// commit a node's transform (scale folded into size, rotation, position) into
// the model. doSnap: snap size+centre to the grid — true only for single-shape
// edits; a rotated/scaled group is committed exactly so it stays rigid.
function bakeNode(c, doSnap) {
  const n = c.node, sp = (v) => (doSnap ? snap(v) : v);
  const sx = Math.abs(n.scaleX()), sy = Math.abs(n.scaleY());
  if (c.type === "rect") {
    const w = Math.max(GRID, sp(n.width() * sx)), h = Math.max(GRID, sp(n.height() * sy));
    n.scaleX(1); n.scaleY(1); c.w = w; c.h = h;
    n.width(w); n.height(h); n.offset({ x: w / 2, y: h / 2 });
  } else {
    const r = Math.max(GRID, sp(n.radius() * sx));
    n.scaleX(1); n.scaleY(1); c.r = r; n.radius(r);
  }
  c.rot = Math.round(n.rotation());
  c.x = sp(n.x()); c.y = sp(n.y());
  n.x(c.x); n.y(c.y);
}
function syncFromNode(c, doSnap = true) {   // commit Konva node back to the model
  const n = c.node, sp = (v) => (doSnap ? snap(v) : v);
  c.x = sp(n.x()); c.y = sp(n.y()); c.rot = Math.round(n.rotation());
  n.x(c.x); n.y(c.y);
  if (c.type === "rect") {
    c.w = sp(n.width()); c.h = sp(n.height());
    n.width(c.w); n.height(c.h); n.offset({ x: c.w / 2, y: c.h / 2 });
  } else {
    c.r = sp(n.radius()); n.radius(c.r);
  }
  layer.batchDraw();
}

function phaseOfBusbar(bb) {
  const m = conductors.find((c) => c.busbar === bb);
  return m ? m.group : "A";
}

// live per-phase total cross-section area (mm^2), shown in geometry mode
function shapeAreaMM2(c) {
  if (c.type === "rect") return c.w * c.h;
  if (c.type === "circle") return Math.PI * c.r * c.r;
  return 0;
}
function updateAreas() {
  const a = {};
  for (const c of conductors) {
    if (c.group == null) continue;
    a[c.group] = (a[c.group] || 0) + shapeAreaMM2(c);
  }
  const parts = Object.keys(a).sort().map((g) => `${g}: ${a[g].toFixed(0)}`);
  $("areas").innerHTML = parts.length ? "Area/phase (mm²) — " + parts.join(", ") : "";
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
  updateAreas();
}

// ---- draw enclosure walls (passive sheet-metal polylines) -----------------
let drawMode = false, wallPts = [], previewLine = null, rubber = null;

function worldPt() {
  const p = stage.getPointerPosition(); if (!p) return null;
  const s = stage.scaleX();
  return { x: snap((p.x - stage.x()) / s), y: snap((p.y - stage.y()) / s) };
}
// one thin rectangle conductor spanning p0->p1 (a wall segment)
function wallRect(p0, p1, t, busbar, material) {
  const dx = p1.x - p0.x, dy = p1.y - p0.y, len = Math.hypot(dx, dy) || GRID;
  return { id: uid, name: "W" + uid, type: "rect", w: len, h: t, r: 0,
           x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2,
           rot: (Math.atan2(dy, dx) * 180) / Math.PI,
           group: null, material, busbar };
}
function startWallMode() {
  if (drawMode) { exitWallMode(); return; }
  exitIsolation(); select(null);
  drawMode = true; wallPts = [];
  stage.draggable(false);
  conductors.forEach((c) => c.node.listening(false));   // clicks fall through to the stage
  const sw = 1.5 / stage.scaleX();
  previewLine = new Konva.Line({ points: [], stroke: "#1f6feb", strokeWidth: sw, listening: false });
  rubber = new Konva.Line({ points: [], stroke: "#1f6feb", strokeWidth: sw, dash: [3, 3], listening: false });
  layer.add(previewLine); layer.add(rubber);
  $("wallBtn").style.background = "#1f6feb"; $("wallBtn").style.color = "#fff";
  $("isoBadge").textContent = "drawing wall — click to add points, double-click/Enter to finish, C to close, Esc cancels";
}
function exitWallMode() {
  drawMode = false; wallPts = [];
  if (previewLine) { previewLine.destroy(); previewLine = null; }
  if (rubber) { rubber.destroy(); rubber = null; }
  stage.draggable(true);
  conductors.forEach((c) => c.node.listening(true));
  $("wallBtn").style.background = ""; $("wallBtn").style.color = "";
  $("isoBadge").textContent = "";
  layer.batchDraw();
}
function wallVertex() {
  const w = worldPt(); if (!w) return;
  wallPts.push(w);
  previewLine.points(wallPts.flatMap((p) => [p.x, p.y]));
  layer.batchDraw();
}
function updateRubber() {
  if (!drawMode || !wallPts.length) return;
  const w = worldPt(); if (!w) return;
  const last = wallPts[wallPts.length - 1];
  rubber.points([last.x, last.y, w.x, w.y]);
  layer.batchDraw();
}
function finishWall(close) {
  // dedupe consecutive coincident points (a finishing double-click adds one)
  const pts = [];
  for (const p of wallPts)
    if (!pts.length || pts[pts.length - 1].x !== p.x || pts[pts.length - 1].y !== p.y) pts.push(p);
  exitWallMode();
  if (pts.length < 2) return;
  const t = Math.max(GRID / 5, +$("wallT").value || 3), material = $("wallMat").value;
  const bb = "bb" + uid++;
  const segs = [];
  for (let i = 0; i < pts.length - 1; i++) segs.push([pts[i], pts[i + 1]]);
  if (close && pts.length >= 3) segs.push([pts[pts.length - 1], pts[0]]);
  let first = null;
  for (const [a, b] of segs) {
    const c = wallRect(a, b, t, bb, material); uid++;
    conductors.push(c); makeNode(c); first = first || c;
  }
  layer.batchDraw(); updateAreas();
  if (first) select(first);   // select the whole wall busbar so it's movable
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

function setHighlight(c, on) {
  c.node.stroke(on ? "#1f6feb" : "#222");
  c.node.strokeWidth(on ? 1.5 : 0.4);
}
function select(c) {
  selGroup.forEach((m) => setHighlight(m, false));   // clear previous
  if (!c) { selected = null; selGroup = []; tr.nodes([]); $("propPanel").hidden = true; layer.batchDraw(); return; }
  selected = c;
  // single click selects the whole busbar (so it moves together); inside
  // isolation only the clicked shape is selected (edit it individually).
  selGroup = editBusbar ? [c] : conductors.filter((k) => k.busbar === c.busbar);
  selGroup.forEach((m) => setHighlight(m, true));
  // one transformer over the whole selection: drag the rotate "hair" to spin the
  // group, the corner/edge anchors to scale it (all about the group centre).
  const hasCircle = selGroup.some((m) => m.type === "circle");
  tr.nodes(selGroup.map((m) => m.node));
  // 15° angle snap only when manipulating a single shape with snap enabled;
  // a group rotates freely so you can dial in any angle.
  tr.rotationSnaps(snapOn && selGroup.length === 1 ? ROT_SNAPS : []);
  tr.keepRatio(hasCircle);   // keep circles round (and multi-shape aspect locked)
  tr.enabledAnchors(hasCircle
    ? ["top-left", "top-right", "bottom-left", "bottom-right"]
    : ["top-left", "top-center", "top-right", "middle-left",
       "middle-right", "bottom-left", "bottom-center", "bottom-right"]);
  $("propPanel").hidden = false;
  fillProps(c);
  layer.batchDraw();
}

// rotate the selected busbar (or shape) about its area-weighted centroid
function rotateGroup(deg) {
  if (!selGroup.length) return;
  let sx = 0, sy = 0, sa = 0;
  for (const m of selGroup) { const a = shapeAreaMM2(m) || 1; sx += a * m.x; sy += a * m.y; sa += a; }
  const cx = sx / sa, cy = sy / sa, th = (deg * Math.PI) / 180, cs = Math.cos(th), sn = Math.sin(th);
  for (const m of selGroup) {
    const dx = m.x - cx, dy = m.y - cy;
    m.x = cx + dx * cs - dy * sn;          // orbit each piece about the centroid
    m.y = cy + dx * sn + dy * cs;
    m.rot = (((m.rot + deg) % 360) + 540) % 360 - 180;   // and spin it, kept in [-180,180]
    const n = m.node; n.x(m.x); n.y(m.y); n.rotation(m.rot);
  }
  if (tr.nodes().length) tr.forceUpdate();
  if (selected) fillProps(selected);
  layer.batchDraw();
  updateAreas();
}

stage.on("click tap", (e) => {
  if (drawMode) { wallVertex(); return; }
  if (e.target === stage) select(null);
});
stage.on("dblclick dbltap", (e) => {
  if (drawMode) { finishWall(false); return; }
  if (e.target === stage) exitIsolation();
});
stage.on("mousemove", () => { if (drawMode) updateRubber(); });

// ---- properties panel -----------------------------------------------------
const $ = (id) => document.getElementById(id);
function fillProps(c) {
  const single = selGroup.length <= 1;   // geometry is per-shape; edit multi via isolation
  $("pW").value = c.type === "rect" ? c.w : c.r;
  $("pH").value = c.h;
  $("pHlabel").parentElement.style.display = c.type === "rect" ? "" : "none";
  $("pX").value = c.x; $("pY").value = c.y; $("pRot").value = c.rot;
  $("pPhase").value = c.group == null ? (c.floating ? "F" : "P") : c.group; $("pMat").value = c.material;
  // size/position are per-shape (edit a group's pieces via isolation); rotation
  // works on a group too — it spins the whole busbar to the typed angle.
  for (const id of ["pW", "pH", "pX", "pY"]) $(id).disabled = !single;
  $("pRot").disabled = false;
}
// doGeom: only re-apply geometry when a geometry field changed — editing phase/
// material must NOT re-snap a sub-grid wall thickness (snap(3mm)->5mm) etc.
function readProps(doGeom) {
  const c = selected; if (!c) return;
  const single = selGroup.length <= 1;
  if (doGeom) {
    if (single) {   // geometry applies to the one shape
      if (c.type === "rect") { c.w = snap(+$("pW").value); c.h = snap(+$("pH").value); }
      else c.r = snap(+$("pW").value);
      c.x = snap(+$("pX").value); c.y = snap(+$("pY").value); c.rot = +$("pRot").value;
      const n = c.node;
      n.x(c.x); n.y(c.y); n.rotation(c.rot);
      if (c.type === "rect") { n.width(c.w); n.height(c.h); n.offset({ x: c.w / 2, y: c.h / 2 }); }
      else n.radius(c.r);
    } else {        // group: rotate the whole busbar to the typed angle (about its centroid)
      const delta = (+$("pRot").value) - c.rot;
      if (delta) rotateGroup(delta);
    }
  }
  // phase + material apply to the whole busbar. "P"/"F" = passive (group null);
  // "F" additionally floats (net current forced to 0); "P" is bonded (V̇/L=0).
  const pv = $("pPhase").value;
  const phase = (pv === "P" || pv === "F") ? null : pv;
  const floating = pv === "F";
  const material = $("pMat").value;
  for (const m of selGroup) {
    m.group = phase; m.material = material; m.floating = floating;
    m.node.fill(nodeColor(m));
  }
  layer.batchDraw();
  updateAreas();
}
["pW", "pH", "pX", "pY", "pRot"].forEach((id) =>
  $(id).addEventListener("input", () => readProps(true)));
["pPhase", "pMat"].forEach((id) =>
  $(id).addEventListener("input", () => readProps(false)));

$("snapToggle").addEventListener("change", (e) => {
  snapOn = e.target.checked;
  if (selected) select(selected);   // re-apply the angle-snap configuration
});

$("meshScale").addEventListener("input", (e) => {
  $("meshScaleVal").textContent = (+e.target.value).toFixed(1) + "×";
});

// ---- scene serialisation (io format, metres) ------------------------------
function toSceneDict() {
  return {
    format: 1, frequency: +$("freq").value, three_phase: $("threephase").checked,
    line_current: +$("current").value, boundary: "dirichlet", order: 1,
    domain_radius: 0, lc_surface: 0, lc_far: 0, group_currents: {},
    mesh_scale: +$("meshScale").value,
    conductors: conductors.map((c) => ({
      name: c.name,
      shape: c.type === "rect"
        ? { type: "rect", width: c.w / 1000, height: c.h / 1000 }
        : { type: "circle", radius: c.r / 1000 },
      placement: [c.x / 1000, c.y / 1000, c.rot],
      material: MAT[c.material], group: c.group, busbar: c.busbar,
      floating: !!c.floating,
    })),
  };
}
function loadSceneDict(d) {
  conductors.forEach((c) => c.node.destroy());
  conductors = []; selected = null; selGroup = []; tr.nodes([]);
  uid = 1;
  $("freq").value = d.frequency; $("threephase").checked = d.three_phase;
  $("current").value = d.line_current;
  const matName = (m) => ({ copper: "Copper", aluminium: "Aluminium", steel: "Steel",
                            stainless: "Stainless" }[m.name] || "Copper");
  for (const cd of d.conductors) {
    const s = cd.shape;
    const c = { id: uid, name: cd.name, type: s.type === "circle" ? "circle" : "rect",
      w: (s.width || 0) * 1000, h: (s.height || 0) * 1000, r: (s.radius || 0) * 1000,
      x: cd.placement[0] * 1000, y: cd.placement[1] * 1000, rot: cd.placement[2],
      group: cd.group ?? null, floating: !!cd.floating,
      material: matName(cd.material), busbar: cd.busbar || ("bb" + uid) };
    uid++; conductors.push(c); makeNode(c);
  }
  layer.batchDraw();
  updateAreas();
}

// ---- toolbar / solve ------------------------------------------------------
$("addBar").onclick = () => addConductor("rect", "A");
$("addRound").onclick = () => addConductor("circle", "A");
$("wallBtn").onclick = startWallMode;
$("del").onclick = () => { if (selected) { selected.node.destroy(); conductors = conductors.filter((c) => c !== selected); select(null); layer.batchDraw(); updateAreas(); } };

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
  updateAreas();
}
$("copy").onclick = copySelection;
$("paste").onclick = pasteClipboard;

document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName);
  if (drawMode && !typing) {   // wall-drawing shortcuts take precedence
    if (e.key === "Escape") { exitWallMode(); e.preventDefault(); return; }
    if (e.key === "Enter") { finishWall(false); e.preventDefault(); return; }
    if (e.key.toLowerCase() === "c") { finishWall(true); e.preventDefault(); return; }
  }
  if (e.key === "Delete" && selected) $("del").onclick();
  if (e.key === "Escape") { exitIsolation(); select(null); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") { copySelection(); e.preventDefault(); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") { pasteClipboard(); e.preventDefault(); }
  if (!typing && !e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "r" && selGroup.length) {
    rotateGroup(e.shiftKey ? -15 : 15); e.preventDefault();   // quick 15° nudge
  }
});

// ---- field view (large overlay) ------------------------------------------
function showFieldView() {
  document.getElementById("fieldOverlay").style.display = "flex";
  requestAnimationFrame(() => renderField());
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

// progress bar shown while booting and while a (background) solve runs
EM._onProgress = (text) => { $("status").textContent = text; };
$("solve").onclick = async () => {
  if (!EM.ready || !conductors.length) return;
  $("status").textContent = "Meshing + solving (in-browser)…";
  $("progress").hidden = false;        // the editor stays interactive meanwhile
  try {
    const t0 = performance.now();
    const data = await EM.solve(toSceneDict());   // runs in the worker, non-blocking
    const ms = (performance.now() - t0).toFixed(0);
    $("resultsBox").hidden = false;
    fillResults(data);
    showFieldView();   // switch the big canvas area to the field result
    $("status").textContent = `Solved in ${ms} ms (${data.num_nodes} nodes). Total loss ${data.total_loss.toPrecision(4)} W/m.`;
  } catch (err) {
    $("status").textContent = "Solve error: " + err; console.error(err);
  }
  $("progress").hidden = true;
};

function fillResults(data) {
  const tb = document.querySelector("#results tbody"); tb.innerHTML = "";
  for (const c of data.conductors) {
    const tr2 = document.createElement("tr");
    const share = c.share == null ? "-" : (c.share * 100).toFixed(0) + "%";
    const grp = c.group == null ? "encl" : (c.group === "float" ? "float" : c.group);   // passive
    tr2.innerHTML = `<td>${grp}</td><td>${c.I.toFixed(0)}</td><td>${share}</td><td>${c.loss.toPrecision(3)}</td>`;
    tb.appendChild(tr2);
  }
  const ag = data.area_group || {}, eff = data.eff_area_90 || {};
  const effLine = Object.keys(ag).sort().map((g) => {
    const a = ag[g], e = eff[g] || 0, pct = a > 0 ? (100 * e / a).toFixed(0) : "0";
    return `${g} ${e.toFixed(0)}/${a.toFixed(0)} (${pct}%)`;
  }).join(" &nbsp; ");
  const head =
    `<b>Total loss: ${data.total_loss.toPrecision(4)} W/m</b><br>` +
    `${data.loss_per_density.toPrecision(3)} W/m per A/mm²` +
    ` &nbsp;|&nbsp; ${data.loss_coeff.toPrecision(3)} W/m per (A/mm²)² (current-indep.)<br>` +
    `(applied ${data.applied_density.toPrecision(3)} A/mm²)<br>` +
    `<span title="Conductor area carrying ≥90% of the phase average |J|. Low % = current crowded into part of the bar.">` +
    `Effective area ≥90% util (mm²): ${effLine}</span><br>`;
  $("summary").innerHTML = head + data.terminals.map((t) =>
    `Term ${t.name}: V̇/L=${t.vgrad.toPrecision(3)} V/m, Z=${t.z_re.toExponential(2)}${t.z_im >= 0 ? "+" : ""}${t.z_im.toExponential(2)}j Ω/m`).join("<br>");
}

function updateInfo() {
  const el = $("fieldInfo");
  if (!el.hidden) el.textContent = (EM.DESCRIPTIONS && EM.DESCRIPTIONS[$("field").value]) || "";
}
$("info").onclick = () => { const el = $("fieldInfo"); el.hidden = !el.hidden; updateInfo(); };
// render the selected field: the "util" entry runs the period-sum buildup,
// everything else draws a single (static) frame.
function renderField() {
  if (!EM._data) return;
  EM.stop();
  if ($("field").value === "util") EM.sumOverPeriod();
  else EM.drawFrame(0);
}
$("field").onchange = () => { updateInfo(); renderField(); };
$("play").onclick = () => { if ($("field").value === "util") EM.sumOverPeriod(); else EM.play(); };
$("staticBtn").onclick = () => {
  EM.stop();
  if ($("field").value === "util" && !EM._util) EM.sumOverPeriod();  // need the map first
  else EM.drawFrame(0);
};

// ---- default scene + boot --------------------------------------------------
function defaultScene() {
  PHASES.forEach((g, i) => {
    const c = { id: uid, name: "Phase " + g, type: "rect", w: 10, h: 50, r: 10,
      x: (i - 1) * 30, y: 0, rot: 0, group: g, material: "Copper", busbar: "bb" + uid };
    uid++; conductors.push(c); makeNode(c);
  });
  layer.batchDraw();
  updateAreas();
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
  document.getElementById("progress").hidden = true;   // boot finished
  document.getElementById("boot").textContent = "ready — edit, then click Solve";
  if (location.search.includes("autosolve")) {
    const m = location.search.match(/field=(\w+)/);
    if (m) document.getElementById("field").value = m[1];
    document.getElementById("solve").click();
  }
});
