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
// fill + "off" styling: Air items render faint & dashed and are excluded from solve
function styleNode(c) {
  const air = c.material === "Air";
  c.node.fill(nodeColor(c));
  c.node.opacity(air ? 0.3 : 1);
  c.node.dash(air ? [4, 4] : []);
}
const MAT = {
  Copper: { name: "copper", sigma: 5.8e7, mu_r: 1.0 },
  Aluminium: { name: "aluminium", sigma: 3.5e7, mu_r: 1.0 },
  Steel: { name: "steel", sigma: 1.0e7, mu_r: 200.0 },
  Stainless: { name: "stainless", sigma: 1.4e6, mu_r: 1.0 },
  Air: { name: "air", sigma: 0.0, mu_r: 1.0 },   // "off" — excluded from the solve
};
const ROT_SNAPS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                   -15, -30, -45, -60, -75, -90, -105, -120, -135, -150, -165];
let snapOn = true;                                       // grid + angle snap for individual edits
const snap = (v) => (snapOn ? Math.round(v / GRID) * GRID : v);

let conductors = [];    // model: list of {id,name,type,w,h,r,x,y,rot,group,material,busbar,node}
let selected = null;     // representative conductor (drives the property panel)
let selGroup = [];       // all selected conductors (whole busbar, or one shape in isolation)
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

// ---- busbar coordinate systems --------------------------------------------
// Each busbar is a Konva.Group carrying that busbar's frame (origin x,y + angle,
// in world mm; the layer itself is un-transformed). Member shapes live INSIDE
// the group in busbar-local coordinates and are never recomputed when the busbar
// moves or rotates -- only the group's transform changes -- so a group stays
// rigid by construction (no relative drift). World coords are derived (group ∘
// local) only when needed: rendering is Konva's job, export bakes in toSceneDict.
const busbarGroups = new Map();   // busbar id -> Konva.Group

function busbarGroup(bb) {
  let g = busbarGroups.get(bb);
  if (g) return g;
  g = new Konva.Group({ x: 0, y: 0, rotation: 0, draggable: true });
  g.setAttr("busbar", bb);
  busbarGroups.set(bb, g);
  layer.add(g);
  bindGroupDrag(g);
  tr.moveToTop();
  return g;
}
function membersOf(bb) { return conductors.filter((c) => c.busbar === bb); }
function dropBusbarIfEmpty(bb) {
  const g = busbarGroups.get(bb);
  if (g && !membersOf(bb).length) { g.destroy(); busbarGroups.delete(bb); }
}

// local (group frame) <-> world mm, with group scale kept at 1 (always baked).
function localToWorld(g, lx, ly) {
  const a = (g.rotation() * Math.PI) / 180, cs = Math.cos(a), sn = Math.sin(a);
  return { x: g.x() + lx * cs - ly * sn, y: g.y() + lx * sn + ly * cs };
}
function worldToLocal(g, wx, wy) {
  const a = (-g.rotation() * Math.PI) / 180, cs = Math.cos(a), sn = Math.sin(a);
  const dx = wx - g.x(), dy = wy - g.y();
  return { x: dx * cs - dy * sn, y: dx * sn + dy * cs };
}
// area-weighted centroid of a busbar's members, in world mm
function groupCentroid(g, members) {
  let sx = 0, sy = 0, sa = 0;
  for (const m of members) {
    const a = shapeAreaMM2(m) || 1, w = localToWorld(g, m.x, m.y);
    sx += a * w.x; sy += a * w.y; sa += a;
  }
  return sa ? { x: sx / sa, y: sy / sa } : { x: g.x(), y: g.y() };
}
// rotate a whole busbar by deg about a world pivot, touching ONLY the group node
// (members stay put in local coords -> perfectly rigid, no drift).
function rotateGroupNode(g, deg, pivot) {
  const a = (deg * Math.PI) / 180, cs = Math.cos(a), sn = Math.sin(a);
  const dx = g.x() - pivot.x, dy = g.y() - pivot.y;
  g.x(pivot.x + dx * cs - dy * sn);
  g.y(pivot.y + dx * sn + dy * cs);
  g.rotation(g.rotation() + deg);
}

// whole-busbar drag (the group is draggable; members are not, so grabbing any
// member drags the group) + scale bake on transform end.
function bindGroupDrag(g) {
  let start = null;
  g.on("dragstart", () => {
    const m = membersOf(g.getAttr("busbar"))[0];
    if (m && !selGroup.includes(m)) select(m);   // grabbing a busbar selects it
    start = { x: g.x(), y: g.y() };
  });
  g.on("dragmove", () => {
    if (start && snapOn) {   // snap the shared delta so the group lands on-grid, rigidly
      g.x(start.x + snap(g.x() - start.x));
      g.y(start.y + snap(g.y() - start.y));
    }
    tr.forceUpdate();
  });
  g.on("dragend", () => { start = null; updateAreas(); });
  g.on("transformend", () => { bakeGroupScale(g); tr.forceUpdate(); if (selected) fillProps(selected); updateAreas(); });
}
// fold a group-node scale into each member's local size + position, then reset
// the group scale to 1 (rotation/position stay on the group node). Folding S into
// the children is exact for positions; sizes use the axis-aligned approximation
// (matches the prior per-shape bake) when a member is itself rotated.
function bakeGroupScale(g) {
  const sx = Math.abs(g.scaleX()), sy = Math.abs(g.scaleY());
  if (Math.abs(sx - 1) < 1e-9 && Math.abs(sy - 1) < 1e-9) return;
  for (const m of membersOf(g.getAttr("busbar"))) {
    m.x *= sx; m.y *= sy;
    if (m.type === "rect") { m.w = Math.max(GRID, m.w * sx); m.h = Math.max(GRID, m.h * sy); }
    else m.r = Math.max(GRID, m.r * (sx + sy) / 2);
    const n = m.node;
    n.x(m.x); n.y(m.y);
    if (m.type === "rect") { n.width(m.w); n.height(m.h); n.offset({ x: m.w / 2, y: m.h / 2 }); }
    else n.radius(m.r);
  }
  g.scale({ x: 1, y: 1 });
}

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
// c.x / c.y / c.rot are busbar-LOCAL (inside c's Konva group). Whole-busbar
// moves/rotations live on the group node (see bindGroupDrag); the member node is
// draggable only inside isolation, where it edits its own local placement.
function makeNode(c) {
  const common = { x: c.x, y: c.y, rotation: c.rot, draggable: false,
                   fill: nodeColor(c), stroke: "#222", strokeWidth: 0.4 };
  let node;
  if (c.type === "rect")
    node = new Konva.Rect({ ...common, width: c.w, height: c.h, offsetX: c.w / 2, offsetY: c.h / 2 });
  else
    node = new Konva.Circle({ ...common, radius: c.r });
  node.on("click tap", () => select(c));
  node.on("dblclick dbltap", () => enterIsolation(c.busbar));
  // member-level drag/transform: only reachable in isolation (single shape).
  node.on("dragmove", () => { node.x(snap(node.x())); node.y(snap(node.y())); });
  node.on("dragend", () => { syncFromNode(c, true); if (selected === c) fillProps(c); tr.forceUpdate(); updateAreas(); });
  node.on("transformend", () => { bakeNode(c, true); tr.forceUpdate(); if (selected) fillProps(selected); updateAreas(); });
  c.node = node;
  busbarGroup(c.busbar).add(node);   // into the busbar's coordinate system
  tr.moveToTop();
  styleNode(c);   // apply Air faint/dashed styling if needed
  return node;
}
// commit a member node's transform (scale folded into size, rotation, position)
// into the model, in busbar-LOCAL coords. Used for single-shape edits in
// isolation; doSnap snaps size + centre + angle to the grid.
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
  c.rot = doSnap ? Math.round(n.rotation()) : n.rotation();
  c.x = sp(n.x()); c.y = sp(n.y());
  n.x(c.x); n.y(c.y);
}
function syncFromNode(c, doSnap = true) {   // commit Konva node back to the model
  const n = c.node, sp = (v) => (doSnap ? snap(v) : v);
  c.x = sp(n.x()); c.y = sp(n.y());
  c.rot = doSnap ? Math.round(n.rotation()) : n.rotation();   // group moves keep exact angle
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
    if (c.group == null || c.material === "Air") continue;   // skip passive & off items
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
  const t = Math.max(GRID / 5, +$("wallT").value || 3), material = "Steel";  // retag in the panel
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
  for (const g of busbarGroups.values()) g.draggable(true);   // whole-busbar drag again
  conductors.forEach((c) => { c.node.draggable(false); c.node.listening(true); styleNode(c); });
  document.getElementById("isoBadge").textContent = "";
  layer.batchDraw();
}
function applyIsolation() {
  for (const g of busbarGroups.values()) g.draggable(false);  // edit members, not the frame
  conductors.forEach((c) => {
    const member = c.busbar === editBusbar;
    styleNode(c);                                   // base fill/opacity (Air stays faint)
    if (!member) c.node.opacity(0.22);              // fade the rest
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
  // normal click selects the whole busbar (the transformer drives its group node,
  // so move/rotate are rigid); inside isolation it selects just the clicked shape.
  const iso = !!editBusbar;
  selGroup = iso ? [c] : membersOf(c.busbar);
  selGroup.forEach((m) => setHighlight(m, true));
  const hasCircle = selGroup.some((m) => m.type === "circle");
  // isolation drives the member node; otherwise the busbar's group node (one
  // rigid transform for the whole bar — rotate hair to spin, anchors to scale).
  tr.nodes([iso ? c.node : busbarGroup(c.busbar)]);
  // 15° angle snap only for a single shape in isolation; a busbar rotates freely.
  tr.rotationSnaps(snapOn && iso ? ROT_SNAPS : []);
  tr.keepRatio(hasCircle);   // keep circles round (and mixed-circle aspect locked)
  tr.enabledAnchors(hasCircle
    ? ["top-left", "top-right", "bottom-left", "bottom-right"]
    : ["top-left", "top-center", "top-right", "middle-left",
       "middle-right", "bottom-left", "bottom-center", "bottom-right"]);
  tr.moveToTop();
  $("propPanel").hidden = false;
  fillProps(c);
  layer.batchDraw();
}

// rotate the selection: a member about its own centre (isolation), or the whole
// busbar about its area-weighted centroid -- the latter only moves the group
// node, so members stay rigid (no relative drift).
function rotateGroup(deg) {
  if (!selGroup.length) return;
  if (editBusbar) {
    const c = selGroup[0];
    c.rot = (((c.rot + deg) % 360) + 540) % 360 - 180;
    c.node.rotation(c.rot);
  } else {
    const g = busbarGroup(selected.busbar);
    rotateGroupNode(g, deg, groupCentroid(g, selGroup));
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
const r2 = (v) => Math.round(v * 100) / 100;   // tidy display of derived floats
function fillProps(c) {
  const iso = !!editBusbar;
  const g = busbarGroup(c.busbar);
  // X/Y/W/H are per-shape: editable in isolation, or for a one-shape busbar.
  const geomEditable = iso || membersOf(c.busbar).length === 1;
  $("pW").value = c.type === "rect" ? c.w : c.r;
  $("pH").value = c.h;
  $("pHlabel").parentElement.style.display = c.type === "rect" ? "" : "none";
  if (iso) {                                   // member-local placement
    $("pX").value = r2(c.x); $("pY").value = r2(c.y); $("pRot").value = r2(c.rot);
  } else {                                      // world placement + busbar angle
    const w = localToWorld(g, c.x, c.y);
    $("pX").value = r2(w.x); $("pY").value = r2(w.y); $("pRot").value = r2(g.rotation());
  }
  $("pPhase").value = c.group == null ? (c.floating ? "F" : "P") : c.group; $("pMat").value = c.material;
  for (const id of ["pW", "pH", "pX", "pY"]) $(id).disabled = !geomEditable;
  $("pRot").disabled = false;
}
function applyShapeSize(c) {
  const n = c.node;
  if (c.type === "rect") { c.w = snap(+$("pW").value); c.h = snap(+$("pH").value); n.width(c.w); n.height(c.h); n.offset({ x: c.w / 2, y: c.h / 2 }); }
  else { c.r = snap(+$("pW").value); n.radius(c.r); }
}
// doGeom: only re-apply geometry when a geometry field changed — editing phase/
// material must NOT re-snap a sub-grid wall thickness (snap(3mm)->5mm) etc.
function readProps(doGeom) {
  const c = selected; if (!c) return;
  const iso = !!editBusbar;
  const g = busbarGroup(c.busbar);
  if (doGeom) {
    if (iso) {   // edit the one member's LOCAL placement
      applyShapeSize(c);
      c.x = snap(+$("pX").value); c.y = snap(+$("pY").value); c.rot = +$("pRot").value;
      c.node.x(c.x); c.node.y(c.y); c.node.rotation(c.rot);
    } else {     // busbar: rotate the group to the typed angle, then (one-shape only)
      const dRot = (+$("pRot").value) - g.rotation();   // size + move via the group frame
      if (dRot) rotateGroupNode(g, dRot, groupCentroid(g, selGroup));
      if (membersOf(c.busbar).length === 1) {
        applyShapeSize(c);
        const want = { x: snap(+$("pX").value), y: snap(+$("pY").value) };
        const cur = localToWorld(g, c.x, c.y);
        g.x(g.x() + (want.x - cur.x)); g.y(g.y() + (want.y - cur.y));   // shift frame so the shape lands at (X,Y)
      }
    }
    if (tr.nodes().length) tr.forceUpdate();
  }
  // phase + material apply to the whole busbar. "P"/"F" = passive (group null);
  // "F" additionally floats (net current forced to 0); "P" is bonded (V̇/L=0).
  const pv = $("pPhase").value;
  const phase = (pv === "P" || pv === "F") ? null : pv;
  const floating = pv === "F";
  const material = $("pMat").value;
  for (const m of selGroup) {
    m.group = phase; m.material = material; m.floating = floating;
    styleNode(m);
  }
  layer.batchDraw();
  updateAreas();
}
["pW", "pH", "pX", "pY", "pRot"].forEach((id) =>
  $(id).addEventListener("input", () => readProps(true)));
["pPhase", "pMat"].forEach((id) =>
  $(id).addEventListener("input", () => readProps(false)));

// toggle the selected busbar on/off by swapping its material to/from Air,
// remembering the previous material so it restores exactly.
function toggleAir() {
  if (!selGroup.length) return;
  const off = selGroup.some((c) => c.material !== "Air");   // any live -> turn all off
  for (const c of selGroup) {
    if (off) { c._prevMat = c.material; c.material = "Air"; }
    else { c.material = c._prevMat || "Copper"; }
    styleNode(c);
  }
  if (selected) fillProps(selected);
  layer.batchDraw(); updateAreas();
}

$("snapToggle").addEventListener("change", (e) => {
  snapOn = e.target.checked;
  if (selected) select(selected);   // re-apply the angle-snap configuration
});

$("meshScale").addEventListener("input", (e) => {
  $("meshScaleVal").textContent = (+e.target.value).toFixed(1) + "×";
});

// ---- scene serialisation (io format, metres) ------------------------------
function toSceneDict(skipAir) {
  const cs = skipAir ? conductors.filter((c) => c.material !== "Air") : conductors;
  return {
    format: 1, frequency: +$("freq").value, three_phase: $("threephase").checked,
    line_current: +$("current").value, boundary: "dirichlet", order: 1,
    domain_radius: 0, lc_surface: 0, lc_far: 0, group_currents: {},
    mesh_scale: +$("meshScale").value,
    conductors: cs.map((c) => {
      const w = localToWorld(busbarGroup(c.busbar), c.x, c.y);   // bake group ∘ local -> world
      return {
        name: c.name,
        shape: c.type === "rect"
          ? { type: "rect", width: c.w / 1000, height: c.h / 1000 }
          : { type: "circle", radius: c.r / 1000 },
        placement: [w.x / 1000, w.y / 1000, busbarGroup(c.busbar).rotation() + c.rot],
        material: MAT[c.material], group: c.group, busbar: c.busbar,
        floating: !!c.floating,
      };
    }),
  };
}
function loadSceneDict(d) {
  conductors.forEach((c) => c.node.destroy());
  for (const g of busbarGroups.values()) g.destroy();
  busbarGroups.clear();
  conductors = []; selected = null; selGroup = []; tr.nodes([]);
  uid = 1;
  $("freq").value = d.frequency; $("threephase").checked = d.three_phase;
  $("current").value = d.line_current;
  const matName = (m) => ({ copper: "Copper", aluminium: "Aluminium", steel: "Steel",
                            stainless: "Stainless", air: "Air" }[m.name] || "Copper");
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
$("del").onclick = () => { if (selected) { const bb = selected.busbar; selected.node.destroy(); conductors = conductors.filter((c) => c !== selected); dropBusbarIfEmpty(bb); select(null); layer.batchDraw(); updateAreas(); } };

// ---- copy / paste geometry ------------------------------------------------
let clipboard = [];          // member snapshots (local coords)
let clipboardGroups = {};    // source busbar id -> {x,y,rot} group frame
function copySelection() {
  const bb = editBusbar || (selected && selected.busbar);
  if (!bb) return;
  const src = membersOf(bb);
  clipboard = src.map((c) => ({ ...c, node: undefined }));  // snapshot, drop node ref
  clipboardGroups = {};
  const g = busbarGroups.get(bb);
  clipboardGroups[bb] = g ? { x: g.x(), y: g.y(), rot: g.rotation() } : { x: 0, y: 0, rot: 0 };
  document.getElementById("isoBadge").textContent = `copied ${clipboard.length} shape(s)`;
}
function pasteClipboard() {
  if (!clipboard.length) return;
  const step = 2 * GRID;
  const remap = {};
  let first = null;
  for (const s of clipboard) {
    if (!(s.busbar in remap)) {        // each source busbar -> a new id + a shifted frame
      const nb = "bb" + uid++;
      remap[s.busbar] = nb;
      const src = clipboardGroups[s.busbar] || { x: 0, y: 0, rot: 0 };
      const g = busbarGroup(nb);
      g.x(src.x + step); g.y(src.y + step); g.rotation(src.rot);   // shift world by step, keep angle
    }
    const c = { ...s, node: undefined, id: uid, name: "C" + uid, busbar: remap[s.busbar] };
    uid++;
    conductors.push(c); makeNode(c);   // member keeps its local coords inside the new frame
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
  if (!typing && !e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "h" && selGroup.length) {
    toggleAir(); e.preventDefault();   // toggle selection off/on (material <-> Air)
  }
});

// ---- top-level views: Designer / EM Results / Thermal --------------------
let currentView = "designer";
function setView(v) {
  currentView = v;
  document.querySelectorAll("#tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.view === v));
  const fo = $("fieldOverlay"), th = $("thermalView");
  $("geobar").style.display = v === "designer" ? "flex" : "none";   // geometry tools = Designer only
  EM.stop();
  if (v === "designer") {
    fo.style.display = "none"; th.style.display = "none";
  } else if (v === "em") {
    th.style.display = "none"; fo.style.display = "flex";
    const has = !!EM._data;
    $("emEmpty").style.display = has ? "none" : "flex";
    if (has) requestAnimationFrame(() => renderField());
  } else if (v === "thermal") {
    fo.style.display = "none"; th.style.display = "flex";
    if (!EM._data) {
      $("thermalEmpty").style.display = "flex"; $("thermalSummary").style.display = "none";
      $("thermalPrereq").textContent = "Solve the EM problem first (Designer → Solve).";
    } else {
      $("thermalEmpty").style.display = "none";
      if (EM._thermal) requestAnimationFrame(() => { renderThermal(); fillThermalSummary(); });
      else requestAnimationFrame(() => computeThermal());   // auto-run on first entry
    }
  }
}
document.querySelectorAll("#tabs .tab").forEach((t) =>
  t.addEventListener("click", () => setView(t.dataset.view)));

// thermal: solve (reusing the cached EM loss) and render
async function computeThermal() {
  if (!EM._data) return;
  $("thermalStatus").textContent = "solving…";
  try {
    const th = await EM.solveThermal({ airflow: +$("airflow").value, ambient: +$("ambient").value });
    renderThermal(); fillThermalSummary();
    $("thermalStatus").textContent = `max ${th.Tmax.toFixed(1)} °C`;
    $("status").textContent = `Thermal: max ${th.Tmax.toFixed(1)} °C, ΔT ${(th.Tmax - th.Tamb).toFixed(1)} K`;
  } catch (err) { $("thermalStatus").textContent = "thermal error: " + err; console.error(err); }
}
function fillThermalSummary() {
  const th = EM._thermal; if (!th) return;
  const pct = (x) => (th.P_total > 0 ? (100 * x / th.P_total).toFixed(0) : "0");
  let h = `<b>Max ${th.Tmax.toFixed(1)} °C</b> (ambient ${th.Tamb.toFixed(0)} °C, ΔT ${(th.Tmax - th.Tamb).toFixed(1)} K)<br>`;
  h += `Heat ${th.P_total.toFixed(1)} W/m → convection ${th.P_conv.toFixed(1)} (${pct(th.P_conv)}%), `
     + `radiation ${th.P_rad.toFixed(1)} (${pct(th.P_rad)}%)<br>`;
  if (th.surf_qmax > 0) {
    const q = th.surf_qmax;
    h += `Peak surface flux ${q >= 1000 ? (q / 1000).toFixed(1) + " k" : q.toFixed(0) + " "}W/m²<br>`;
  }
  if (th.ir_pairs && th.ir_pairs.length) {
    h += `IR between busbars: ` + th.ir_pairs.slice(0, 5)
      .map((p) => `${p.from}→${p.to} ${p.watts.toFixed(2)} W/m`).join(", ") + `<br>`;
  }
  if (th.dT_air > 0.05) {
    h += `Airflow over ${th.duct_len.toFixed(0)} m: air ${th.Tamb.toFixed(0)} → ${th.air_out.toFixed(1)} °C, `
       + `busbar max ${th.Tmax.toFixed(1)} (inlet) → ${th.bar_max_out.toFixed(1)} °C (outlet)<br>`;
  }
  if (th.vmax > 0) {
    h += `Air: peak ${th.vmax.toFixed(2)} m/s, ΔP ${th.dP.toFixed(3)} Pa over ${th.duct_len.toFixed(0)} m<br>`;
  }
  h += th.conductors.map((c) => `${c.name}: ${c.Tmax.toFixed(1)}°`).join(" · ");
  const el = $("thermalSummary"); el.innerHTML = h; el.style.display = "block";
}
// render the active thermal view (mesh views in engine.js; 3D extrusion here)
function renderThermal() {
  if (EM._thermalView === "3d") draw3D();
  else EM.drawThermal();
}

// ---- representative pseudo-3D: extrude the cross-section to the 1 m length,
// colored by the inlet->outlet temperature gradient; drag to rotate. ---------
EM._3d = { az: 0.6, el: 0.5 };
// world-space (mm) placement of a member = busbar frame ∘ its local placement
function conductorWorld(c) {
  const g = busbarGroup(c.busbar), w = localToWorld(g, c.x, c.y);
  return { x: w.x, y: w.y, rot: g.rotation() + c.rot };
}
function shapeOutline(c) {                       // world-space (mm) cross-section ring
  let pts;
  if (c.type === "rect") { const hw = c.w / 2, hh = c.h / 2; pts = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]; }
  else { pts = []; for (let i = 0; i < 16; i++) { const a = 2 * Math.PI * i / 16; pts.push([c.r * Math.cos(a), c.r * Math.sin(a)]); } }
  const wc = conductorWorld(c), th = (wc.rot * Math.PI) / 180, cs = Math.cos(th), sn = Math.sin(th);
  return pts.map(([lx, ly]) => [wc.x + cs * lx - sn * ly, wc.y + sn * lx + cs * ly]);
}
function draw3D() {
  const th = EM._thermal; if (!th || !conductors.length) return;
  const cv = $("thermalCanvas"), W = (cv.width = cv.clientWidth), H = (cv.height = cv.clientHeight);
  const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, W, H);
  const tmap = {}; th.conductors.forEach((c) => (tmap[c.name] = c.Tmean));
  const Tmin = th.Tamb, Tmax = Math.max(th.bar_max_out, Tmin + 1e-3), dT = th.dT_air || 0, L = (th.duct_len || 1) * 1000;
  const live = conductors.filter((c) => c.material !== "Air");
  let cx = 0, cy = 0; for (const c of live) { const w = conductorWorld(c); cx += w.x; cy += w.y; } cx /= live.length || 1; cy /= live.length || 1;
  const a = EM._3d.az, e = EM._3d.el, ca = Math.cos(a), sa = Math.sin(a), ce = Math.cos(e), se = Math.sin(e);
  const proj = (x, y, z) => {                    // mm -> projection plane (centered)
    const X = x - cx, Y = -(y - cy), Z = z - L / 2;
    const x1 = X * ca + Z * sa, z1 = -X * sa + Z * ca;
    return [x1, Y * ce - z1 * se, Y * se + z1 * ce];   // [u, v, depth]
  };
  const col = (t) => { const c2 = inferno((t - Tmin) / (Tmax - Tmin)); return `rgb(${c2[0]},${c2[1]},${c2[2]})`; };
  const faces = [];
  for (const c of live) {
    const ring = shapeOutline(c), M = ring.length, T0 = tmap[c.name] != null ? tmap[c.name] : Tmin;
    const wc = conductorWorld(c);
    for (let i = 0; i < M; i++) {                 // side faces
      const p = ring[i], q = ring[(i + 1) % M];
      const f = [proj(p[0], p[1], 0), proj(q[0], q[1], 0), proj(q[0], q[1], L), proj(p[0], p[1], L)];
      faces.push({ pts: f, depth: (f[0][2] + f[1][2] + f[2][2] + f[3][2]) / 4, fill: col(T0 + dT * 0.5) });
    }
    faces.push({ pts: ring.map((p) => proj(p[0], p[1], 0)), depth: proj(wc.x, wc.y, 0)[2], fill: col(T0) });           // inlet
    faces.push({ pts: ring.map((p) => proj(p[0], p[1], L)), depth: proj(wc.x, wc.y, L)[2], fill: col(T0 + dT) });      // outlet
  }
  let umin = 1e9, umax = -1e9, vmin = 1e9, vmax = -1e9;
  for (const f of faces) for (const p of f.pts) { umin = Math.min(umin, p[0]); umax = Math.max(umax, p[0]); vmin = Math.min(vmin, p[1]); vmax = Math.max(vmax, p[1]); }
  const s = 0.8 * Math.min(W / (umax - umin || 1), H / (vmax - vmin || 1));
  const ox = W / 2 - s * (umin + umax) / 2, oy = H / 2 - s * (vmin + vmax) / 2;
  const S = (p) => [ox + s * p[0], oy + s * p[1]];
  faces.sort((A, B) => A.depth - B.depth);        // far (small depth) first
  for (const f of faces) {
    ctx.beginPath(); const p0 = S(f.pts[0]); ctx.moveTo(p0[0], p0[1]);
    for (let i = 1; i < f.pts.length; i++) { const pp = S(f.pts[i]); ctx.lineTo(pp[0], pp[1]); }
    ctx.closePath(); ctx.fillStyle = f.fill; ctx.fill();
    ctx.strokeStyle = "rgba(15,15,15,0.45)"; ctx.lineWidth = 0.6; ctx.stroke();
  }
  // length / flow label + colorbar
  ctx.fillStyle = "#222"; ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center";
  const inP = S(proj(cx, cy, 0)), outP = S(proj(cx, cy, L));
  ctx.fillText("inlet (cool)", inP[0], inP[1] + 4); ctx.fillText("outlet (hot) →air", outP[0], outP[1] - 6);
  ctx.textAlign = "left";
  _tbar(ctx, W, H, Tmax, Tmin, "°C", (x) => x.toFixed(0));
}
(function bind3dRotate() {
  const cv = $("thermalCanvas"); let drag = null;
  cv.addEventListener("mousedown", (e) => { if (EM._thermalView === "3d") drag = { x: e.clientX, y: e.clientY, az: EM._3d.az, el: EM._3d.el }; });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    EM._3d.az = drag.az + (e.clientX - drag.x) * 0.01;
    EM._3d.el = Math.max(-1.4, Math.min(1.4, drag.el + (e.clientY - drag.y) * 0.01));
    draw3D();
  });
  window.addEventListener("mouseup", () => { drag = null; });
})();

$("computeThermal").onclick = computeThermal;
$("thermalField").addEventListener("change", (e) => {
  EM._thermalView = e.target.value;
  if (EM._thermal) renderThermal();
});
// keep the in-view buttons working, now as view switches
function showFieldView() { setView("em"); }
function hideFieldView() { setView("designer"); }
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
  if (!EM.ready) return;
  const active = conductors.filter((c) => c.material !== "Air");
  if (!active.length) { $("status").textContent = "Add at least one non-air conductor."; return; }
  $("status").textContent = "Meshing + solving (in-browser)…";
  $("progress").hidden = false;        // the editor stays interactive meanwhile
  try {
    const t0 = performance.now();
    const data = await EM.solve(toSceneDict(true));   // skip Air items; runs in the worker
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
  if (EM._thermal && document.getElementById("thermalView").style.display === "flex")
    renderThermal();
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
