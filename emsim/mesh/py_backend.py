"""Pure-Python mesher (NumPy + scipy.spatial) — a gmsh-free backend.

Gmsh is a native library that cannot run in WebAssembly (Pyodide), so the web
build needs a mesher built only from packages available there. This backend
produces the same plain-array :class:`~emsim.mesh.mesh.Mesh` as
:mod:`emsim.mesh.gmsh_backend`, so everything downstream is unchanged.

Approach: sample every conductor outline and the outer circle into boundary
points, fill the interior with graded Poisson-disk points (fine near conductor
surfaces, relative to the skin depth, via a size field), Delaunay-triangulate,
then tag each triangle by the innermost containing shape. P1 only.

It is intentionally simple; gmsh remains the higher-quality default for the
desktop app. This backend targets "good enough in the browser".
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from emsim.geometry.shapes import Circle, Placement, Polygon, Rectangle
from emsim.mesh.gmsh_backend import AIR_TAG, WIRE_TAG, _innermost_tag
from emsim.mesh.mesh import Mesh


def _outline_points(shape, pl: Placement, h: float) -> np.ndarray:
    """Sample a shape's boundary into world-space points spaced ~h apart."""
    if isinstance(shape, Circle):
        n = max(16, int(math.ceil(2 * math.pi * shape.radius / h)))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([pl.x + shape.radius * np.cos(t),
                                pl.y + shape.radius * np.sin(t)])
    if isinstance(shape, Rectangle):
        hw, hh = shape.width / 2, shape.height / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    elif isinstance(shape, Polygon):
        corners = list(shape.points)
    else:  # pragma: no cover
        raise TypeError(f"unsupported shape {shape!r}")
    c, s = math.cos(math.radians(pl.rotation)), math.sin(math.radians(pl.rotation))
    pts = []
    n = len(corners)
    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        seg = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / h)))
        for k in range(seg):
            f = k / seg
            lx, ly = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
            pts.append((pl.x + c * lx - s * ly, pl.y + s * lx + c * ly))
    return np.asarray(pts)


def _poisson_fill(R, boundary, hfield, lc_min, lc_far,
                  fine_bbox=None, grade_distance=None) -> np.ndarray:
    """Graded interior points: dense (lc_min) near boundaries, coarse far.

    Candidates are generated multi-resolution: coarse grids (lc_far and a mid
    level) over the whole disk for the bulk, plus a fine (lc_min) grid only over
    the conductor bounding box + grading band -- and that fine grid is capped, so
    a large domain with thin features (a sheet-metal enclosure) stays tractable
    instead of generating ~(2R/lc_min)^2 candidates over the entire disk.
    """
    sets = []
    # coarse + mid candidates over the whole disk fill the bulk cheaply
    for step in (lc_far, max(lc_far / 4.0, lc_min)):
        g = np.arange(-R + step, R, step)
        xx, yy = np.meshgrid(g, g)
        sets.append(np.column_stack([xx.ravel(), yy.ravel()]))
    # fine candidates only near the conductors (their bbox + grading band), capped
    if fine_bbox is not None and lc_min < lc_far:
        pad = grade_distance or 0.0
        x0 = max(fine_bbox[0] - pad, -R); x1 = min(fine_bbox[1] + pad, R)
        y0 = max(fine_bbox[2] - pad, -R); y1 = min(fine_bbox[3] + pad, R)
        step = lc_min
        n_est = max(1.0, (x1 - x0) / step) * max(1.0, (y1 - y0) / step)
        cap = 150000.0
        if n_est > cap:
            step *= math.sqrt(n_est / cap)   # coarsen the fine grid to stay bounded
        gx = np.arange(x0, x1, step); gy = np.arange(y0, y1, step)
        xx, yy = np.meshgrid(gx, gy)
        sets.append(np.column_stack([xx.ravel(), yy.ravel()]))

    cand = np.vstack(sets)
    cand = cand[np.hypot(cand[:, 0], cand[:, 1]) < R - 0.5 * lc_min]
    h_cand = hfield(cand)
    order = np.argsort(h_cand)        # smallest target spacing first
    cand, h_cand = cand[order], h_cand[order]

    # Poisson-disk reject against an incrementally grown cKDTree. A uniform
    # spatial hash can't handle the 0.9 mm-to-50 mm size range of an enclosure
    # (fine cells overload); a periodically rebuilt KD-tree + per-batch dedup
    # stays near-linear regardless of the size spread.
    base = np.asarray(boundary, dtype=float)
    tree = cKDTree(base)
    chunks = []
    BATCH = 256
    for i in range(0, len(cand), BATCH):
        P, H = cand[i:i + BATCH], h_cand[i:i + BATCH]
        keep = tree.query(P)[0] >= 0.8 * H          # far enough from fixed points
        Pk, Hk = P[keep], H[keep]
        picked = []
        for k in range(len(Pk)):
            p, h = Pk[k], Hk[k]
            r2 = (0.8 * h) ** 2
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= r2 for q in picked):
                picked.append(p)
        if picked:
            ch = np.asarray(picked)
            chunks.append(ch)
            base = np.vstack([base, ch])
            tree = cKDTree(base)

    return np.vstack(chunks) if chunks else np.empty((0, 2))


def mesh_model(
    regions: list,
    R: float,
    air_tag: int = AIR_TAG,
    *,
    lc_surface: float | None = None,
    lc_far: float,
    region_sizes: list[float] | None = None,
    grade_distance: float | None = None,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Mesh placed conductor shapes in an air disk (gmsh-free). P1 only.

    ``region_sizes`` (optional, one target edge length per region, aligned with
    ``regions``) sizes each region by its own characteristic dimension and grades
    locally back out to ``lc_far``. This keeps a thin sheet-metal wall fine only
    near the wall instead of pinning the *whole* domain to the thinnest feature
    (which is what made enclosures blow the node count up). Falls back to the
    scalar ``lc_surface`` for every region when omitted (back-compat).
    """
    if order != 1:
        raise NotImplementedError("py_backend supports first-order (P1) only")
    if region_sizes is None:
        if lc_surface is None:
            raise ValueError("provide lc_surface or region_sizes")
        region_sizes = [lc_surface] * len(regions)
    lc_min = min(region_sizes)
    if grade_distance is None:
        grade_distance = 2.0 * max(s.char_size() for s, _, _ in regions)

    # sample each region's outline at its own target spacing, remembering the
    # local size per boundary point so the size field grades from it
    cond_bdry_parts, bdry_lc_parts = [], []
    for (s, pl, _), h in zip(regions, region_sizes):
        p = _outline_points(s, pl, h)
        cond_bdry_parts.append(p)
        bdry_lc_parts.append(np.full(len(p), h))
    cond_bdry = np.vstack(cond_bdry_parts)
    bdry_lc = np.concatenate(bdry_lc_parts)

    nO = max(48, int(math.ceil(2 * math.pi * R / lc_far)))
    t = np.linspace(0, 2 * np.pi, nO, endpoint=False)
    outer = np.column_stack([R * np.cos(t), R * np.sin(t)])
    boundary = np.vstack([cond_bdry, outer])

    tree = cKDTree(cond_bdry)

    def hfield(P):
        d, idx = tree.query(P)
        lc_near = bdry_lc[idx]                       # size of the nearest outline
        slope = (lc_far - lc_near) / grade_distance
        return np.clip(lc_near + slope * d, lc_near, lc_far)

    fine_bbox = (float(cond_bdry[:, 0].min()), float(cond_bdry[:, 0].max()),
                 float(cond_bdry[:, 1].min()), float(cond_bdry[:, 1].max()))
    interior = _poisson_fill(R, boundary, hfield, lc_min, lc_far,
                             fine_bbox=fine_bbox, grade_distance=grade_distance)
    pts = np.vstack([boundary, interior]) if interior.size else boundary

    # Deduplicate coincident points. Overlapping conductor outlines (e.g. a
    # composite T/L busbar made of several rectangles) sample the same point
    # twice; duplicate nodes orphan stiffness rows -> "factor exactly singular".
    quant = np.round(pts / (lc_min * 1e-3)).astype(np.int64)
    _, keep = np.unique(quant, axis=0, return_index=True)
    pts = pts[np.sort(keep)]

    tris = Delaunay(pts).simplices
    centroids = pts[tris].mean(axis=1)
    inside = np.hypot(centroids[:, 0], centroids[:, 1]) <= R * (1 + 1e-9)
    tris = tris[inside]
    centroids = centroids[inside]

    # Drop degenerate (near-zero-area) triangles, then remove any node left
    # unreferenced. Either would make the stiffness matrix exactly singular.
    v = pts[tris]
    area2 = np.abs((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
                   - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
    good = area2 > 1e-9 * np.median(area2)
    tris = tris[good]
    centroids = centroids[good]
    used = np.unique(tris)
    remap = np.full(pts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    pts = pts[used]
    tris = remap[tris]

    ordered = sorted(regions, key=lambda r: r[0].area())
    region_tag = np.empty(tris.shape[0], dtype=np.int64)
    for i, (cx, cy) in enumerate(centroids):
        tag = air_tag
        for shape, pl, rtag in ordered:
            if shape.contains(cx, cy, pl):
                tag = rtag
                break
        region_tag[i] = tag

    mesh = Mesh(nodes=pts, tris=tris, region_tag=region_tag)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    mesh.boundary_nodes = np.where(radii >= R * (1 - 1e-9))[0].astype(np.int64)
    return mesh


def mesh_round_wire(a, R, lc_surface, lc_far, grade_distance=None, order=1, verbose=False):
    """Single round wire in air (gmsh-free), for validating this backend."""
    return mesh_model(
        [(Circle(a), Placement(0, 0, 0), WIRE_TAG)],
        R, AIR_TAG, lc_surface=lc_surface, lc_far=lc_far,
        grade_distance=grade_distance, order=order,
    )
