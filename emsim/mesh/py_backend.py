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


def _poisson_fill(R, boundary, hfield, lc_surface, lc_far) -> np.ndarray:
    """Graded interior points: dense (lc_surface) near boundaries, coarse far."""
    cell = lc_far  # spatial-hash cell >= max spacing
    grid: dict[tuple[int, int], list[np.ndarray]] = {}

    def add(p):
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(p)

    def ok(p, h):
        ci, cj = int(p[0] // cell), int(p[1] // cell)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for q in grid.get((ci + di, cj + dj), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < h * h:
                        return False
        return True

    for p in boundary:  # boundary points seed the hash but are NOT returned
        add(p)

    # candidate interior points on a fine grid, processed fine-spacing-first
    accepted: list[np.ndarray] = []
    step = lc_surface
    gx = np.arange(-R + step, R, step)
    xx, yy = np.meshgrid(gx, gx)
    cand = np.column_stack([xx.ravel(), yy.ravel()])
    cand = cand[np.hypot(cand[:, 0], cand[:, 1]) < R - 0.5 * lc_surface]
    h_cand = hfield(cand)
    for idx in np.argsort(h_cand):  # smallest target spacing first
        p, h = cand[idx], h_cand[idx]
        if ok(p, 0.8 * h):
            add(p)
            accepted.append(p)

    return np.asarray(accepted) if accepted else np.empty((0, 2))


def mesh_model(
    regions: list,
    R: float,
    air_tag: int = AIR_TAG,
    *,
    lc_surface: float,
    lc_far: float,
    grade_distance: float | None = None,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Mesh placed conductor shapes in an air disk (gmsh-free). P1 only."""
    if order != 1:
        raise NotImplementedError("py_backend supports first-order (P1) only")
    if grade_distance is None:
        grade_distance = 2.0 * max(s.char_size() for s, _, _ in regions)

    cond_bdry = np.vstack([_outline_points(s, pl, lc_surface) for s, pl, _ in regions])
    nO = max(48, int(math.ceil(2 * math.pi * R / lc_far)))
    t = np.linspace(0, 2 * np.pi, nO, endpoint=False)
    outer = np.column_stack([R * np.cos(t), R * np.sin(t)])
    boundary = np.vstack([cond_bdry, outer])

    tree = cKDTree(cond_bdry)
    slope = (lc_far - lc_surface) / grade_distance

    def hfield(P):
        d = tree.query(P)[0]
        return np.clip(lc_surface + slope * d, lc_surface, lc_far)

    interior = _poisson_fill(R, boundary, hfield, lc_surface, lc_far)
    pts = np.vstack([boundary, interior]) if interior.size else boundary

    # Deduplicate coincident points. Overlapping conductor outlines (e.g. a
    # composite T/L busbar made of several rectangles) sample the same point
    # twice; duplicate nodes orphan stiffness rows -> "factor exactly singular".
    quant = np.round(pts / (lc_surface * 1e-3)).astype(np.int64)
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
