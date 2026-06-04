"""Gmsh backend: build meshes for canonical geometries.

This module is the only place that imports ``gmsh``. It converts a geometry
into the plain-array :class:`~emsim.mesh.mesh.Mesh`, after which nothing
downstream needs Gmsh.

* :func:`mesh_disks` -- a set of (possibly nested / composite) circular
  conductor regions embedded in an air disk, with the outer circle marked as
  the Dirichlet (gauge-pin) boundary and a size field refining toward the
  conductor surfaces.
* :func:`mesh_round_wire` -- the milestone-1 single-wire convenience wrapper.

Nested regions (e.g. a composite core + shell sharing a centre) are resolved
by the *innermost containing disk* rule, evaluated at a sample interior point
of each meshed sub-region, so concentric regions need not be distinguishable
by centroid or area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emsim.mesh.mesh import Mesh

WIRE_TAG = 1
AIR_TAG = 2
KELVIN_TAG = 3


@dataclass
class Disk:
    """A circular conductor region: centre, radius and the region tag."""

    cx: float
    cy: float
    radius: float
    tag: int


def _innermost_tag(px: float, py: float, disks: list[Disk], air_tag: int) -> int:
    """Region tag at a point: innermost (smallest) disk containing it, else air."""
    best_tag = air_tag
    best_r = np.inf
    for d in disks:
        if np.hypot(px - d.cx, py - d.cy) <= d.radius * (1.0 + 1e-9):
            if d.radius < best_r:
                best_r = d.radius
                best_tag = d.tag
    return best_tag


def _add_background_view(background: tuple[np.ndarray, np.ndarray, np.ndarray]) -> int:
    """Create a Gmsh PostView holding per-node target sizes on a prior mesh.

    ``background = (nodes, tris, node_sizes)``. Returns the view tag. Used as a
    background mesh field so a remesh follows the adaptive size distribution.
    """
    import gmsh

    nodes, tris, sizes = background
    P = nodes[tris]  # (T,3,2)
    S = sizes[tris]  # (T,3)
    T = tris.shape[0]
    data = np.concatenate(
        [P[:, :, 0], P[:, :, 1], np.zeros((T, 3)), S], axis=1
    ).reshape(-1)
    view = gmsh.view.add("bgmesh")
    gmsh.view.addListData(view, "ST", T, data.tolist())
    return view


def mesh_disks(
    disks: list[Disk],
    R: float,
    air_tag: int = AIR_TAG,
    *,
    lc_surface: float,
    lc_far: float,
    grade_distance: float | None = None,
    background: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Mesh circular conductor regions inside an air disk of radius ``R``.

    Parameters
    ----------
    disks:
        Conductor regions. For a composite (nested) bar, pass both the inner
        disk and the outer disk; the annular shell is resolved automatically.
    R:
        Outer air radius (m), with ``R`` larger than all conductors.
    air_tag:
        Region tag for the air.
    lc_surface, lc_far:
        Target element sizes at the conductor surfaces and the outer boundary.
    grade_distance:
        Distance over which size grows from ``lc_surface`` to ``lc_far``.
        Defaults to twice the largest conductor radius.
    """
    import gmsh

    if grade_distance is None:
        grade_distance = 2.0 * max(d.radius for d in disks)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        occ = gmsh.model.occ

        outer = occ.addDisk(0.0, 0.0, 0.0, R, R)
        disk_tags = [occ.addDisk(d.cx, d.cy, 0.0, d.radius, d.radius) for d in disks]
        occ.fragment([(2, outer)], [(2, t) for t in disk_tags])
        occ.synchronize()

        if background is not None:
            # Follow a per-node target-size field from a prior solution.
            view = _add_background_view(background)
            pv = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(pv, "ViewTag", view)
            gmsh.model.mesh.field.setAsBackgroundMesh(pv)
        else:
            # Size field: refine toward conductor boundary curves (relative to the
            # skin depth via lc_surface). Conductor circles are identified by their
            # bounding box (centre + radius); the outer R circle is excluded.
            cond_curves: list[int] = []
            for _, ctag in gmsh.model.getEntities(1):
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, ctag)
                ccx, ccy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
                crad = 0.5 * (xmax - xmin)
                for d in disks:
                    if (
                        abs(crad - d.radius) < 1e-6 * R
                        and np.hypot(ccx - d.cx, ccy - d.cy) < 1e-6 * R
                    ):
                        cond_curves.append(ctag)
                        break

            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist, "CurvesList", cond_curves)
            gmsh.model.mesh.field.setNumber(dist, "Sampling", 300)
            thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr, "InField", dist)
            gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_surface)
            gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc_far)
            gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(thr, "DistMax", grade_distance)
            gmsh.model.mesh.field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        mesh = _extract_disks(disks, air_tag)
    finally:
        gmsh.finalize()

    return mesh


def _extract_disks(disks: list[Disk], air_tag: int) -> Mesh:
    """Extract the meshed model, classifying each surface by innermost disk."""
    import gmsh

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    max_tag = int(node_tags.max())
    tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
    tag_to_idx[node_tags] = np.arange(node_tags.size)
    nodes = coords[:, :2].copy()

    R = max(np.hypot(nodes[:, 0], nodes[:, 1]))

    npe = {2: 3, 9: 6}  # gmsh element type -> nodes per triangle (P1 / P2)
    tris_list: list[np.ndarray] = []
    region_list: list[np.ndarray] = []
    for _, surf in gmsh.model.getEntities(2):
        etypes, _etags, enodes = gmsh.model.mesh.getElements(2, surf)
        for et, en in zip(etypes, enodes):
            if et not in npe:  # triangles only
                continue
            conn = tag_to_idx[np.asarray(en, dtype=np.int64).reshape(-1, npe[et])]
            # classify this surface by a sample interior point (first centroid)
            sample = nodes[conn[0, :3]].mean(axis=0)
            region = _innermost_tag(sample[0], sample[1], disks, air_tag)
            tris_list.append(conn)
            region_list.append(np.full(conn.shape[0], region, dtype=np.int64))

    tris = np.vstack(tris_list)
    region_tag = np.concatenate(region_list)
    mesh = Mesh(nodes=nodes, tris=tris, region_tag=region_tag)

    radii = np.hypot(nodes[:, 0], nodes[:, 1])
    mesh.boundary_nodes = np.where(np.abs(radii - R) < 1e-6 * R)[0].astype(np.int64)
    return mesh


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
    """Mesh arbitrary placed conductor shapes inside an air disk of radius ``R``.

    ``regions`` is a list of ``(shape, placement, tag)`` where ``shape`` exposes
    ``occ_add(occ, placement)``, ``contains(x, y, placement)`` and ``area()``
    (see :mod:`emsim.geometry.shapes`). Overlapping/nested regions are resolved
    by the innermost (smallest-area) containing shape; everything else is air.
    """
    import gmsh

    if grade_distance is None:
        grade_distance = 2.0 * max(s.char_size() for s, _, _ in regions)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        occ = gmsh.model.occ
        outer = occ.addDisk(0.0, 0.0, 0.0, R, R)
        region_tags = [shape.occ_add(occ, pl) for shape, pl, _ in regions]
        occ.fragment([(2, outer)], [(2, t) for t in region_tags])
        occ.synchronize()

        # refine toward every non-outer curve (conductor / enclosure boundaries)
        cond_curves: list[int] = []
        for _, ctag in gmsh.model.getEntities(1):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, ctag)
            half = 0.5 * max(xmax - xmin, ymax - ymin)
            if half < R * (1.0 - 1e-3):
                cond_curves.append(ctag)

        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", cond_curves)
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 400)
        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_surface)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc_far)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", grade_distance)
        gmsh.model.mesh.field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        mesh = _extract_model(regions, air_tag)
    finally:
        gmsh.finalize()
    return mesh


def _extract_model(regions: list, air_tag: int) -> Mesh:
    """Extract the meshed model, classifying surfaces by innermost shape."""
    import gmsh

    # order regions by area ascending so the first containing one is innermost
    ordered = sorted(regions, key=lambda r: r[0].area())

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    max_tag = int(node_tags.max())
    tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
    tag_to_idx[node_tags] = np.arange(node_tags.size)
    nodes = coords[:, :2].copy()
    R = float(np.max(np.hypot(nodes[:, 0], nodes[:, 1])))

    npe = {2: 3, 9: 6}
    tris_list: list[np.ndarray] = []
    region_list: list[np.ndarray] = []
    for _, surf in gmsh.model.getEntities(2):
        etypes, _etags, enodes = gmsh.model.mesh.getElements(2, surf)
        for et, en in zip(etypes, enodes):
            if et not in npe:
                continue
            conn = tag_to_idx[np.asarray(en, dtype=np.int64).reshape(-1, npe[et])]
            sx, sy = nodes[conn[0, :3]].mean(axis=0)
            tag = air_tag
            for shape, pl, rtag in ordered:
                if shape.contains(sx, sy, pl):
                    tag = rtag
                    break
            tris_list.append(conn)
            region_list.append(np.full(conn.shape[0], tag, dtype=np.int64))

    mesh = Mesh(
        nodes=nodes,
        tris=np.vstack(tris_list),
        region_tag=np.concatenate(region_list),
    )
    radii = np.hypot(nodes[:, 0], nodes[:, 1])
    mesh.boundary_nodes = np.where(np.abs(radii - R) < 1e-6 * R)[0].astype(np.int64)
    return mesh


def mesh_kelvin_disk(
    rim_xy: np.ndarray,
    center_size: float,
    verbose: bool = False,
) -> tuple[Mesh, np.ndarray, int]:
    r"""Mesh the Kelvin (mirror) disk that represents the open exterior.

    The exterior region r > R is mapped by the 2D inversion ``rho = R^2 / r``
    onto a disk of radius R in ``rho``-space. Because the 2D Laplacian's
    Dirichlet integral is conformally invariant, the exterior energy equals the
    *standard* (constant-nu) stiffness assembled on this disk -- no
    radius-dependent coefficient is needed in 2D. The rim ``rho = R`` coincides
    with the physical outer circle (shared nodes) and the centre ``rho = 0`` is
    the image of infinity (pinned to A_z = 0).

    Parameters
    ----------
    rim_xy:
        ``(K, 2)`` rim vertices, ordered counter-clockwise, taken from the
        physical mesh's outer-boundary nodes. The Kelvin disk reuses these
        exact points (transfinite boundary segments => no extra rim nodes) so
        the rim is conforming with the physical boundary.
    center_size:
        Target mesh size at the disk centre (the far field is smooth, so this
        can be coarse).

    Returns
    -------
    mesh:
        The Kelvin disk mesh (all elements tagged ``KELVIN_TAG``).
    rim_nodes:
        Node indices of the rim, in the same order as ``rim_xy`` (for merging).
    center_node:
        Node index of the centre (to be pinned).
    """
    import gmsh

    rim_xy = np.asarray(rim_xy, dtype=np.float64)
    k = rim_xy.shape[0]
    # local spacing for each rim point (distance to the next, cyclic)
    nxt = np.roll(rim_xy, -1, axis=0)
    spacing = np.hypot(*(nxt - rim_xy).T)
    R = float(np.median(np.hypot(rim_xy[:, 0], rim_xy[:, 1])))

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        geo = gmsh.model.geo
        pts = [
            geo.addPoint(rim_xy[i, 0], rim_xy[i, 1], 0.0, spacing[i]) for i in range(k)
        ]
        center = geo.addPoint(0.0, 0.0, 0.0, center_size)
        lines = [geo.addLine(pts[i], pts[(i + 1) % k]) for i in range(k)]
        for ln in lines:
            geo.mesh.setTransfiniteCurve(ln, 2)  # endpoints only -> no rim nodes added
        loop = geo.addCurveLoop(lines)
        surf = geo.addPlaneSurface([loop])
        geo.synchronize()
        gmsh.model.mesh.embed(0, [center], 2, surf)  # force centre to be a node
        # Refine toward the rim (Distance/Threshold) so the near-interface field
        # is resolved like the physical boundary, coarsening toward the centre.
        rim_lc = float(np.median(spacing))
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", lines)
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 400)
        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", rim_lc)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", center_size)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", R)
        gmsh.model.mesh.field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        max_tag = int(node_tags.max())
        tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
        tag_to_idx[node_tags] = np.arange(node_tags.size)
        nodes = coords[:, :2].copy()

        etypes, _etags, enodes = gmsh.model.mesh.getElements(2, surf)
        tris = None
        for et, en in zip(etypes, enodes):
            if et == 2:
                tris = tag_to_idx[np.asarray(en, dtype=np.int64).reshape(-1, 3)]
        region_tag = np.full(tris.shape[0], KELVIN_TAG, dtype=np.int64)

        rim_nodes = np.array(
            [int(tag_to_idx[int(gmsh.model.mesh.getNodes(0, p)[0][0])]) for p in pts],
            dtype=np.int64,
        )
        center_node = int(tag_to_idx[int(gmsh.model.mesh.getNodes(0, center)[0][0])])
    finally:
        gmsh.finalize()

    mesh = Mesh(nodes=nodes, tris=tris, region_tag=region_tag)
    return mesh, rim_nodes, center_node


def mesh_round_wire(
    a: float,
    R: float,
    lc_surface: float,
    lc_far: float,
    grade_distance: float | None = None,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Mesh a single round wire (radius ``a``, tag ``WIRE_TAG``) in air (``AIR_TAG``).

    A skin-relative size field refines toward the conductor surface.
    """
    return mesh_disks(
        [Disk(0.0, 0.0, a, WIRE_TAG)],
        R,
        AIR_TAG,
        lc_surface=lc_surface,
        lc_far=lc_far,
        grade_distance=grade_distance,
        order=order,
        verbose=verbose,
    )
