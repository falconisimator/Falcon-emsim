r"""Assemble an open-boundary mesh by gluing a Kelvin disk to a physical disk.

The physical disk (radius R, containing the conductors) is meshed normally;
its outer-circle nodes form the interface. A Kelvin disk (rho-space image of
the exterior under rho = R^2 / r) is meshed with a *conforming* rim, then the
two are merged into a single :class:`~emsim.mesh.mesh.Mesh`:

* rim nodes are shared (same global index) -> conforming coupling at r = R;
* the Kelvin centre (image of infinity) is the sole Dirichlet pin (A_z = 0),
  which also fixes the gauge.

Because the 2D Kelvin transform is conformal, the merged mesh is solved with
the *ordinary* constant-coefficient assembler -- the Kelvin elements are simply
air (KELVIN_TAG) occupying rho-coordinates.
"""

from __future__ import annotations

import numpy as np

from emsim.mesh.mesh import Mesh


def ordered_boundary(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return the outer-boundary node indices and coords ordered CCW by angle."""
    idx = mesh.boundary_nodes
    xy = mesh.nodes[idx]
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    order = np.argsort(theta)
    return idx[order], xy[order]


def combine_kelvin(
    physical: Mesh,
    kelvin: Mesh,
    rim_physical: np.ndarray,
    rim_kelvin: np.ndarray,
    center_kelvin: int,
) -> Mesh:
    """Merge a physical mesh and a Kelvin disk into one open-boundary mesh.

    ``rim_physical[i]`` and ``rim_kelvin[i]`` are the matched rim node indices
    (same order). ``center_kelvin`` is the Kelvin centre node (pinned).
    """
    if rim_physical.shape != rim_kelvin.shape:
        raise ValueError(
            f"rim mismatch: physical {rim_physical.shape} vs kelvin {rim_kelvin.shape}"
        )
    npn = physical.num_nodes
    nkn = kelvin.num_nodes
    mapping = np.full(nkn, -1, dtype=np.int64)
    mapping[rim_kelvin] = rim_physical  # merge rim onto physical indices

    nxt = npn
    for j in range(nkn):
        if mapping[j] < 0:
            mapping[j] = nxt
            nxt += 1
    n_total = nxt

    nodes = np.zeros((n_total, 2), dtype=np.float64)
    nodes[:npn] = physical.nodes
    nodes[mapping] = kelvin.nodes  # rim coords identical, so overwrite is a no-op

    kel_tris = mapping[kelvin.tris]
    tris = np.vstack([physical.tris, kel_tris])
    region_tag = np.concatenate([physical.region_tag, kelvin.region_tag])
    center_global = int(mapping[center_kelvin])

    return Mesh(
        nodes=nodes,
        tris=tris,
        region_tag=region_tag,
        boundary_nodes=np.array([center_global], dtype=np.int64),
    )


def open_mesh(physical: Mesh, center_size: float | None = None) -> Mesh:
    """Build the combined open-boundary mesh from a physical disk mesh.

    The physical mesh must carry its outer-circle nodes in ``boundary_nodes``.
    """
    from emsim.mesh.gmsh_backend import mesh_kelvin_disk

    rim_idx, rim_xy = ordered_boundary(physical)
    R = float(np.hypot(rim_xy[0, 0], rim_xy[0, 1]))
    if center_size is None:
        center_size = R / 6.0
    kelvin, rim_kelvin, center_kelvin = mesh_kelvin_disk(rim_xy, center_size)
    return combine_kelvin(physical, kelvin, rim_idx, rim_kelvin, center_kelvin)
