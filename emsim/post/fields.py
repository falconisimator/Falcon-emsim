r"""Derived fields from the solved vector potential.

Current density (z-component), per element / quadrature point:

    J_z = -j omega sigma A_z + sigma (V_dot/L)

Magnetic flux density (in-plane), per element (constant on P1):

    B_x = dA_z/dy,   B_y = -dA_z/dx
"""

from __future__ import annotations

import numpy as np

from emsim.fem import elements
from emsim.mesh.gmsh_backend import KELVIN_TAG
from emsim.results import Solution


def element_B(solution: Solution) -> np.ndarray:
    """Per-element complex flux density at the centroid, shape ``(M, 2)``.

    For P1 this is the (constant) element field; for P2 it is B evaluated at the
    centroid -- a representative value for maps and recovery.
    """
    from emsim.fem import shapes

    mesh = solution.mesh
    order = mesh.order
    verts = mesh.triangle_vertices()
    gradL, _ = elements.shape_gradients(verts)  # (M,3,2) P1 barycentric grads
    centroid = np.array([[1 / 3, 1 / 3, 1 / 3]])
    dNdL = shapes.shape_grads_bary(order, centroid)[0]  # (K,3)
    gradN = np.einsum("kl,mld->mkd", dNdL, gradL)  # (M,K,2)
    a_el = solution.a[mesh.tris]  # (M,K)
    gradA = np.einsum("mk,mkd->md", a_el, gradN)
    Bx = gradA[:, 1]
    By = -gradA[:, 0]
    return np.stack([Bx, By], axis=1)


def element_B_magnitude(solution: Solution) -> np.ndarray:
    """Per-element |B| using complex (phasor amplitude) magnitudes, shape ``(M,)``."""
    B = element_B(solution)
    return np.sqrt(np.abs(B[:, 0]) ** 2 + np.abs(B[:, 1]) ** 2)


def nodal_B(solution: Solution, physical_only: bool = True) -> np.ndarray:
    """Area-weighted nodal recovery of B (complex), shape ``(N, 2)``.

    Smooths the piecewise-constant element B onto nodes -- used for smooth |B|
    maps, contour (force) integration, and the ZZ error estimator. Kelvin-disk
    elements (rho-space) are excluded by default so the recovered field lives
    in physical coordinates.
    """
    mesh = solution.mesh
    B_el = element_B(solution)  # (M,2)
    area = mesh.areas()
    w = area.copy()
    if physical_only:
        w[mesh.region_tag == KELVIN_TAG] = 0.0
    acc = np.zeros((mesh.num_nodes, 2), dtype=np.complex128)
    wsum = np.zeros(mesh.num_nodes, dtype=np.float64)
    for k in range(3):  # vertices only
        np.add.at(acc, mesh.tris[:, k], B_el * w[:, None])
        np.add.at(wsum, mesh.tris[:, k], w)
    good = wsum > 0
    acc[good] /= wsum[good, None]
    return acc


def _physical_triangulation(solution: Solution):
    """matplotlib Triangulation + trifinder over the physical (non-Kelvin) tris."""
    from matplotlib.tri import Triangulation

    mesh = solution.mesh
    mask = mesh.region_tag != KELVIN_TAG
    phys_tris = mesh.tris[mask][:, :3]  # vertices only (matplotlib needs 3-node)
    triang = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], phys_tris)
    return triang, triang.get_trifinder(), phys_tris


def sample_B(
    solution: Solution, pts: np.ndarray, nB: np.ndarray | None = None
) -> np.ndarray:
    """Interpolate the recovered nodal B at arbitrary points, shape ``(P, 2)``.

    Points outside the physical mesh return NaN. Used for EMI sampling and
    Maxwell-stress contour integration.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    if nB is None:
        nB = nodal_B(solution)
    triang, finder, phys_tris = _physical_triangulation(solution)
    ti = finder(pts[:, 0], pts[:, 1])
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.complex128)
    valid = ti >= 0
    if not np.any(valid):
        return out
    nodes = solution.mesh.nodes
    tri_nodes = phys_tris[ti[valid]]  # (V,3)
    v = nodes[tri_nodes]  # (V,3,2)
    px, py = pts[valid, 0], pts[valid, 1]
    xA, yA = v[:, 0, 0], v[:, 0, 1]
    xB, yB = v[:, 1, 0], v[:, 1, 1]
    xC, yC = v[:, 2, 0], v[:, 2, 1]
    den = (yB - yC) * (xA - xC) + (xC - xB) * (yA - yC)
    l1 = ((yB - yC) * (px - xC) + (xC - xB) * (py - yC)) / den
    l2 = ((yC - yA) * (px - xC) + (xA - xC) * (py - yC)) / den
    l3 = 1.0 - l1 - l2
    bary = np.stack([l1, l2, l3], axis=1)  # (V,3)
    Bnodes = nB[tri_nodes]  # (V,3,2)
    out[valid] = np.einsum("vk,vkd->vd", bary, Bnodes)
    return out


def _group_index(solution: Solution, name: str) -> int:
    return solution.group_order.index(name)


def element_group_voltage(solution: Solution) -> np.ndarray:
    """Per-element V_dot/L from the owning parallel group (0 where none)."""
    mesh = solution.mesh
    uvals = np.zeros(mesh.num_tris, dtype=np.complex128)
    for group in solution.groups:
        gi = _group_index(solution, group.name)
        mask = mesh.tris_in_regions(group.tag_set)
        uvals[mask] = solution.u[gi]
    return uvals


def element_Jz(solution: Solution) -> np.ndarray:
    """Per-element complex current density J_z at the centroid, shape ``(M,)``.

    ``J_z = -j omega sigma A_z + sigma (V_dot/L)``; the instantaneous distribution
    at phase ``phi`` is ``Re(J_z e^{j phi})``.
    """
    from emsim.fem import shapes
    from emsim.fem.assembly import element_material_arrays

    mesh = solution.mesh
    _, sigma = element_material_arrays(mesh, solution.materials)
    n = shapes.shape_values(mesh.order, np.array([[1 / 3, 1 / 3, 1 / 3]]))[0]  # (K,)
    a_c = solution.a[mesh.tris] @ n  # A_z at element centroids (M,)
    u_el = element_group_voltage(solution)
    return -1j * solution.omega * sigma * a_c + sigma * u_el


def current_density_at_quadrature(
    solution: Solution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex J_z at each quadrature point of every element.

    Returns ``(Jz, weights, sigma)`` where ``Jz`` and ``weights`` are
    ``(M, Q)`` and ``sigma`` is the per-element conductivity ``(M,)``.
    """
    from emsim.fem import shapes
    from emsim.fem.assembly import element_material_arrays

    mesh = solution.mesh
    order = mesh.order
    _, sigma = element_material_arrays(mesh, solution.materials)
    area = mesh.areas()
    # |J|^2 is degree 2*order; integrate exactly.
    bary, w = shapes.quadrature(2 if order == 1 else 4)
    N = shapes.shape_values(order, bary)  # (Q,K)
    weights = area[:, None] * w[None, :]  # (M,Q)
    a_el = solution.a[mesh.tris]  # (M,K)
    a_q = np.einsum("qk,mk->mq", N, a_el)  # A_z at quad points (M,Q)
    u_el = element_group_voltage(solution)  # (M,)
    omega = solution.omega
    Jz = -1j * omega * sigma[:, None] * a_q + sigma[:, None] * u_el[:, None]
    return Jz, weights, sigma
