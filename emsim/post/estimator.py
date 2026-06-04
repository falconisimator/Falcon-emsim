r"""Zienkiewicz-Zhu (ZZ) error estimator targeting the loss functional.

The recovered (smoothed) nodal field ``B*`` is compared to the raw
piecewise-constant element field ``B_h``. The per-element energy-norm
indicator

    eta_e^2 = integral_e (1/mu) |B* - B_h|^2 dA

is large where the field is under-resolved -- conductor surfaces and skin
layers -- which is exactly where loss accuracy is determined. The indicators
drive a target element-size field for the next mesh.
"""

from __future__ import annotations

import numpy as np

from emsim.fem import elements
from emsim.fem.assembly import element_material_arrays
from emsim.mesh.gmsh_backend import KELVIN_TAG
from emsim.post.fields import element_B, nodal_B
from emsim.results import Solution


def zz_error_indicators(solution: Solution) -> np.ndarray:
    """Per-element ZZ energy-norm error indicators ``eta_e``, shape ``(M,)``.

    Kelvin-disk elements are assigned zero (they represent the exact exterior).
    """
    mesh = solution.mesh
    B_el = element_B(solution)  # (M,2)
    nB = nodal_B(solution)  # (N,2)
    verts = mesh.triangle_vertices()
    _, weights, shape_vals = elements.quadrature_points(verts)  # (M,Q),(Q,3)
    Bstar = np.einsum("qi,mid->mqd", shape_vals, nB[mesh.tris])  # (M,Q,2)
    diff = Bstar - B_el[:, None, :]  # (M,Q,2)
    inv_mu, _ = element_material_arrays(mesh, solution.materials)
    dens = inv_mu[:, None] * (np.abs(diff[:, :, 0]) ** 2 + np.abs(diff[:, :, 1]) ** 2)
    eta2 = (dens * weights).sum(axis=1)
    eta2[mesh.region_tag == KELVIN_TAG] = 0.0
    return np.sqrt(eta2)


def global_error(eta: np.ndarray) -> float:
    """Global energy-norm error estimate sqrt(sum eta_e^2)."""
    return float(np.sqrt(np.sum(eta**2)))


def target_node_sizes(
    solution: Solution,
    eta: np.ndarray,
    h_min: float,
    h_max: float,
    theta: float = 1.0,
) -> np.ndarray:
    """Target mesh size at each node from the ZZ indicators (equidistribution).

    For P1 the energy-norm error scales ~ O(h) per element, so to equidistribute
    the error toward ``theta * eta_rms`` we scale each element size by
    ``(eta_target / eta_e)``, clamped to ``[h_min, h_max]``, then average onto
    nodes.
    """
    mesh = solution.mesh
    phys = mesh.region_tag != KELVIN_TAG
    area = mesh.areas()
    h_e = np.sqrt(2.0 * area)  # characteristic element size
    eta_rms = np.sqrt(np.mean(eta[phys] ** 2))
    eta_target = max(theta * eta_rms, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = eta_target / np.maximum(eta, 1e-300)
    h_new = np.clip(h_e * scale, h_min, h_max)

    node_h = np.full(mesh.num_nodes, h_max, dtype=np.float64)
    cnt = np.zeros(mesh.num_nodes, dtype=np.float64)
    acc = np.zeros(mesh.num_nodes, dtype=np.float64)
    for k in range(3):
        np.add.at(acc, mesh.tris[phys][:, k], h_new[phys])
        np.add.at(cnt, mesh.tris[phys][:, k], 1.0)
    good = cnt > 0
    node_h[good] = acc[good] / cnt[good]
    return node_h
